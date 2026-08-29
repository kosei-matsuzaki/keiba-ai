"""CLI: Backtest evaluation — NDCG, hit rates, and ROI.

Usage:
    uv run python -m ai.evaluation.backtest --model <path>
                                           [--db PATH]
                                           [--start YYYY-MM-DD]
                                           [--end YYYY-MM-DD]
                                           [--baseline favorite]
                                           [--win-ev-threshold 1.1]
                                           [--place-ev-threshold 1.05]
                                           [--exclude-top-rank 0]
                                           [--min-popularity N]
                                           [--max-popularity N]

When --baseline favorite is given, the same dataset is also evaluated under
the dumb "always bet on the lowest-odds horse" strategy, and the output
becomes a nested dict {model: {...}, baseline_favorite: {...}, delta: {...}}.

Betting filters (--exclude-top-rank / --min-popularity / --max-popularity)
apply only to the model side. analyze_place_bets.py で発見した
「rank 1-2 は payback 0.10、人気 4-12 帯は payback 1.8-3.1」という構造
に対し、CLI から戦略チューニングできるようにする。Baseline (favorite)
側は常に 1 番人気に賭ける性質上、これらフィルタは適用しない。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from sqlalchemy import select

from ai.core.labels import assign_relevance
from ai.core.probabilities import plackett_luce_place_prob
from ai.inference.confidence import is_place_worth_buying, pick_confidence
from ai.inference.predict import predict_race
from ai.model.registry import load_model_full
from core.paths import db_path
from core.settings_store import SettingsStore, resolve_model_path
from db.models import ModelRun
from db.session import make_engine, session_scope
from features.builder import build_training_frame

if TYPE_CHECKING:
    from ai.model.registry import ModelBundle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

#: 単勝で買うオッズの下限。実運用の `settings.win_min_odds` と同じ役割。
#: **EV 閾値ではない** — 既定の "top1" ルールは EV を使わない。
WIN_MIN_ODDS = 1.1

#: 旧ルール (--win-bet-rule ev / --place-bet-rule ev) 専用の EV 閾値。
#: **実運用では EV を全券種で廃止済み**で、これらは「昔の買い方と比べる」
#: 分析用途にだけ残している。既定のルールでは一切参照されない。
LEGACY_WIN_EV_THRESHOLD = 1.1
LEGACY_PLACE_EV_THRESHOLD = 1.05

# Bootstrap CI metrics — keys listed here get `_ci_low` / `_ci_high` companions
# in the returned metrics dict when bootstrap is enabled.
_BOOTSTRAP_METRIC_KEYS = ("ndcg1", "ndcg3", "top1_hit", "place_hit", "payback_win", "payback_place")


def _bet_excluded(
    rank: int,
    row: pd.Series,
    exclude_top_rank: int,
    min_popularity: int | None,
    max_popularity: int | None,
) -> bool:
    """Return True if the horse should be skipped by the betting filters.

    `rank` is 0-indexed from the top of the model's predicted order, so
    `rank < exclude_top_rank` removes the model's top picks. Popularity
    filters are inclusive ([min, max]); NaN popularity is treated as
    excluded whenever any popularity bound is set.
    """
    if exclude_top_rank > 0 and rank < exclude_top_rank:
        return True
    if min_popularity is not None or max_popularity is not None:
        pop = row.get("popularity")
        if pop is None or pd.isna(pop):
            return True
        pop_int = int(pop)
        if min_popularity is not None and pop_int < min_popularity:
            return True
        if max_popularity is not None and pop_int > max_popularity:
            return True
    return False


# 市場オッズ由来の 3 着内確率を PL (Harville) で出すときの冪。1.0 だと本命の
# 3 着内確率を過大評価し、穴馬の推定複勝オッズが実払戻より甘くなる (実測: 実オッズ
# 6.0 以上の帯で推定/実際 = 1.37)。implied^MARKET_PLACE_POWER を正規化してから PL に
# 通すことで補正する (Henery / discounted Harville と同じ形)。valid 1,465 レースで
# 3 着内の log-loss を最小化して 0.85 (lam=1.0 の 0.4111 → 0.4077)。
MARKET_PLACE_POWER = 0.85


def _estimate_place_odds(
    race_frame: pd.DataFrame,
    k: int = 3,
    takeout: float = 0.20,
    market_power: float = MARKET_PLACE_POWER,
) -> dict[str, float]:
    """Estimate each horse's 複勝 decimal odds from PRE-RACE win odds (leak-free).

    The legacy place-EV path used `min(confirmed post-race place payouts)` as the
    odds for every horse — a lookahead leak (the bet decision peeked at the race
    result). This estimates place odds using only entries.odds_win (known before
    the off): win odds → implied win prob → Plackett-Luce P(top-k) → fair place
    odds discounted by the place-pool takeout.

    Returns {horse_id: estimated_place_decimal_odds}. Horses without a usable
    odds_win are omitted. Returns {} when fewer than 2 horses have odds.
    """
    sub = race_frame[["horse_id", "odds_win"]].copy()
    sub = sub[sub["odds_win"].notna() & (sub["odds_win"] > 0)]
    if len(sub) < 2:
        return {}
    implied = 1.0 / sub["odds_win"].to_numpy(dtype=np.float64)
    implied = implied / implied.sum()  # strip the win-pool overround
    # Harville バイアス補正 (MARKET_PLACE_POWER)
    implied = np.power(implied, float(market_power))
    implied = implied / implied.sum()
    scores = np.log(np.clip(implied, 1e-12, None))
    p_top_k = np.clip(plackett_luce_place_prob(scores, k=k), 1e-6, 1.0)
    est_odds = (1.0 - takeout) / p_top_k
    return dict(zip(sub["horse_id"].to_numpy(), est_odds, strict=False))


def _parse_payout_place(json_str: str | None) -> dict[int, int]:
    """Parse payout_place JSON string into {finish_position: payout_yen} dict.

    Expected format: '{"1": 120, "2": 240, "3": 180}' where values are
    payout per 100 yen bet (Japanese convention).
    Returns empty dict if json_str is None or unparsable.
    """
    if not json_str:
        return {}
    try:
        raw = json.loads(json_str)
        return {int(k): int(v) for k, v in raw.items()}
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}


def _bootstrap_ci(
    per_race: dict[str, np.ndarray],
    iters: int,
    seed: int,
    ci: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Race-level bootstrap CI for ndcg / hit-rate / payback metrics.

    Per-race resampling preserves the natural noise unit (one race = one
    independent draw). For payback metrics, the resampled payback is
    `sum(payout) / sum(invested)` across the resampled races so that the
    CI accounts for both the rate and the bet-volume variance.

    Args:
        per_race: dict with arrays of equal length N (one entry per race):
            ndcg1, ndcg3, top1_hit, place_hit,
            win_invested, win_payout, place_invested, place_payout.
        iters: bootstrap iteration count. Must be > 0.
        seed: RNG seed for reproducibility.
        ci: confidence level in (0, 1). Default 0.95 → 2.5%/97.5% percentiles.

    Returns:
        dict mapping metric key → (lower, upper). When all resampled
        iterations yield NaN (e.g. invested==0 in every bootstrap sample),
        the bounds are NaN.
    """
    n = len(per_race["ndcg1"])
    if n == 0 or iters <= 0:
        return {k: (float("nan"), float("nan")) for k in _BOOTSTRAP_METRIC_KEYS}

    rng = np.random.default_rng(seed)
    # idx: shape (iters, n) — each row is a bootstrap sample of race indices
    idx = rng.integers(0, n, size=(iters, n))

    samples: dict[str, np.ndarray] = {}
    # Mean-style metrics: average over the resampled races
    for key in ("ndcg1", "ndcg3", "top1_hit", "place_hit"):
        vals = per_race[key]  # shape (n,)
        # vals[idx] → shape (iters, n); mean across axis=1 → shape (iters,)
        samples[key] = vals[idx].mean(axis=1)

    # Payback metrics: sum(payout) / sum(invested) over the resampled races,
    # NaN when sum(invested) == 0.
    for kind in ("win", "place"):
        invested = per_race[f"{kind}_invested"][idx].sum(axis=1)
        payout = per_race[f"{kind}_payout"][idx].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            samples[f"payback_{kind}"] = np.where(invested > 0, payout / invested, np.nan)

    alpha = (1.0 - ci) / 2.0
    lo_p, hi_p = alpha * 100.0, (1.0 - alpha) * 100.0
    out: dict[str, tuple[float, float]] = {}
    for key, vals in samples.items():
        if np.all(np.isnan(vals)):
            out[key] = (float("nan"), float("nan"))
        else:
            out[key] = (
                float(np.nanpercentile(vals, lo_p)),
                float(np.nanpercentile(vals, hi_p)),
            )
    return out


def _add_ci_fields(metrics: dict, ci_map: dict[str, tuple[float, float]]) -> None:
    """Merge bootstrap CI bounds into a flat metrics dict.

    For each metric key, adds `<key>_ci_low` and `<key>_ci_high` fields.
    Easier to consume from the Dashboard / persisted JSON than a nested dict.
    """
    for key, (lo, hi) in ci_map.items():
        metrics[f"{key}_ci_low"] = lo
        metrics[f"{key}_ci_high"] = hi


def _evaluate_favorite_baseline(
    frame: pd.DataFrame,
    *,
    bootstrap_iters: int = 0,
    bootstrap_seed: int = 42,
) -> dict:
    """Evaluate the 'always bet on the lowest-odds horse' baseline.

    Strategy: per race, identify the horse with the lowest odds_win and bet
    100 yen on win + 100 yen on place. Skips races where no horse has a
    valid odds_win.

    Returns metrics with the same keys as `evaluate()` for direct comparison.
    """
    ndcg1_list: list[float] = []
    ndcg3_list: list[float] = []
    top1_hits: list[int] = []
    place_hits: list[int] = []

    # Per-race accumulators (each entry = one race's stake/payout for the
    # favorite bet). Kept alongside the running totals so bootstrap can
    # resample by race.
    per_race_win_invested: list[float] = []
    per_race_win_payout: list[float] = []
    per_race_place_invested: list[float] = []
    per_race_place_payout: list[float] = []

    win_bets = 0
    win_invested = 0.0
    win_gross_payout = 0.0
    place_bets = 0
    place_invested = 0.0
    place_gross_payout = 0.0

    for race_id in frame["race_id"].unique():
        race_frame = frame[frame["race_id"] == race_id].copy()
        if len(race_frame) < 2:
            continue

        valid = race_frame.dropna(subset=["odds_win"])
        if valid.empty:
            continue

        # NDCG: score = -odds_win so the lowest-odds horse ranks #1
        true_rel = race_frame["relevance"].values.reshape(1, -1)
        score_map = {row["horse_id"]: -float(row["odds_win"]) for _, row in valid.iterrows()}
        # Horses without odds_win get a very small score so they rank last
        pred_scores = np.array(
            [score_map.get(h, -1e10) for h in race_frame["horse_id"]]
        ).reshape(1, -1)
        ndcg1_list.append(float(ndcg_score(true_rel, pred_scores, k=1)))
        ndcg3_list.append(float(ndcg_score(true_rel, pred_scores, k=3)))

        # The favourite = lowest odds_win
        favourite = valid.sort_values("odds_win").iloc[0]
        fav_finish = favourite.get("finish_position")
        fav_finish_int = (
            int(fav_finish)
            if fav_finish is not None
            and not pd.isna(fav_finish)
            and float(fav_finish) == int(fav_finish)
            else None
        )

        top1_hits.append(1 if fav_finish_int == 1 else 0)
        place_hits.append(1 if fav_finish_int is not None and fav_finish_int <= 3 else 0)

        # Always bet 100 on win on the favourite
        win_bets += 1
        win_invested += 100
        race_win_payout = (
            float(favourite["odds_win"]) * 100 if fav_finish_int == 1 else 0.0
        )
        win_gross_payout += race_win_payout
        per_race_win_invested.append(100.0)
        per_race_win_payout.append(race_win_payout)

        # Always bet 100 on place on the favourite (when payout_place is known)
        payout_place_raw: str | None = None
        if "payout_place" in race_frame.columns:
            vals = race_frame["payout_place"].dropna()
            if not vals.empty:
                payout_place_raw = vals.iloc[0]
        payout_place_map = _parse_payout_place(payout_place_raw)
        if payout_place_map:
            place_bets += 1
            place_invested += 100
            race_place_payout = (
                float(payout_place_map[fav_finish_int])
                if fav_finish_int in payout_place_map
                else 0.0
            )
            place_gross_payout += race_place_payout
            per_race_place_invested.append(100.0)
            per_race_place_payout.append(race_place_payout)
        else:
            # No place data → no bet, but keep arrays aligned for bootstrap.
            per_race_place_invested.append(0.0)
            per_race_place_payout.append(0.0)

    n_races = len(ndcg1_list)
    out = {
        "n_races": n_races,
        "ndcg1": float(np.mean(ndcg1_list)) if ndcg1_list else float("nan"),
        "ndcg3": float(np.mean(ndcg3_list)) if ndcg3_list else float("nan"),
        "top1_hit": float(np.mean(top1_hits)) if top1_hits else float("nan"),
        "place_hit": float(np.mean(place_hits)) if place_hits else float("nan"),
        "win_bets": win_bets,
        "win_invested": win_invested,
        "win_gross_payout": win_gross_payout,
        "payback_win": (win_gross_payout / win_invested) if win_invested > 0 else float("nan"),
        "place_bets": place_bets,
        "place_invested": place_invested,
        "place_gross_payout": place_gross_payout,
        "payback_place": (
            (place_gross_payout / place_invested) if place_invested > 0 else float("nan")
        ),
    }

    if bootstrap_iters > 0 and n_races > 0:
        per_race_arr = {
            "ndcg1": np.asarray(ndcg1_list, dtype=np.float64),
            "ndcg3": np.asarray(ndcg3_list, dtype=np.float64),
            "top1_hit": np.asarray(top1_hits, dtype=np.float64),
            "place_hit": np.asarray(place_hits, dtype=np.float64),
            "win_invested": np.asarray(per_race_win_invested, dtype=np.float64),
            "win_payout": np.asarray(per_race_win_payout, dtype=np.float64),
            "place_invested": np.asarray(per_race_place_invested, dtype=np.float64),
            "place_payout": np.asarray(per_race_place_payout, dtype=np.float64),
        }
        ci_map = _bootstrap_ci(per_race_arr, bootstrap_iters, bootstrap_seed)
        _add_ci_fields(out, ci_map)
        out["bootstrap_iters"] = int(bootstrap_iters)

    return out


def _delta_metrics(model: dict, baseline: dict) -> dict:
    """Compute model − baseline for headline comparison metrics.

    NaN on either side propagates to NaN; integer-only fields (counts) are skipped.
    """
    keys = ["ndcg1", "ndcg3", "top1_hit", "place_hit", "payback_win", "payback_place"]
    out: dict[str, float] = {}
    for k in keys:
        m = model.get(k)
        b = baseline.get(k)
        if m is None or b is None or pd.isna(m) or pd.isna(b):
            out[k] = float("nan")
        else:
            out[k] = float(m) - float(b)
    return out


def _persist_metrics_to_model_run(
    engine, model_path: Path, model_metrics: dict
) -> bool:
    """Merge `model_metrics` into the matching ModelRun's metrics_json.

    Match strategy: model_path strict equal first, then by basename
    (timestamp like "20260502-224015") to be robust to slash differences
    between Windows / WSL or relative vs absolute paths.

    Returns True if a row was updated, False if no matching ModelRun found.
    """
    requested = str(Path(model_path).resolve())
    target_name = Path(model_path).name

    with session_scope(engine) as session:
        # Try exact resolved-path match first
        run = session.scalar(
            select(ModelRun).where(ModelRun.model_path == requested)
        )
        if run is None:
            # Fall back to basename (timestamp) match — robust across OS
            for candidate in session.scalars(select(ModelRun)).all():
                if Path(candidate.model_path).name == target_name:
                    run = candidate
                    break
        if run is None:
            log.warning("No ModelRun matched model_path=%s; skip persist", model_path)
            return False

        existing = json.loads(run.metrics_json) if run.metrics_json else {}
        merged = {**existing, **model_metrics}
        run.metrics_json = json.dumps(merged, ensure_ascii=False)
        log.info(
            "Persisted evaluation metrics into ModelRun id=%d (merged keys: %s)",
            run.id,
            sorted(set(model_metrics.keys()) - set(existing.keys())),
        )
        return True


def evaluate(
    model_path: Path,
    db: Path | None = None,
    start: str | None = None,
    end: str | None = None,
    baseline: str | None = None,
    persist: bool = False,
    *,
    win_min_odds: float = WIN_MIN_ODDS,
    legacy_win_ev_threshold: float = LEGACY_WIN_EV_THRESHOLD,
    win_bet_rule: str = "top1",
    place_bet_rule: str = "topk",
    place_top_k: int = 1,
    legacy_place_ev_threshold: float = LEGACY_PLACE_EV_THRESHOLD,
    probability_model_path: Path | None = None,
    place_min_confidence: float = 0.30,
    exclude_top_rank: int = 0,
    min_popularity: int | None = None,
    max_popularity: int | None = None,
    bootstrap_iters: int = 0,
    bootstrap_seed: int = 42,
    bundle: ModelBundle | None = None,
    place_odds_mode: str = "estimated",
    place_takeout: float = 0.20,
    market_place_power: float = MARKET_PLACE_POWER,
) -> dict:
    """Run backtest evaluation and return metrics dict.

    When `baseline` is None (default), returns the flat model metrics dict
    (backwards compatible). When baseline=='favorite', returns
    {"model": {...}, "baseline_favorite": {...}, "delta": {...}}.

    `win_bet_rule` は単勝の買い方:
      - "top1" (既定) … **モデル 1 位の馬**を `odds > win_min_odds` のときだけ買う。
        これは**オッズの下限**であって EV 閾値ではない (実運用と同じ)。
      - "ev"          … `win_prob × odds > legacy_win_ev_threshold` の馬すべて。
        **実運用では廃止済み**で、昔の買い方と比べるための分析用途。

    温度を NLL 較正すると "ev" は平坦な確率 × 大穴オッズで偽 EV を量産して回収率が
    落ちる (実測 0.698)。旧既定の 0.912 は温度がグリッド端に張り付いて win_prob が
    飽和し、EV 条件が実質「1 位を買う」に退化していた結果だった。その戦略を
    **確率に依存しない形で明示**したのが "top1" で、較正済み確率のもとで
    **0.931 と旧既定を上回る** (test 19ヶ月 5,404 レース実測)。

    `place_bet_rule` は複勝の買い方:
      - "topk" (既定) … **モデル上位 `place_top_k` 頭**を無条件に買う。実測 (test 19ヶ月)
        k=1 で 5,402 点・回収率 **0.887**、k=2 で 0.860、k=3 で 0.837 と k が小さいほど良い。
      - "ev"          … `place_prob × 推定複勝オッズ > legacy_place_ev_threshold`。
        **実運用では廃止済み**で、分析用途にだけ残している。
        43,464 点・回収率 0.654 で、1 番人気ベタ買いの複勝 0.850 にも負ける。

    複勝の EV は単に狂っているのではなく **順序が逆**。実測 (test 19ヶ月) の EV 帯別
    回収率は 0.0-0.9 帯が 0.832 で最良、2.0 以上が 0.573 で最悪と単調減少しており、
    高 EV = 推定オッズの高い穴馬 = 確率もオッズも最も過大評価される帯、という構造。
    温度・冪 (Harville 補正) のような単調変換では直らない。

    `probability_model_path` は**確率専用モデル** (proper scoring rule で学習)。
    指定すると、実運用と同じく **AI の本命に対するそのモデルの確率が
    `place_min_confidence` 未満のレースでは複勝を買わない**。指定しないと
    複勝を全レースで買う = 実運用が確率モデルを使っている場合、**評価と本番が
    別物になる**。Dashboard の KPI は `--persist` が書いた値を読むので、
    ここを揃えないと画面の数字が「利用者が実際に得る数字」でなくなる
    (2026-08-24 に同じ型のズレを一度直している)。

    `persist=True` で評価結果を model_runs.metrics_json に merge する
    (Dashboard 側 metrics endpoint がこの値を読む)。

    Betting filters:
      - `exclude_top_rank=N` → モデル予測上位 N 頭を bet 対象から除外
        (analyze_place_bets で本命 rank 1 が payback 0.10 と判明したため)
      - `min_popularity=K` / `max_popularity=K` → 人気が K 番より下/上を除外
        (1 = 1 番人気)。NaN popularity はフィルタ有効時に常に除外

    Bootstrap CI (`bootstrap_iters > 0`):
      - race 単位の置換抽出で ndcg1 / ndcg3 / top1_hit / place_hit /
        payback_win / payback_place の 95% 信頼区間を計算し、
        `<metric>_ci_low` / `<metric>_ci_high` キーで返す。
      - `bootstrap_seed` で再現可能 (default 42)。
      - `baseline='favorite'` 指定時は baseline 側にも同じ iter/seed で
        CI を付与する (左右対称な比較のため)。
    """
    resolved_db = db or db_path()
    engine = make_engine(resolved_db)

    # Pre-built bundle override (used by sweep callers that need to evaluate an
    # in-memory bundle without writing it to disk). When absent, fall back to
    # the on-disk artifacts.
    if bundle is None:
        bundle = load_model_full(model_path)

    log.info("Building evaluation frame from %s", resolved_db)
    with session_scope(engine) as session:
        frame = build_training_frame(session, train_start=start, train_end=end)

    if frame.empty:
        log.warning("No evaluation data found.")
        return {}

    frame["relevance"] = frame["finish_position"].map(assign_relevance)

    # Per-race metrics
    ndcg1_list: list[float] = []
    ndcg3_list: list[float] = []
    top1_hits: list[int] = []
    place_hits: list[int] = []

    # Per-race stake/payout (one entry per *evaluated* race, same length as the
    # ndcg lists). Bootstrap CI resamples on this axis. Races with no triggered
    # bet contribute 0 invested / 0 payout — required so the resampled index
    # stays aligned across all per-race arrays.
    per_race_win_invested: list[float] = []
    per_race_win_payout: list[float] = []
    per_race_place_invested: list[float] = []
    per_race_place_payout: list[float] = []

    # 確率専用モデル。指定されていれば、実運用と同じく **AI の本命に対する
    # そのモデルの確率がしきい値未満のレースでは複勝を買わない**。揃えないと
    # 「評価と本番が別物」になり、Dashboard の KPI が実際に得る数字でなくなる。
    prob_bundle = None
    if probability_model_path is not None:
        try:
            prob_bundle = load_model_full(probability_model_path)
            log.info("Confidence model loaded from %s", probability_model_path)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "confidence model load failed (%s): %s — 複勝は全レースで買う",
                probability_model_path, exc,
            )
    n_place_skipped = 0

    # Betting simulation — payback rate convention (回収率): gross_payout / invested
    # 1.00 = break-even, 1.10 = 10% profit, 0.80 = 20% loss
    win_bets = 0
    win_gross_payout = 0.0  # 払戻金合計（賭け金は含まない）
    win_invested = 0.0      # 賭け金合計

    # Place betting simulation (複勝)
    place_bets = 0
    place_gross_payout = 0.0
    place_invested = 0.0

    race_ids = frame["race_id"].unique()
    # arch-3 (history_feat_dim>0) は推論時に過去走系列を DB から引くため、
    # ループ中セッションを開いたままにする。session を渡さないと履歴が zero に
    # degrade し、学習時 (metrics_json) と別モデルを評価することになる。
    with session_scope(engine) as pred_session:
        for race_id in race_ids:
            race_frame = frame[frame["race_id"] == race_id].copy()
            if len(race_frame) < 2:
                continue

            # Per-race stake/payout accumulators (added to the global running
            # totals AND to the per-race arrays for bootstrap).
            race_win_invested = 0.0
            race_win_payout = 0.0
            race_place_invested = 0.0
            race_place_payout = 0.0

            # bundle 経由で推論 (NN)
            preds = predict_race(bundle, race_frame, session=pred_session)
            # Merge actual finish positions + popularity (needed for betting filters)
            actual_cols = ["horse_id", "finish_position", "odds_win", "relevance"]
            if "popularity" in race_frame.columns:
                actual_cols.append("popularity")
            actual = race_frame[actual_cols].copy()
            preds = preds.merge(actual, on="horse_id", how="left")

            # NDCG
            true_rel = race_frame["relevance"].values.reshape(1, -1)
            # Align scores to same order as race_frame
            score_map = dict(zip(preds["horse_id"], preds["score"], strict=False))
            pred_scores = np.array([score_map.get(h, 0.0) for h in race_frame["horse_id"]]).reshape(1, -1)
            ndcg1_list.append(float(ndcg_score(true_rel, pred_scores, k=1)))
            ndcg3_list.append(float(ndcg_score(true_rel, pred_scores, k=3)))

            # Top-1 hit: does the horse ranked #1 by model actually finish 1st?
            top_horse = preds.iloc[0]  # sorted by score desc
            top1_hits.append(1 if top_horse["finish_position"] == 1 else 0)

            # Place hit: is at least one of top-3 model picks in actual top-3?
            top3_horses = set(preds.iloc[:3]["horse_id"])
            actual_top3 = set(
                actual[actual["finish_position"].notna() & (actual["finish_position"] <= 3)]["horse_id"]
            )
            place_hits.append(1 if top3_horses & actual_top3 else 0)

            # Win betting: "ev" は win_prob × odds > 閾値、"top1" はモデル 1 位のみを
            # odds > 閾値 で買う。どちらも rank/popularity フィルタを通った馬が対象。
            for rank, (_, row) in enumerate(preds.iterrows()):
                if _bet_excluded(rank, row, exclude_top_rank, min_popularity, max_popularity):
                    continue
                odds = row.get("odds_win")
                if odds is None or pd.isna(odds):
                    continue
                if win_bet_rule == "top1":
                    take = rank == 0 and odds > win_min_odds
                else:
                    take = row["win_prob"] * odds > legacy_win_ev_threshold
                if take:
                    # デプロイ (ai.betting.strategy.assign_flat_stakes) と同じ 1 点定額。
                    # Kelly (資金比率) は賭け金決定から廃止済みで、評価側にだけ残すと
                    # 「アプリが実行できない戦略」を測ることになるため置かない。
                    bet_size = 100
                    win_bets += 1
                    win_invested += bet_size
                    race_win_invested += bet_size
                    if row.get("finish_position") == 1:
                        win_gross_payout += odds * bet_size
                        race_win_payout += odds * bet_size

            # 複勝の確信度フィルタ (実運用と同じ)。確率モデルが AI の本命に与える
            # 確率がしきい値未満なら、このレースの複勝は買わない。
            if prob_bundle is not None and not preds.empty:
                conf = pick_confidence(
                    prob_bundle, race_frame, preds.iloc[0]["horse_id"], session=session
                )
                if not is_place_worth_buying(conf, place_min_confidence):
                    n_place_skipped += 1
                    per_race_place_invested.append(race_place_invested)
                    per_race_place_payout.append(race_place_payout)
                    continue

            # Place betting (複勝): requires payout_place data on the race frame
            # race_frame may carry payout_place if the training frame includes it.
            # Look it up from the race_frame column if present.
            payout_place_raw: str | None = None
            if "payout_place" in race_frame.columns:
                vals = race_frame["payout_place"].dropna()
                if not vals.empty:
                    payout_place_raw = vals.iloc[0]

            payout_place_map = _parse_payout_place(payout_place_raw)
            if payout_place_map:
                # Odds used for the place EV *decision*:
                #   "estimated" (default, leak-free): per-horse 複勝 odds estimated
                #     from PRE-RACE win odds via Plackett-Luce.
                #   "min_payout" (legacy): min(confirmed post-race payout) shared by
                #     all horses — a lookahead leak kept only for back-compat / A-B.
                # Settlement (realized payout) always uses the confirmed payout_place.
                if place_odds_mode == "estimated":
                    est_place_odds = _estimate_place_odds(
                        race_frame, takeout=place_takeout,
                        market_power=market_place_power,
                    )
                    min_odds = None  # not used in this mode
                else:
                    min_odds = min(payout_place_map.values()) / 100.0
                    est_place_odds = None

                for rank, (_, row) in enumerate(preds.iterrows()):
                    if _bet_excluded(
                        rank, row, exclude_top_rank, min_popularity, max_popularity
                    ):
                        continue
                    if est_place_odds is not None:
                        place_odds = est_place_odds.get(row["horse_id"])
                        if place_odds is None:
                            continue  # no pre-race odds → cannot price this bet
                    else:
                        place_odds = min_odds
                    if place_bet_rule == "topk":
                        take_place = rank < place_top_k
                    else:
                        take_place = (
                            row["place_prob"] * place_odds > legacy_place_ev_threshold
                        )
                    if take_place:
                        place_bet_size = 100
                        place_bets += 1
                        place_invested += place_bet_size
                        race_place_invested += place_bet_size
                        finish_pos = row.get("finish_position")
                        # 同着（finish_position が小数 = 1.5/2.5 等）は日本競馬で複勝対象外（返還）
                        # のため整数着順のみカウントし、複勝 ROI を過大評価しないようにする。
                        if (
                            finish_pos is not None
                            and not pd.isna(finish_pos)
                            and float(finish_pos) == int(finish_pos)
                            and int(finish_pos) in payout_place_map
                        ):
                            race_payout = payout_place_map[int(finish_pos)] * (place_bet_size / 100)
                            place_gross_payout += race_payout
                            race_place_payout += race_payout

            per_race_win_invested.append(race_win_invested)
            per_race_win_payout.append(race_win_payout)
            per_race_place_invested.append(race_place_invested)
            per_race_place_payout.append(race_place_payout)

    n_races = len(ndcg1_list)
    metrics = {
        "n_races": n_races,
        "ndcg1": float(np.mean(ndcg1_list)) if ndcg1_list else float("nan"),
        "ndcg3": float(np.mean(ndcg3_list)) if ndcg3_list else float("nan"),
        "top1_hit": float(np.mean(top1_hits)) if top1_hits else float("nan"),
        # 上位 3 推奨のうち少なくとも 1 頭が実際に 3 着以内に入ったレース割合
        "place_hit": float(np.mean(place_hits)) if place_hits else float("nan"),
        "win_bets": win_bets,
        "win_invested": win_invested,
        "win_gross_payout": win_gross_payout,
        # 回収率 = 払戻金合計 / 賭け金合計（1.00 が損益分岐）
        "payback_win": (win_gross_payout / win_invested) if win_invested > 0 else float("nan"),
        # 複勝回収率
        "place_bets": place_bets,
        "place_invested": place_invested,
        "place_gross_payout": place_gross_payout,
        "payback_place": (
            (place_gross_payout / place_invested) if place_invested > 0 else float("nan")
        ),
        # Record the betting filter params so that persisted metrics_json /
        # CLI JSON dump explains under what strategy the numbers were produced.
        "win_min_odds": float(win_min_odds),
        "win_bet_rule": win_bet_rule,

        "place_bet_rule": place_bet_rule,
        # 確率モデルを使ったか (使うと複勝の対象レースが減るので、後から
        # 「どの条件で測った数字か」を判別できるようにする)
        "probability_model": (
            Path(probability_model_path).name if probability_model_path else None
        ),
        "place_min_confidence": (
            place_min_confidence if probability_model_path else None
        ),
        "n_place_skipped": n_place_skipped,
        "place_top_k": int(place_top_k),
        "exclude_top_rank": int(exclude_top_rank),
        "min_popularity": min_popularity,
        "max_popularity": max_popularity,
        "bet_sizing": "flat",
        "place_odds_mode": place_odds_mode,
        "place_takeout": float(place_takeout) if place_odds_mode == "estimated" else None,
        "market_place_power": float(market_place_power),
    }

    if bootstrap_iters > 0 and n_races > 0:
        per_race_arr = {
            "ndcg1": np.asarray(ndcg1_list, dtype=np.float64),
            "ndcg3": np.asarray(ndcg3_list, dtype=np.float64),
            "top1_hit": np.asarray(top1_hits, dtype=np.float64),
            "place_hit": np.asarray(place_hits, dtype=np.float64),
            "win_invested": np.asarray(per_race_win_invested, dtype=np.float64),
            "win_payout": np.asarray(per_race_win_payout, dtype=np.float64),
            "place_invested": np.asarray(per_race_place_invested, dtype=np.float64),
            "place_payout": np.asarray(per_race_place_payout, dtype=np.float64),
        }
        ci_map = _bootstrap_ci(per_race_arr, bootstrap_iters, bootstrap_seed)
        _add_ci_fields(metrics, ci_map)
        metrics["bootstrap_iters"] = int(bootstrap_iters)
        metrics["bootstrap_seed"] = int(bootstrap_seed)

    log.info("Evaluation metrics: %s", metrics)

    if persist:
        # Dashboard が読みやすいよう、top-level に flat な model 系キー
        # (top1_hit / payback_win 等) を merge する。baseline mode でも
        # 比較用 baseline / delta は混ぜず、model 側のみ保存。
        _persist_metrics_to_model_run(engine, model_path, metrics)

    if baseline == "favorite":
        baseline_metrics = _evaluate_favorite_baseline(
            frame,
            bootstrap_iters=bootstrap_iters,
            bootstrap_seed=bootstrap_seed,
        )
        log.info("Baseline (favorite) metrics: %s", baseline_metrics)
        return {
            "model": metrics,
            "baseline_favorite": baseline_metrics,
            "delta": _delta_metrics(metrics, baseline_metrics),
        }

    return metrics


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate keiba-ai model via backtest")
    parser.add_argument("--model", type=Path, required=True, help="Path to model directory")
    parser.add_argument("--db", type=Path, default=None, help="Path to SQLite DB")
    parser.add_argument("--start", default=None, help="Evaluation start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Evaluation end date YYYY-MM-DD")
    parser.add_argument(
        "--baseline",
        choices=["favorite"],
        default=None,
        help="Also evaluate a baseline strategy alongside the model and report deltas",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help=(
            "Merge the evaluation metrics into the matching model_runs row's "
            "metrics_json so that the Dashboard's MetricCard picks them up."
        ),
    )
    parser.add_argument(
        "--win-min-odds",
        type=float,
        default=WIN_MIN_ODDS,
        help=f"単勝で買うオッズの下限 (既定 {WIN_MIN_ODDS})。EV 閾値ではない。",
    )
    parser.add_argument(
        "--place-ev-threshold",
        type=float,
        default=LEGACY_PLACE_EV_THRESHOLD,
        help=(
            f"--place-bet-rule ev のときだけ使う EV 閾値 (既定 "
            f"{LEGACY_PLACE_EV_THRESHOLD})。実運用では廃止済み。"
        ),
    )
    parser.add_argument(
        "--probability-model", type=Path, default=None,
        help=(
            "確率専用モデルのディレクトリ。指定すると実運用と同じく、AI の本命に対する"
            "そのモデルの確率が --place-min-confidence 未満のレースで複勝を買わない。"
            "**未指定なら settings.json の probability_model_path を使う** "
            "(評価と本番がズレないように既定で揃える)。'none' で明示的に無効化。"
        ),
    )
    parser.add_argument(
        "--place-min-confidence", type=float, default=None,
        help="複勝を買う確信度の下限 (既定: settings.json の place_min_confidence)",
    )
    parser.add_argument(
        "--exclude-top-rank",
        type=int,
        default=0,
        help=(
            "Skip the model's top-N predicted horses when betting "
            "(0 = no exclusion). E.g. 2 removes ranks 1-2."
        ),
    )
    parser.add_argument(
        "--min-popularity",
        type=int,
        default=None,
        help="Lower bound on popularity rank (inclusive); 1 = favourite.",
    )
    parser.add_argument(
        "--max-popularity",
        type=int,
        default=None,
        help="Upper bound on popularity rank (inclusive).",
    )
    parser.add_argument(
        "--win-bet-rule",
        choices=["ev", "top1"],
        default="top1",
        help=(
            "単勝の買い方。'ev' (既定) は win_prob × odds > --win-ev-threshold の馬すべて。"
            "'top1' はモデル 1 位の馬を odds > --win-ev-threshold のときだけ買う。"
        ),
    )
    parser.add_argument(
        "--place-bet-rule",
        choices=["ev", "topk"],
        default="topk",
        help="複勝の買い方。'topk' (既定) はモデル上位 --place-top-k 頭、'ev' は期待値条件 (旧既定)。",
    )
    parser.add_argument("--place-top-k", type=int, default=1, help="--place-bet-rule topk の k")
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=0,
        help=(
            "Race-level bootstrap iteration count for 95%% CI on ndcg / hit / "
            "payback metrics. 0 (default) = no CI. 1000 is a reasonable choice "
            "for production reports."
        ),
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="RNG seed for bootstrap resampling. Default 42 (reproducible).",
    )
    parser.add_argument(
        "--place-odds-mode",
        choices=["estimated", "min_payout"],
        default="estimated",
        help=(
            "Odds source for the place EV decision. 'estimated' (default, "
            "leak-free) prices each horse's 複勝 odds from pre-race win odds via "
            "Plackett-Luce. 'min_payout' is the legacy min(confirmed payout) — a "
            "lookahead leak, kept for A/B comparison only."
        ),
    )
    parser.add_argument(
        "--place-takeout",
        type=float,
        default=0.20,
        help="Place-pool takeout for the estimated place odds (default 0.20).",
    )
    args = parser.parse_args()

    # 既定は settings.json に合わせる。評価だけ別条件で走ると、Dashboard の KPI が
    # 「利用者が実際に得る数字」でなくなる (2026-08-24 に同じ型のズレを直している)。
    _settings = SettingsStore().load()
    if args.probability_model is not None:
        prob_model_path = (
            None if str(args.probability_model) == "none"
            else resolve_model_path(str(args.probability_model))
        )
    else:
        prob_model_path = resolve_model_path(_settings.get("probability_model_path"))
    place_min_conf = (
        args.place_min_confidence
        if args.place_min_confidence is not None
        else float(_settings.get("place_min_confidence", 0.30))
    )
    if prob_model_path is not None:
        print(f"確率モデル: {prob_model_path.name} (複勝は確信度 {place_min_conf:.2f} 以上)")

    metrics = evaluate(
        model_path=args.model,
        db=args.db,
        start=args.start,
        end=args.end,
        baseline=args.baseline,
        persist=args.persist,
        win_min_odds=args.win_min_odds,
        win_bet_rule=args.win_bet_rule,
        place_bet_rule=args.place_bet_rule,
        place_top_k=args.place_top_k,
        legacy_place_ev_threshold=args.place_ev_threshold,
        probability_model_path=prob_model_path,
        place_min_confidence=place_min_conf,
        exclude_top_rank=args.exclude_top_rank,
        min_popularity=args.min_popularity,
        max_popularity=args.max_popularity,
        bootstrap_iters=args.bootstrap_iters,
        bootstrap_seed=args.bootstrap_seed,
        place_odds_mode=args.place_odds_mode,
        place_takeout=args.place_takeout,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
