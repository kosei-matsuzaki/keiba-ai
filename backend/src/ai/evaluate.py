"""CLI: Backtest evaluation — NDCG, hit rates, and ROI.

Usage:
    uv run python -m ai.evaluate --model <path>
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

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from sqlalchemy import select

from ai.labels import assign_relevance
from ai.predict import predict_race
from ai.registry import load_model_full
from core.paths import db_path
from db.models import ModelRun  # noqa: F401
from db.session import make_engine, session_scope
from features.builder import build_training_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

WIN_EV_THRESHOLD = 1.1   # Expected value threshold for win bet
PLACE_EV_THRESHOLD = 1.05  # Expected value threshold for place bet

# Bootstrap CI metrics — keys listed here get `_ci_low` / `_ci_high` companions
# in the returned metrics dict when bootstrap is enabled.
_BOOTSTRAP_METRIC_KEYS = ("ndcg1", "ndcg3", "top1_hit", "place_hit", "payback_win", "payback_place")


def kelly_bet_size(
    win_prob: float,
    odds: float,
    bankroll: float,
    kappa: float = 0.25,
    min_bet: int = 100,
) -> int:
    """Fractional Kelly bet size (100 yen 単位で丸め).

    Fractional Kelly fraction: f = kappa * edge / b
    where edge = win_prob * odds - 1 and b = odds - 1.

    Args:
        win_prob: Estimated win probability.
        odds: Decimal odds (payout per 1 yen bet, e.g. 3.5 means 3.5x return).
        bankroll: Current bankroll in yen.
        kappa: Fractional Kelly coefficient (0 < kappa <= 1). Default 0.25.
        min_bet: Minimum bet size in yen; also the rounding unit. Default 100.

    Returns:
        Bet size in yen, a multiple of min_bet. Returns 0 when edge <= 0.
    """
    edge = win_prob * odds - 1.0
    if edge <= 0:
        return 0
    b = odds - 1.0
    if b <= 0:
        return 0
    fraction = kappa * edge / b
    raw_size = bankroll * fraction
    rounded = int(raw_size / min_bet) * min_bet
    return rounded if rounded >= min_bet else 0


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
    from db.models.model_run import ModelRun  # local import to avoid cycles

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
    win_ev_threshold: float = WIN_EV_THRESHOLD,
    place_ev_threshold: float = PLACE_EV_THRESHOLD,
    exclude_top_rank: int = 0,
    min_popularity: int | None = None,
    max_popularity: int | None = None,
    bet_sizing: str = "fixed",
    kelly_kappa: float = 0.25,
    bankroll: float = 100_000.0,
    bootstrap_iters: int = 0,
    bootstrap_seed: int = 42,
) -> dict:
    """Run backtest evaluation and return metrics dict.

    When `baseline` is None (default), returns the flat model metrics dict
    (backwards compatible). When baseline=='favorite', returns
    {"model": {...}, "baseline_favorite": {...}, "delta": {...}}.

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

    bundle = load_model_full(model_path)
    use_kelly = bet_sizing == "kelly"

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

        # bundle.model_type で GBDT / NN を自動切替
        preds = predict_race(bundle, race_frame)
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

        # Win betting: bet if win_prob × odds_win > win_ev_threshold AND
        # the horse passes the rank/popularity filters.
        for rank, (_, row) in enumerate(preds.iterrows()):
            if _bet_excluded(rank, row, exclude_top_rank, min_popularity, max_popularity):
                continue
            odds = row.get("odds_win")
            if odds is None or pd.isna(odds):
                continue
            ev = row["win_prob"] * odds
            if ev > win_ev_threshold:
                if use_kelly:
                    bet_size = kelly_bet_size(
                        float(row["win_prob"]), float(odds), bankroll,
                        kappa=kelly_kappa,
                    )
                    if bet_size == 0:
                        continue
                else:
                    bet_size = 100
                win_bets += 1
                win_invested += bet_size
                race_win_invested += bet_size
                if row.get("finish_position") == 1:
                    win_gross_payout += odds * bet_size
                    race_win_payout += odds * bet_size

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
            # Determine the minimum payout across 1st/2nd/3rd place for EV calculation.
            # Using min payout gives a conservative estimate of expected return.
            min_payout = min(payout_place_map.values())
            # min_payout is in yen per 100 yen bet, so odds = min_payout / 100
            min_odds = min_payout / 100.0

            for rank, (_, row) in enumerate(preds.iterrows()):
                if _bet_excluded(
                    rank, row, exclude_top_rank, min_popularity, max_popularity
                ):
                    continue
                ev = row["place_prob"] * min_odds
                if ev > place_ev_threshold:
                    if use_kelly:
                        place_bet_size = kelly_bet_size(
                            float(row["place_prob"]), min_odds, bankroll,
                            kappa=kelly_kappa,
                        )
                        if place_bet_size == 0:
                            continue
                    else:
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
        "win_ev_threshold": float(win_ev_threshold),
        "place_ev_threshold": float(place_ev_threshold),
        "exclude_top_rank": int(exclude_top_rank),
        "min_popularity": min_popularity,
        "max_popularity": max_popularity,
        "bet_sizing": bet_sizing,
        "kelly_kappa": float(kelly_kappa) if use_kelly else None,
        "bankroll": float(bankroll) if use_kelly else None,
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
        "--win-ev-threshold",
        type=float,
        default=WIN_EV_THRESHOLD,
        help=f"EV threshold for win bets (default {WIN_EV_THRESHOLD}).",
    )
    parser.add_argument(
        "--place-ev-threshold",
        type=float,
        default=PLACE_EV_THRESHOLD,
        help=f"EV threshold for place bets (default {PLACE_EV_THRESHOLD}).",
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
        "--bet-sizing",
        choices=["fixed", "kelly"],
        default="fixed",
        help=(
            "Bet sizing strategy. 'fixed' (default) bets 100 yen per pick. "
            "'kelly' uses Fractional Kelly formula scaled by --bankroll."
        ),
    )
    parser.add_argument(
        "--kelly-kappa",
        type=float,
        default=0.25,
        help="Fractional Kelly coefficient (0 < kappa <= 1). Default 0.25.",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        default=100_000.0,
        help="Starting bankroll in yen for Kelly bet sizing. Default 100000.",
    )
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
    args = parser.parse_args()

    metrics = evaluate(
        model_path=args.model,
        db=args.db,
        start=args.start,
        end=args.end,
        baseline=args.baseline,
        persist=args.persist,
        win_ev_threshold=args.win_ev_threshold,
        place_ev_threshold=args.place_ev_threshold,
        exclude_top_rank=args.exclude_top_rank,
        min_popularity=args.min_popularity,
        max_popularity=args.max_popularity,
        bet_sizing=args.bet_sizing,
        kelly_kappa=args.kelly_kappa,
        bankroll=args.bankroll,
        bootstrap_iters=args.bootstrap_iters,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
