"""End-to-end backtest simulation for the active model.

Loops over all races in a given window, runs predict + recommendations
+ settle (using actual finish_position and confirmed payouts) and
aggregates ROI / hit-rate by bet_type / race_class / course.

This is the engine behind the Ledger 「シミュレーション」 tab.

Strategy presets translate user-friendly choices to internal Kelly /
EV-threshold parameters:

  conservative:  kelly=0.10, min_ev=1.30  (高 EV 案件のみ少額で)
  balanced:      kelly=0.25, min_ev=1.10  (現行 default)
  aggressive:    kelly=0.40, min_ev=1.00  (positive edge ならどれも賭ける)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from sqlalchemy.orm import Session

from ai.betting.odds import (
    compute_past_race_odds,
    compute_race_odds_with_sources,
)
from ai.betting.strategy import recommend_for_race
from ai.inference.predict import predict_race, predict_race_with_combinations
from ai.model.registry import ModelBundle, load_model_full
from core.bet_types import COMBINATION_BET_TYPES
from core.logging import get_logger
from db.odds_db import init_odds_db, make_odds_engine
from features.builder import build_training_frame

log = get_logger(__name__)


StrategyName = Literal["conservative", "balanced", "aggressive"]

STRATEGY_PRESETS: dict[StrategyName, dict[str, float]] = {
    "conservative": {"kelly_fraction": 0.10, "min_ev": 1.30},
    "balanced":     {"kelly_fraction": 0.25, "min_ev": 1.10},
    "aggressive":   {"kelly_fraction": 0.40, "min_ev": 1.00},
}

# 単勝 / 複勝 / 連系 すべての券種を simulation 対象とする
DEFAULT_BET_TYPES: list[str] = list(COMBINATION_BET_TYPES)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GroupStats:
    """Aggregated stats for a single group (bet_type / race_class / course)."""

    label: str
    n_bets: int = 0
    invested: int = 0
    payout: float = 0.0
    hits: int = 0

    @property
    def payback_rate(self) -> float:
        """payout / invested. 0 when no bets."""
        return float(self.payout) / float(self.invested) if self.invested > 0 else 0.0

    @property
    def hit_rate(self) -> float:
        """hits / n_bets. 0 when no bets."""
        return self.hits / self.n_bets if self.n_bets > 0 else 0.0

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "n_bets": self.n_bets,
            "invested": self.invested,
            "payout": round(self.payout),
            "payback_rate": round(self.payback_rate, 4),
            "hit_rate": round(self.hit_rate, 4),
        }


@dataclass
class BankrollPoint:
    """資産推移グラフ用の 1 日分のスナップショット。

    日跨ぎで複数 race ある場合、最後の race 終了時点の bankroll を採用する。
    """

    date: str           # YYYY-MM-DD
    bankroll: int       # その日の最終 race 後の残高
    invested: int       # その日の累計 stake
    payout: float       # その日の累計 payout
    n_bets: int         # その日の bet 件数


@dataclass
class SimulationResult:
    """Top-level simulation result.

    n_races: total races within window (including ones where no bets fired)
    n_settled_races: subset where finish_position was available (i.e. past)
    final_bankroll: 期間終了時の残高 (= budget + 累計 profit、ただし途中で 0 になれば 0)
    peak_bankroll: 期間中の最高残高
    bankroll_timeseries: 日次の資産推移 (グラフ用)
    """

    window_start: str | None
    window_end: str | None
    model_path: str
    strategy: StrategyName
    budget: int
    n_races: int = 0
    n_settled_races: int = 0
    final_bankroll: int = 0
    peak_bankroll: int = 0
    summary: GroupStats = field(default_factory=lambda: GroupStats(label="all"))
    by_bet_type: list[GroupStats] = field(default_factory=list)
    by_race_class: list[GroupStats] = field(default_factory=list)
    by_course: list[GroupStats] = field(default_factory=list)
    bankroll_timeseries: list[BankrollPoint] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "window": {"start": self.window_start, "end": self.window_end},
            "model_path": self.model_path,
            "strategy": self.strategy,
            "budget": self.budget,
            "n_races": self.n_races,
            "n_settled_races": self.n_settled_races,
            "final_bankroll": self.final_bankroll,
            "peak_bankroll": self.peak_bankroll,
            "summary": self.summary.as_dict(),
            "by_bet_type": [g.as_dict() for g in self.by_bet_type],
            "by_race_class": [g.as_dict() for g in self.by_race_class],
            "by_course": [g.as_dict() for g in self.by_course],
            "bankroll_timeseries": [
                {
                    "date": p.date,
                    "bankroll": p.bankroll,
                    "invested": p.invested,
                    "payout": round(p.payout),
                    "n_bets": p.n_bets,
                }
                for p in self.bankroll_timeseries
            ],
        }


# ---------------------------------------------------------------------------
# Settlement helpers
# ---------------------------------------------------------------------------


# 連系 (馬連 / ワイド / 馬単 / 三連複 / 三連単) の miss を最大何件 log する か。
# KEIBA_DEBUG_SIM_MISSES=1 のときのみ有効。0% hit_rate の根本原因が
# combo 表記不一致なのか pure miss なのかを切り分けるための診断ログ。
_DEBUG_MISSES_LIMIT = 20
_debug_misses_emitted = 0


def _settle_candidates(
    candidates: list,
    race_id: str,
    finish_to_pp: dict[int, int],
    past_odds: dict[str, dict[str, float]],
) -> list[dict]:
    """Determine hit/miss + payout for each recommended candidate.

    Args:
        candidates: list[BetCandidate] from recommend_for_race (stake > 0).
        race_id: target race.
        finish_to_pp: {finish_position: post_position}. Only contains finished horses.
        past_odds: {bet_type: {combo: confirmed_odds_multiplier}} from
            compute_past_race_odds (winners only for 連系; all horses for 単勝/複勝).

    Returns:
        list[dict] with keys: bet_type, stake, payout, hit (0/1)
    """
    global _debug_misses_emitted
    debug_misses = os.environ.get("KEIBA_DEBUG_SIM_MISSES", "0") == "1"

    winner_pp = finish_to_pp.get(1)
    top3 = {finish_to_pp.get(p) for p in (1, 2, 3) if finish_to_pp.get(p) is not None}

    settlements: list[dict] = []
    tan_odds = past_odds.get("単勝", {})
    fuku_odds = past_odds.get("複勝", {})

    for cand in candidates:
        if cand.stake <= 0:
            continue

        hit = False
        payout = 0.0

        if cand.bet_type == "単勝":
            # combo は post_position 文字列。winner と一致したら hit
            if winner_pp is not None and cand.combo == str(winner_pp):
                hit = True
                odds = tan_odds.get(cand.combo, 0.0)
                payout = cand.stake * odds
        elif cand.bet_type == "複勝":
            # combo は post_position 文字列。top-3 にいたら hit
            try:
                pp = int(cand.combo)
            except (ValueError, TypeError):
                pp = None
            if pp is not None and pp in top3:
                hit = True
                odds = fuku_odds.get(cand.combo, 0.0)
                payout = cand.stake * odds
        else:
            # 連系: payouts dict に combo が登録されていれば hit
            confirmed = past_odds.get(cand.bet_type, {}).get(cand.combo)
            if confirmed is not None:
                hit = True
                payout = cand.stake * confirmed
            elif debug_misses and _debug_misses_emitted < _DEBUG_MISSES_LIMIT:
                # combo 表記の不一致 vs 純粋な miss を切り分ける診断 log。
                # past_odds[bet_type] の登録 combo を最大 3 件並べて、cand.combo
                # がそれと比較して妥当かどうかを目視できるようにする。
                _debug_misses_emitted += 1
                bet_keys = list(past_odds.get(cand.bet_type, {}).keys())
                top3_pps = [finish_to_pp.get(p) for p in (1, 2, 3)]
                log.info(
                    "[SIM_DEBUG_MISS] race=%s bet_type=%s cand.combo=%r "
                    "past_keys=%r (sample) top3_pps=%r",
                    race_id, cand.bet_type, cand.combo,
                    bet_keys[:3], top3_pps,
                )

        settlements.append({
            "bet_type": cand.bet_type,
            "stake": int(cand.stake),
            "payout": float(payout),
            "hit": 1 if hit else 0,
            "source": getattr(cand, "est_odds_source", "unknown"),
        })

    return settlements


# ---------------------------------------------------------------------------
# Main simulation entrypoint
# ---------------------------------------------------------------------------


def simulate_active_model(
    session: Session,
    model_path: Path,
    start: str | None,
    end: str | None,
    budget: int,
    strategy: StrategyName = "balanced",
    max_stake_per_race_pct: float = 0.05,
    enabled_bet_types: list[str] | None = None,
    top_n_horses: int = 3,
    max_stake_per_race_yen: int | None = None,
    *,
    bundle: ModelBundle | None = None,
    bet_sink: list[dict] | None = None,
) -> SimulationResult:
    """Run end-to-end backtest using active model + recommendations.

    Args:
        session: SQLAlchemy session bound to the keiba DB.
        model_path: Path to an NN model directory (model.pt + meta.json,
            optionally preprocessor.pkl / temperature_scaler.pkl).
        start / end: window date range (YYYY-MM-DD), inclusive. Both optional.
        budget: 初期資産 (円)。各 race ごとに残資産 (= budget + 累計 profit) を
            bankroll として Kelly stake を計算する (compounding wealth)。
            payout は次 race の bet 余力に加算される。資産が最小単位 (100 円) を
            下回れば以降の race は実質 bet しない (破産)。
        strategy: preset key from STRATEGY_PRESETS.
        max_stake_per_race_pct: per-race stake cap (default 5% of 残資産).
        enabled_bet_types: subset of DEFAULT_BET_TYPES to consider.
            None = all types.
        top_n_horses: top-N horses for box / formation candidates.
        max_stake_per_race_yen: 1 race の累計 stake の絶対上限 (円)。
            compounding wealth で bankroll が膨らんでも 1 race の bet 額が
            無限にインフレしないようにする。None で無効 (pct cap のみ)。

    Returns:
        SimulationResult with summary, by_bet_type, by_race_class, by_course,
        final_bankroll, peak_bankroll, and bankroll_timeseries (日次推移).
    """
    preset = STRATEGY_PRESETS[strategy]
    types = enabled_bet_types or DEFAULT_BET_TYPES

    # Allow callers (notably ad-hoc experiments) to pass a pre-built bundle
    # so they can attach ensemble fields or override calibrators without
    # writing the changes back to disk.  When omitted we load from disk.
    if bundle is None:
        log.info(
            "Loading active model bundle from %s (strategy=%s, budget=%d)",
            model_path, strategy, budget,
        )
        bundle = load_model_full(model_path)
    else:
        log.info(
            "Using pre-built bundle (model_dir=%s, strategy=%s, budget=%d)",
            getattr(bundle, "model_dir", model_path), strategy, budget,
        )

    log.info("Building feature frame for window %s..%s", start, end)
    frame = build_training_frame(session, train_start=start, train_end=end)

    result = SimulationResult(
        window_start=start,
        window_end=end,
        model_path=str(model_path),
        strategy=strategy,
        budget=budget,
    )

    if frame.empty:
        log.warning("No races in window — returning empty simulation result")
        return result

    # Aggregation buckets (keyed by group label)
    bet_type_groups: dict[str, GroupStats] = {}
    race_class_groups: dict[str, GroupStats] = {}
    course_groups: dict[str, GroupStats] = {}

    race_ids = list(frame["race_id"].unique())
    result.n_races = len(race_ids)
    log.info("Simulating %d races...", result.n_races)

    # Compounding wealth: budget を初期資産として、各 race ごとに
    #   bankroll <- bankroll - sum(stake) + sum(payout)
    # で更新する。payout は次の race の bet 余力に加算され、
    # 自信のあるレース (高 EV) ほど Kelly が多めに賭ける挙動になる。
    # bankroll が最小 stake (100 円) を下回ると recommend_for_race 内の
    # cap × 5% も 100 円未満となり実質賭け不可 (= 破産)。
    current_bankroll = budget
    peak_bankroll = budget
    # 日次バケット: その日の累計 stake / payout / 最後の race 終了時の bankroll。
    daily_buckets: dict[str, dict[str, float | int]] = {}

    # odds.db の実オッズで EV 選択を実測ベースにする。未 backfill のレースは
    # load_race_odds が {} を返し、従来の Plackett-Luce 推定へフォールバックする
    # （後方互換）。読み取り専用なので close は loop 後にまとめて行う。
    odds_engine = make_odds_engine()
    init_odds_db(odds_engine)
    odds_session = Session(bind=odds_engine)

    n_settled = 0
    for race_id in race_ids:
        race_frame = frame[frame["race_id"] == race_id]
        if race_frame.empty or len(race_frame) < 2:
            continue

        # Predictions (NN bundle 経由)
        try:
            preds = predict_race(bundle, race_frame)
        except Exception as exc:  # noqa: BLE001
            log.warning("predict_race failed for %s: %s", race_id, exc)
            continue

        # Attach post_position (recommend_for_race needs it)
        pp_map = dict(zip(race_frame["horse_id"].values, race_frame["post_position"].values, strict=True))
        preds["post_position"] = preds["horse_id"].map(pp_map)

        # Combination predictions + odds (with implied fill)
        race_odds, race_odds_sources = compute_race_odds_with_sources(
            session, race_id, odds_session=odds_session
        )
        try:
            combos_by_type = predict_race_with_combinations(
                bundle,
                race_frame,
                session=session,
                race_odds=race_odds,
                race_odds_sources=race_odds_sources,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("predict_race_with_combinations failed for %s: %s", race_id, exc)
            continue

        # Apply min_ev filter (strategy preset)
        min_ev = preset["min_ev"]
        for bt in list(combos_by_type.keys()):
            combos_by_type[bt] = [
                c for c in combos_by_type[bt]
                if c.ev is not None and c.ev >= min_ev
            ]

        # Compounding wealth: 残資産 current_bankroll を Kelly base として渡す。
        # 0 円のときは recommend_for_race 内で cap=0 → 全 stake が 0 に丸まる。
        # Recommend
        rec = recommend_for_race(
            predictions=preds,
            combinations_by_type=combos_by_type,
            race_id=race_id,
            bankroll=current_bankroll,
            kelly_fraction=preset["kelly_fraction"],
            max_stake_per_race_pct=max_stake_per_race_pct,
            top_n_horses=top_n_horses,
            enabled_bet_types=types,
            max_stake_per_race_yen=max_stake_per_race_yen,
        )

        # Determine finish_position map (only finished races settle)
        finished_rows = race_frame[race_frame["finish_position"].notna()]
        if finished_rows.empty:
            continue
        finish_to_pp: dict[int, int] = {}
        for _, row in finished_rows.iterrows():
            try:
                fp = int(row["finish_position"])
                pp = int(row["post_position"])
                finish_to_pp[fp] = pp
            except (ValueError, TypeError):
                continue
        if not finish_to_pp:
            continue

        n_settled += 1

        past_odds = compute_past_race_odds(session, race_id)

        # Aggregate per-race attributes
        race_class = (
            race_frame["race_class"].dropna().iloc[0]
            if "race_class" in race_frame.columns
               and not race_frame["race_class"].dropna().empty
            else "unknown"
        )
        course = (
            race_frame["course"].dropna().iloc[0]
            if "course" in race_frame.columns
               and not race_frame["course"].dropna().empty
            else "unknown"
        )

        settlements = _settle_candidates(
            rec.candidates, race_id, finish_to_pp, past_odds
        )

        # Compounding wealth: race ごとに資産更新。
        # NaN / Inf ガード: odds が壊れた値だと payout が NaN になり得るので 0 に丸める。
        race_invested = sum(int(s["stake"]) for s in settlements)
        race_payout_raw = sum(float(s["payout"]) for s in settlements)
        race_payout = race_payout_raw if math.isfinite(race_payout_raw) else 0.0
        current_bankroll = max(0, current_bankroll - race_invested + int(round(race_payout)))
        if current_bankroll > peak_bankroll:
            peak_bankroll = current_bankroll

        # 日次バケット update (race の date 単位で集約)
        race_date_str = (
            str(race_frame["date"].iloc[0])
            if "date" in race_frame.columns and not race_frame.empty
            else ""
        )
        if race_date_str:
            bucket = daily_buckets.setdefault(
                race_date_str,
                {"invested": 0, "payout": 0.0, "n_bets": 0, "bankroll_at_end": current_bankroll},
            )
            bucket["invested"] = int(bucket["invested"]) + race_invested
            bucket["payout"] = float(bucket["payout"]) + race_payout
            bucket["n_bets"] = int(bucket["n_bets"]) + len(settlements)
            # 同一日内の race は順次処理されるので、最後の race 後の bankroll が残る
            bucket["bankroll_at_end"] = current_bankroll

        for s in settlements:
            # NaN を 0 として扱う (集計 / pydantic int 化で落ちないため)
            s_payout = (
                float(s["payout"])
                if math.isfinite(float(s["payout"]))
                else 0.0
            )
            # Optional per-bet record sink (CI / source-coverage analysis).
            if bet_sink is not None:
                bet_sink.append({
                    "race_id": race_id,
                    "bet_type": s["bet_type"],
                    "stake": int(s["stake"]),
                    "payout": s_payout,
                    "hit": int(s["hit"]),
                    "source": s.get("source", "unknown"),
                })
            # global summary
            result.summary.n_bets += 1
            result.summary.invested += s["stake"]
            result.summary.payout += s_payout
            result.summary.hits += s["hit"]
            # by bet_type
            grp = bet_type_groups.setdefault(
                s["bet_type"], GroupStats(label=s["bet_type"])
            )
            grp.n_bets += 1
            grp.invested += s["stake"]
            grp.payout += s_payout
            grp.hits += s["hit"]
            # by race_class
            cls_grp = race_class_groups.setdefault(
                race_class, GroupStats(label=str(race_class))
            )
            cls_grp.n_bets += 1
            cls_grp.invested += s["stake"]
            cls_grp.payout += s_payout
            cls_grp.hits += s["hit"]
            # by course
            crs_grp = course_groups.setdefault(
                course, GroupStats(label=str(course))
            )
            crs_grp.n_bets += 1
            crs_grp.invested += s["stake"]
            crs_grp.payout += s_payout
            crs_grp.hits += s["hit"]

    odds_session.close()

    result.n_settled_races = n_settled
    result.final_bankroll = current_bankroll
    result.peak_bankroll = peak_bankroll
    # Sort groups by invested desc for predictable display order
    result.by_bet_type = sorted(
        bet_type_groups.values(), key=lambda g: g.invested, reverse=True
    )
    result.by_race_class = sorted(
        race_class_groups.values(), key=lambda g: g.invested, reverse=True
    )
    result.by_course = sorted(
        course_groups.values(), key=lambda g: g.invested, reverse=True
    )
    # 日次 bankroll 推移を date 昇順で list 化 (グラフ用)
    result.bankroll_timeseries = [
        BankrollPoint(
            date=d,
            bankroll=int(v["bankroll_at_end"]),
            invested=int(v["invested"]),
            payout=float(v["payout"]),
            n_bets=int(v["n_bets"]),
        )
        for d, v in sorted(daily_buckets.items())
    ]

    log.info(
        "Done. %d settled races, %d bets, payback=%.3f, hit_rate=%.3f, "
        "final_bankroll=%d (peak=%d, initial=%d)",
        n_settled, result.summary.n_bets,
        result.summary.payback_rate, result.summary.hit_rate,
        result.final_bankroll, result.peak_bankroll, budget,
    )
    return result
