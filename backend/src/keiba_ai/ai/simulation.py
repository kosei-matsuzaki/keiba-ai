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

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
from sqlalchemy.orm import Session

from keiba_ai.ai.bet_odds import (
    compute_past_race_odds,
    compute_race_odds_with_sources,
)
from keiba_ai.ai.bet_strategy import recommend_for_race
from keiba_ai.ai.predict import predict_race, predict_race_with_combinations
from keiba_ai.ai.registry import load_model_full
from keiba_ai.core.logging import get_logger
from keiba_ai.features.builder import build_training_frame

log = get_logger(__name__)


StrategyName = Literal["conservative", "balanced", "aggressive"]

STRATEGY_PRESETS: dict[StrategyName, dict[str, float]] = {
    "conservative": {"kelly_fraction": 0.10, "min_ev": 1.30},
    "balanced":     {"kelly_fraction": 0.25, "min_ev": 1.10},
    "aggressive":   {"kelly_fraction": 0.40, "min_ev": 1.00},
}

# 単勝 / 複勝 / 連系 すべての券種を simulation 対象とする
DEFAULT_BET_TYPES: list[str] = [
    "単勝", "複勝", "馬連", "ワイド", "馬単", "三連複", "三連単",
]


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
class SimulationResult:
    """Top-level simulation result.

    n_races: total races within window (including ones where no bets fired)
    n_settled_races: subset where finish_position was available (i.e. past)
    """

    window_start: str | None
    window_end: str | None
    model_path: str
    strategy: StrategyName
    budget: int
    n_races: int = 0
    n_settled_races: int = 0
    summary: GroupStats = field(default_factory=lambda: GroupStats(label="all"))
    by_bet_type: list[GroupStats] = field(default_factory=list)
    by_race_class: list[GroupStats] = field(default_factory=list)
    by_course: list[GroupStats] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "window": {"start": self.window_start, "end": self.window_end},
            "model_path": self.model_path,
            "strategy": self.strategy,
            "budget": self.budget,
            "n_races": self.n_races,
            "n_settled_races": self.n_settled_races,
            "summary": self.summary.as_dict(),
            "by_bet_type": [g.as_dict() for g in self.by_bet_type],
            "by_race_class": [g.as_dict() for g in self.by_race_class],
            "by_course": [g.as_dict() for g in self.by_course],
        }


# ---------------------------------------------------------------------------
# Settlement helpers
# ---------------------------------------------------------------------------


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

        settlements.append({
            "bet_type": cand.bet_type,
            "stake": int(cand.stake),
            "payout": float(payout),
            "hit": 1 if hit else 0,
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
) -> SimulationResult:
    """Run end-to-end backtest using active model + recommendations.

    Args:
        session: SQLAlchemy session bound to the keiba DB.
        model_path: Path to a model directory (model.txt + binary.txt + calibrator.pkl).
        start / end: window date range (YYYY-MM-DD), inclusive. Both optional.
        budget: bankroll (yen) used as the Kelly-stake base.
        strategy: preset key from STRATEGY_PRESETS.
        max_stake_per_race_pct: per-race stake cap (default 5%).
        enabled_bet_types: subset of DEFAULT_BET_TYPES to consider.
            None = all types.
        top_n_horses: top-N horses for box / formation candidates.

    Returns:
        SimulationResult with summary, by_bet_type, by_race_class, by_course.
    """
    preset = STRATEGY_PRESETS[strategy]
    types = enabled_bet_types or DEFAULT_BET_TYPES

    log.info(
        "Loading active model bundle from %s (strategy=%s, budget=%d)",
        model_path, strategy, budget,
    )
    bundle = load_model_full(model_path)

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

    n_settled = 0
    for race_id in race_ids:
        race_frame = frame[frame["race_id"] == race_id]
        if race_frame.empty or len(race_frame) < 2:
            continue

        # Predictions (calibrated when bundle has binary + calibrator)
        try:
            preds = predict_race(
                bundle.lambdarank, race_frame,
                binary_model=bundle.binary, calibrator=bundle.calibrator,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("predict_race failed for %s: %s", race_id, exc)
            continue

        # Attach post_position (recommend_for_race needs it)
        pp_map = dict(zip(race_frame["horse_id"].values, race_frame["post_position"].values))
        preds["post_position"] = preds["horse_id"].map(pp_map)

        # Combination predictions + odds (with implied fill)
        race_odds, race_odds_sources = compute_race_odds_with_sources(session, race_id)
        try:
            combos_by_type = predict_race_with_combinations(
                bundle.lambdarank, race_frame,
                session=session,
                race_odds=race_odds,
                race_odds_sources=race_odds_sources,
                binary_model=bundle.binary,
                calibrator=bundle.calibrator,
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

        # Recommend
        rec = recommend_for_race(
            predictions=preds,
            combinations_by_type=combos_by_type,
            race_id=race_id,
            bankroll=budget,
            kelly_fraction=preset["kelly_fraction"],
            max_stake_per_race_pct=max_stake_per_race_pct,
            top_n_horses=top_n_horses,
            enabled_bet_types=types,
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

        for s in settlements:
            # global summary
            result.summary.n_bets += 1
            result.summary.invested += s["stake"]
            result.summary.payout += s["payout"]
            result.summary.hits += s["hit"]
            # by bet_type
            grp = bet_type_groups.setdefault(
                s["bet_type"], GroupStats(label=s["bet_type"])
            )
            grp.n_bets += 1
            grp.invested += s["stake"]
            grp.payout += s["payout"]
            grp.hits += s["hit"]
            # by race_class
            cls_grp = race_class_groups.setdefault(
                race_class, GroupStats(label=str(race_class))
            )
            cls_grp.n_bets += 1
            cls_grp.invested += s["stake"]
            cls_grp.payout += s["payout"]
            cls_grp.hits += s["hit"]
            # by course
            crs_grp = course_groups.setdefault(
                course, GroupStats(label=str(course))
            )
            crs_grp.n_bets += 1
            crs_grp.invested += s["stake"]
            crs_grp.payout += s["payout"]
            crs_grp.hits += s["hit"]

    result.n_settled_races = n_settled
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

    log.info(
        "Done. %d settled races, %d bets, payback=%.3f, hit_rate=%.3f",
        n_settled, result.summary.n_bets,
        result.summary.payback_rate, result.summary.hit_rate,
    )
    return result
