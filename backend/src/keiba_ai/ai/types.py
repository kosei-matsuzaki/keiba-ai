"""Shared type definitions for the AI layer.

Placing CombinationPrediction here avoids a circular dependency between
keiba_ai.ai.predict (which produces predictions) and keiba_ai.api.schemas
(which consumes them via routers).  Both layers import from this module.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class CombinationPrediction(BaseModel):
    """Single combination bet prediction with EV estimate.

    Attributes:
        combo: Human-readable bet combination string.
            - 単勝/複勝: post_position string (e.g. '5')
            - 馬連/ワイド/三連複: post positions joined by '-' in ascending order (e.g. '3-7')
            - 馬単/三連単: post positions joined by '→' in order (e.g. '3→7')
        prob: Estimated probability for this combination.
        est_odds: Confirmed odds multiplier. None when odds are not available for this combo
            (e.g. non-winning combos in past-race mode, or race_odds not yet fetched).
        ev: Expected value = prob * est_odds. None when est_odds is None.
        post_positions: Tuple of post position numbers involved (ascending for
            unordered bets, prediction-order for ordered bets).
    """

    combo: str
    prob: float
    est_odds: float | None
    ev: float | None
    post_positions: tuple[int, ...]


class BetCandidate(BaseModel):
    """A single bet recommendation with assigned stake.

    Attributes:
        bet_type: 馬券種 (e.g. '単勝', '馬連').
        combo: Human-readable combination string (same format as CombinationPrediction.combo).
        pattern: Buy pattern used to generate this candidate.
        prob: Estimated probability for this combination.
        est_odds: Confirmed odds multiplier. None when odds data is unavailable for this combo.
        ev: Expected value = prob * est_odds. None when est_odds is None.
        stake: Recommended stake in yen (0 if not recommended, EV <= 1.0, or est_odds is None).
        post_positions: Tuple of post position numbers involved.
    """

    bet_type: str
    combo: str
    pattern: Literal["nagashi", "box", "formation"]
    prob: float
    est_odds: float | None
    ev: float | None
    stake: int
    post_positions: tuple[int, ...]


class RecommendationResult(BaseModel):
    """Complete recommendation output for one race.

    Attributes:
        race_id: Target race identifier.
        bankroll_at_decision: Bankroll used for stake calculation.
        candidates: List of bet candidates with non-zero stake (after assign_stakes).
    """

    race_id: str
    bankroll_at_decision: int
    candidates: list[BetCandidate]
