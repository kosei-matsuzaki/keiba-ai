"""Shared type definitions for the AI layer.

Placing CombinationPrediction here avoids a circular dependency between
keiba_ai.ai.predict (which produces predictions) and keiba_ai.api.schemas
(which consumes them via routers).  Both layers import from this module.
"""

from __future__ import annotations

from pydantic import BaseModel


class CombinationPrediction(BaseModel):
    """Single combination bet prediction with EV estimate.

    Attributes:
        combo: Human-readable bet combination string.
            - 単勝/複勝: post_position string (e.g. '5')
            - 馬連/ワイド/三連複: post positions joined by '-' in ascending order (e.g. '3-7')
            - 馬単/三連単: post positions joined by '→' in order (e.g. '3→7')
        prob: Estimated probability for this combination.
        est_odds: Estimated odds multiplier (baseline from historical payouts).
        ev: Expected value = prob * est_odds.
        post_positions: Tuple of post position numbers involved (ascending for
            unordered bets, prediction-order for ordered bets).
    """

    combo: str
    prob: float
    est_odds: float
    ev: float
    post_positions: tuple[int, ...]
