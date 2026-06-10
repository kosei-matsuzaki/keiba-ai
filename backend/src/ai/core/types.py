"""Shared type definitions for the AI layer.

Placing CombinationPrediction here avoids a circular dependency between
ai.inference.predict (which produces predictions) and api.schemas
(which consumes them via routers).  Both layers import from this module.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# "scraped" = odds.db に取り込んだ実市場オッズ（全 combo 確定オッズ）。
# "confirmed" = payouts / entries.odds_win 由来（連系は当選 combo のみ）。
# "implied"   = 単勝からの Plackett-Luce 推定。"unknown" = est_odds 取得不能。
EstOddsSource = Literal["confirmed", "scraped", "implied", "unknown"]


class CombinationPrediction(BaseModel):
    """Single combination bet prediction with EV estimate.

    Attributes:
        combo: Human-readable bet combination string.
            - 単勝/複勝: post_position string (e.g. '5')
            - 馬連/ワイド/三連複: post positions joined by '-' in ascending order (e.g. '3-7')
            - 馬単/三連単: post positions joined by '→' in order (e.g. '3→7')
        prob: Estimated probability for this combination.
        est_odds: Odds multiplier used for EV. None when neither confirmed nor
            implied odds are available (e.g. tansho missing).
        est_odds_source: Where est_odds came from.
            - "confirmed": payouts / entries.odds_win 由来の確定値
            - "implied": 単勝オッズから Plackett-Luce で推定した値
            - "unknown": 推定不能（est_odds は None）
        ev: Expected value = prob * est_odds. None when est_odds is None.
        post_positions: Tuple of post position numbers involved (ascending for
            unordered bets, prediction-order for ordered bets).
    """

    combo: str
    prob: float
    est_odds: float | None
    est_odds_source: EstOddsSource = "unknown"
    ev: float | None
    post_positions: tuple[int, ...]


class BetCandidate(BaseModel):
    """A single bet recommendation with assigned stake.

    Attributes:
        bet_type: 馬券種 (e.g. '単勝', '馬連').
        combo: Human-readable combination string (same format as CombinationPrediction.combo).
        pattern: Buy pattern used to generate this candidate.
        prob: Estimated probability for this combination.
        est_odds: Odds multiplier used for EV. None when neither confirmed nor
            implied odds are available.
        est_odds_source: 同 CombinationPrediction.est_odds_source。
        ev: Expected value = prob * est_odds. None when est_odds is None.
        stake: Recommended stake in yen (0 if not recommended, EV <= 1.0, or est_odds is None).
        post_positions: Tuple of post position numbers involved.
    """

    bet_type: str
    combo: str
    pattern: Literal["nagashi", "box", "formation"]
    prob: float
    est_odds: float | None
    est_odds_source: EstOddsSource = "unknown"
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
