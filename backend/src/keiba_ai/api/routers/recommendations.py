"""GET /api/recommendations/{race_id} — recommended bet candidates with Kelly stakes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from keiba_ai.ai.bet_strategy import recommend_for_race
from keiba_ai.ai.predict import predict_race, predict_race_with_combinations
from keiba_ai.ai.registry import get_active, load_model
from keiba_ai.api.deps import get_session, get_settings_store
from keiba_ai.core.settings_store import SettingsStore
from keiba_ai.features.builder import build_inference_frame

router = APIRouter()


class RecommendationCandidate(BaseModel):
    bet_type: str
    combo: str
    pattern: str
    prob: float
    est_odds: float
    ev: float
    stake: int
    post_positions: list[int]


class RecommendationsResponse(BaseModel):
    race_id: str
    bankroll_at_decision: int
    candidates: list[RecommendationCandidate]


@router.get("/recommendations/{race_id}", response_model=RecommendationsResponse)
def get_recommendations(
    race_id: str,
    session: Annotated[Session, Depends(get_session)],
    store: Annotated[SettingsStore, Depends(get_settings_store)],
    top_n_horses: Annotated[int, Query(ge=1, le=18, description="Top-N horses for box/formation candidates (1-18)")] = 3,
    top_k: Annotated[int, Query(ge=1, le=200, description="Combination upper limit per bet type (1-200)")] = 50,
) -> RecommendationsResponse:
    """Return recommended bet candidates for a race.

    Flow:
    1. Resolve active model (503 if none).
    2. Build inference frame for race_id (404 if not found or empty).
    3. Run predict_race to get win_prob / place_prob per horse.
    4. Run predict_race_with_combinations for combination EVs.
    5. Load Settings (bankroll, kelly_fraction, etc.) and call recommend_for_race.
    6. Return RecommendationsResponse.
    """
    active_path = get_active(session)
    if active_path is None:
        raise HTTPException(
            status_code=503,
            detail="No active model. Train and activate a model first.",
        )

    try:
        frame = build_inference_frame(session, race_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if frame.empty:
        raise HTTPException(status_code=404, detail=f"No entries found for race {race_id!r}")

    model = load_model(active_path)

    # Step 3: win_prob / place_prob per horse
    predictions = predict_race(model, frame)

    # Join post_position from frame so recommend_for_race can build top_pps.
    # predict_race returns horse_id-indexed rows without post_position.
    pp_map = dict(zip(frame["horse_id"].values, frame["post_position"].values))
    predictions["post_position"] = predictions["horse_id"].map(pp_map)

    # Step 4: combination EVs (capped by top_k for performance)
    combinations_by_type = predict_race_with_combinations(
        model,
        frame,
        session=session,
        top_k_combinations=top_k,
    )

    # Step 5: load settings and run recommendation logic
    settings = store.load()
    bankroll: int = int(settings.get("bankroll", 100_000))
    kelly_fraction: float = float(settings.get("kelly_fraction", 0.25))
    max_stake_per_race_pct: float = float(settings.get("max_stake_per_race_pct", 0.05))
    enabled_bet_types: list[str] = list(settings.get("enabled_bet_types", ["単勝", "複勝", "ワイド", "馬連"]))

    result = recommend_for_race(
        predictions=predictions,
        combinations_by_type=combinations_by_type,
        race_id=race_id,
        bankroll=bankroll,
        kelly_fraction=kelly_fraction,
        max_stake_per_race_pct=max_stake_per_race_pct,
        top_n_horses=top_n_horses,
        enabled_bet_types=enabled_bet_types,
    )

    candidates = [
        RecommendationCandidate(
            bet_type=c.bet_type,
            combo=c.combo,
            pattern=c.pattern,
            prob=c.prob,
            est_odds=c.est_odds,
            ev=c.ev,
            stake=c.stake,
            post_positions=list(c.post_positions),
        )
        for c in result.candidates
    ]

    return RecommendationsResponse(
        race_id=result.race_id,
        bankroll_at_decision=result.bankroll_at_decision,
        candidates=candidates,
    )
