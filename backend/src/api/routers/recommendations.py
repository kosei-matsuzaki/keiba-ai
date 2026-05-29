"""GET /api/recommendations/{race_id} — recommended bet candidates with Kelly stakes."""

from __future__ import annotations

import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ai.bet_odds import compute_race_odds_with_sources
from ai.bet_strategy import recommend_for_race
from ai.predict import predict_race, predict_race_with_combinations
from ai.registry import get_active, load_model_full
from api.deps import build_inference_frame_or_404, get_session, get_settings_store
from core.logging import get_logger
from core.settings_store import SettingsStore

logger = get_logger(__name__)

router = APIRouter()


class RecommendationCandidate(BaseModel):
    bet_type: str
    combo: str
    pattern: str
    prob: float
    est_odds: float | None
    est_odds_source: Literal["confirmed", "implied", "unknown"] = "unknown"
    ev: float | None
    stake: int
    post_positions: list[int]


class RecommendationsResponse(BaseModel):
    race_id: str
    bankroll_at_decision: int
    candidates: list[RecommendationCandidate]
    odds_source: Literal["live", "past", "unknown"] = "unknown"


def _resolve_odds_source(
    session: Session,
    race_id: str,
) -> tuple[
    dict[str, dict[str, float]] | None,
    dict[str, dict[str, str]] | None,
    Literal["live", "past", "unknown"],
]:
    """Determine odds + per-combo source labels.

    Priority: live > past > unknown for the high-level odds_source label.
    Per-combo source label is "confirmed" / "implied" (set by
    compute_race_odds_with_sources). Missing combos remain absent from the dict.

    Returns:
        (race_odds, sources, odds_source_label).
        race_odds / sources are None when no data is available at all.
    """
    odds, sources = compute_race_odds_with_sources(session, race_id)
    if not odds:
        return None, None, "unknown"

    # high-level label: live data exists ⇄ live; otherwise past
    has_live = any(
        src == "confirmed"
        for combos in sources.values()
        for src in combos.values()
    )
    # ざっくり live と past を区別: 過去レース判定は date < today で行う
    from sqlalchemy import select as sa_select

    from db.models.race import Race as RaceModel

    race_row = session.execute(
        sa_select(RaceModel.date).where(RaceModel.race_id == race_id)
    ).first()

    today_str = datetime.date.today().isoformat()
    is_past = race_row is not None and race_row.date < today_str

    label: Literal["live", "past", "unknown"]
    if is_past:
        label = "past"
    elif has_live:
        label = "live"
    else:
        # 当日レースだが live odds 取得前 → tansho-implied だけが入っている
        label = "live"  # UI 上は "live" 扱いでよい (tansho 由来も market データ)

    return odds, sources, label


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
    3. Run predict_race (GBDT/NN 自動切替) to get win_prob / place_prob per horse.
    4. Resolve race odds: live → past → unknown.
    5. Run predict_race_with_combinations for combination EVs.
    6. Load Settings (bankroll, kelly_fraction, etc.) and call recommend_for_race.
    7. Return RecommendationsResponse.
    """
    active_path = get_active(session)
    if active_path is None:
        raise HTTPException(
            status_code=503,
            detail="No active model. Train and activate a model first.",
        )

    frame = build_inference_frame_or_404(session, race_id)

    bundle = load_model_full(active_path)

    # Step 3: win_prob / place_prob per horse
    # GBDT/NN を bundle.model_type で自動切替 (calibrator / temperature 等は内部で適用)
    predictions = predict_race(bundle, frame)

    # Join post_position from frame so recommend_for_race can build top_pps.
    # predict_race returns horse_id-indexed rows without post_position.
    pp_map = dict(zip(frame["horse_id"].values, frame["post_position"].values, strict=True))
    predictions["post_position"] = predictions["horse_id"].map(pp_map)

    # Step 4: resolve confirmed + implied odds + per-combo source
    race_odds, race_odds_sources, odds_source = _resolve_odds_source(session, race_id)
    if odds_source == "unknown":
        logger.warning(
            "No confirmed odds available for race %s — est_odds will be null", race_id
        )

    # Step 5: combination EVs (capped by top_k for performance)
    combinations_by_type = predict_race_with_combinations(
        bundle,
        frame,
        session=session,
        top_k_combinations=top_k,
        race_odds=race_odds,
        race_odds_sources=race_odds_sources,
    )

    # Step 6: load settings and run recommendation logic
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
            est_odds_source=c.est_odds_source,
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
        odds_source=odds_source,
    )
