"""GET /api/recommendations/{race_id} — recommended bet candidates with flat stakes."""

from __future__ import annotations

import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ai.betting.odds import compute_race_odds_with_sources
from ai.betting.strategy import recommend_for_race
from ai.inference.predict import predict_race, predict_race_with_combinations
from ai.model.registry import get_active, load_model_full
from api.deps import (
    build_inference_frame_or_404,
    get_odds_session,
    get_session,
    get_settings_store,
)
from core.bet_types import DEFAULT_ENABLED_BET_TYPES
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
    est_odds_source: Literal["confirmed", "scraped", "implied", "unknown"] = "unknown"
    ev: float | None
    stake: int
    post_positions: list[int]


class RecommendationsResponse(BaseModel):
    race_id: str
    race_budget: int
    candidates: list[RecommendationCandidate]
    odds_source: Literal["live", "past", "unknown"] = "unknown"


def _resolve_odds_source(
    session: Session,
    odds_session: Session | None,
    race_id: str,
) -> tuple[
    dict[str, dict[str, float]] | None,
    dict[str, dict[str, str]] | None,
    Literal["live", "past", "unknown"],
]:
    """Determine odds + per-combo source labels.

    High-level label は日付ベース: 過去レース → "past"、当日レース → "live"、
    オッズ皆無 → "unknown"。当日レースは payouts 確定前のため単勝(entries.odds_win)
    由来の市場オッズが中心だが、UI 上は "live"(=現時点の市場オッズ)扱いとする。
    Per-combo source label は "confirmed" / "implied" (compute_race_odds_with_sources
    が付与)。取得できない combo は dict から欠落する。

    Returns:
        (race_odds, sources, odds_source_label).
        race_odds / sources are None when no data is available at all.
    """
    odds, sources = compute_race_odds_with_sources(
        session, race_id, odds_session=odds_session
    )
    if not odds:
        return None, None, "unknown"

    # 過去レース判定は date < today。当日レースは市場(単勝)オッズ由来でも "live" 扱い。
    from sqlalchemy import select as sa_select

    from db.models.race import Race as RaceModel

    race_row = session.execute(
        sa_select(RaceModel.date).where(RaceModel.race_id == race_id)
    ).first()

    today_str = datetime.date.today().isoformat()
    is_past = race_row is not None and race_row.date < today_str

    label: Literal["live", "past", "unknown"] = "past" if is_past else "live"
    return odds, sources, label


@router.get("/recommendations/{race_id}", response_model=RecommendationsResponse)
def get_recommendations(
    race_id: str,
    session: Annotated[Session, Depends(get_session)],
    odds_session: Annotated[Session, Depends(get_odds_session)],
    store: Annotated[SettingsStore, Depends(get_settings_store)],
    top_n_horses: Annotated[int, Query(ge=1, le=18, description="Top-N horses for box/formation candidates (1-18)")] = 3,
    top_k: Annotated[int, Query(ge=1, le=200, description="Combination upper limit per bet type (1-200)")] = 50,
    # ── このレースだけ Settings を上書きするための任意パラメータ ──
    # 未指定なら Settings の値を使う。全レース共通の既定値は Settings 側に置き、
    # 「このレースだけ予算を絞る / 券種を単複に限る」といった判断をここで通す。
    race_budget: Annotated[
        int | None,
        Query(ge=100, description="このレースに使う上限 (円)。未指定なら設定値"),
    ] = None,
    stake_unit: Annotated[
        int | None,
        Query(ge=100, description="1 点あたりの賭け金 (円)。未指定なら設定値"),
    ] = None,
    bet_types: Annotated[
        str | None,
        Query(description="このレースだけの対象券種 (カンマ区切り。未指定なら設定値)"),
    ] = None,
) -> RecommendationsResponse:
    """Return recommended bet candidates for a race.

    Flow:
    1. Resolve active model (503 if none).
    2. Build inference frame for race_id (404 if not found or empty).
    3. Run predict_race to get win_prob / place_prob per horse.
    4. Resolve race odds: live → past → unknown.
    5. Run predict_race_with_combinations for combination EVs.
    6. Load Settings (race_budget, stake_unit, etc.) and call recommend_for_race.
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
    # bundle 経由で推論 (temperature scaler 等は内部で適用)
    predictions = predict_race(bundle, frame, session=session)

    # Join post_position from frame so recommend_for_race can build top_pps.
    # predict_race returns horse_id-indexed rows without post_position.
    pp_map = dict(zip(frame["horse_id"].values, frame["post_position"].values, strict=True))
    predictions["post_position"] = predictions["horse_id"].map(pp_map)

    # Step 4: resolve confirmed + scraped(実オッズ) + implied odds + per-combo source
    race_odds, race_odds_sources, odds_source = _resolve_odds_source(
        session, odds_session, race_id
    )
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
    # クエリで渡された分だけ、このレースに限って設定を上書きする。
    settings = store.load()
    eff_budget: int = (
        race_budget if race_budget is not None else int(settings.get("race_budget", 5_000))
    )
    eff_unit: int = (
        stake_unit if stake_unit is not None else int(settings.get("stake_unit", 100))
    )
    # EV 条件を使うのは **連系だけ**。単勝・複勝はモデルの本命 (1 位) を買うルール
    # なので閾値を持たない (strategy.recommend_for_race / docs/ai-model.md)。
    eff_min_ev: float = float(settings.get("win_ev_threshold", 1.1))
    # 単勝のオッズ下限
    eff_win_min_odds: float = float(settings.get("win_min_odds", 1.1))
    if bet_types is not None:
        requested = [t.strip() for t in bet_types.split(",") if t.strip()]
        unknown = [t for t in requested if t not in DEFAULT_ENABLED_BET_TYPES]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown bet_types: {', '.join(unknown)}",
            )
        if not requested:
            raise HTTPException(status_code=422, detail="bet_types must not be empty")
        eff_bet_types = requested
    else:
        eff_bet_types = list(settings.get("enabled_bet_types", DEFAULT_ENABLED_BET_TYPES))

    result = recommend_for_race(
        predictions=predictions,
        combinations_by_type=combinations_by_type,
        race_id=race_id,
        race_budget=eff_budget,
        stake_unit=eff_unit,
        min_ev=eff_min_ev,
        win_min_odds=eff_win_min_odds,
        top_n_horses=top_n_horses,
        enabled_bet_types=eff_bet_types,
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
        race_budget=result.race_budget,
        candidates=candidates,
        odds_source=odds_source,
    )
