"""GET /api/predictions/{race_id} — per-horse prediction probabilities."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from keiba_ai.ai.predict import predict_race_with_combinations, predict_race_with_shap
from keiba_ai.ai.registry import get_active, load_model
from keiba_ai.api.deps import get_session
from keiba_ai.api.schemas import (
    CombinationPredictions,
    HorsePrediction,
    PredictionResponse,
)
from keiba_ai.db.models.model_run import ModelRun
from keiba_ai.features.builder import build_inference_frame

router = APIRouter()


@router.get("/predictions/{race_id}", response_model=PredictionResponse)
def get_predictions(
    race_id: str,
    session: Annotated[Session, Depends(get_session)],
    include_combinations: Annotated[bool, Query(description="Compute combination bet EV predictions")] = True,
    top_k: Annotated[int | None, Query(ge=1, description="Limit each bet type to top-K combinations by EV")] = None,
) -> PredictionResponse:
    active_path = get_active(session)
    if active_path is None:
        raise HTTPException(status_code=503, detail="No active model. Train and activate a model first.")

    try:
        frame = build_inference_frame(session, race_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if frame.empty:
        raise HTTPException(status_code=404, detail=f"No entries found for race {race_id!r}")

    model = load_model(active_path)
    result_df = predict_race_with_shap(model, frame)

    # Resolve model_runs id for the active model
    active_run = session.scalars(
        select(ModelRun).where(ModelRun.is_active == 1).limit(1)
    ).first()
    model_id = active_run.id if active_run else 0

    predictions = [
        HorsePrediction(
            horse_id=str(row["horse_id"]),
            score=float(row["score"]),
            win_prob=float(row["win_prob"]),
            place_prob=float(row["place_prob"]),
            top_features=list(row.get("top_features") or []),
        )
        for _, row in result_df.iterrows()
    ]

    combinations_out: CombinationPredictions | None = None
    if include_combinations:
        combo_map = predict_race_with_combinations(
            model,
            frame,
            session=session,
            top_k_combinations=top_k,
        )
        combinations_out = CombinationPredictions(
            tansho=combo_map.get("単勝", []),
            fukusho=combo_map.get("複勝", []),
            umaren=combo_map.get("馬連", []),
            wide=combo_map.get("ワイド", []),
            umatan=combo_map.get("馬単", []),
            sanrenpuku=combo_map.get("三連複", []),
            sanrentan=combo_map.get("三連単", []),
        )

    return PredictionResponse(
        race_id=race_id,
        model_id=model_id,
        predictions=predictions,
        combinations=combinations_out,
    )
