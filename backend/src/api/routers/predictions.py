"""Prediction endpoints.

- GET /api/predictions/bulk       — bulk top-N predictions for multiple races
- GET /api/predictions/{race_id}  — per-horse prediction probabilities

NOTE: /predictions/bulk must be declared BEFORE /predictions/{race_id} so that
FastAPI does not match the literal "bulk" as a race_id path parameter.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ai.inference.predict import (
    predict_race,
    predict_race_with_combinations,
    predict_race_with_shap,
)
from ai.model.registry import get_active, load_model_full
from api.deps import build_inference_frame_or_404, get_session
from api.schemas import (
    BulkPredictionsResponse,
    CombinationPredictions,
    HorsePrediction,
    PredictionResponse,
    RacePredictionSummary,
    TopHorse,
)
from db.models.horse import Horse
from db.models.model_run import ModelRun
from features.builder import build_inference_frame

log = logging.getLogger(__name__)

router = APIRouter()

_BULK_MAX_RACE_IDS = 100


@router.get("/predictions/bulk", response_model=BulkPredictionsResponse)
def get_bulk_predictions(
    session: Annotated[Session, Depends(get_session)],
    race_ids: Annotated[str, Query(description="カンマ区切り race_id リスト（最大 100 件）")] = "",
    top_n: Annotated[int, Query(ge=1, le=20, description="返す上位馬の件数")] = 3,
) -> BulkPredictionsResponse:
    """複数レースの上位 top_n 馬予想を一括取得する。

    - active モデルが無い場合は全 race を空の RacePredictionSummary で返す（503 ではなく空）。
    - entries が無い race も空の RacePredictionSummary を返す。
    - 最大 _BULK_MAX_RACE_IDS 件まで処理する。
    """
    if not race_ids.strip():
        return BulkPredictionsResponse(predictions={})

    parsed_ids = [rid.strip() for rid in race_ids.split(",") if rid.strip()]
    parsed_ids = parsed_ids[:_BULK_MAX_RACE_IDS]

    # active モデルが無ければ全レースを空で返す
    active_path = get_active(session)
    if active_path is None:
        log.info("No active model; returning empty bulk predictions for %d races", len(parsed_ids))
        return BulkPredictionsResponse(
            predictions={rid: RacePredictionSummary(top_horses=[]) for rid in parsed_ids}
        )

    bundle = load_model_full(active_path)

    # horse_id → horse_name マップをキャッシュ（per-request）
    horse_name_cache: dict[str, str | None] = {}

    def _horse_name(horse_id: str) -> str | None:
        if horse_id not in horse_name_cache:
            h = session.get(Horse, horse_id)
            horse_name_cache[horse_id] = h.name if h else None
        return horse_name_cache[horse_id]

    result: dict[str, RacePredictionSummary] = {}

    for race_id in parsed_ids:
        try:
            frame = build_inference_frame(session, race_id)
        except ValueError:
            result[race_id] = RacePredictionSummary(top_horses=[])
            continue

        if frame.empty:
            result[race_id] = RacePredictionSummary(top_horses=[])
            continue

        try:
            pred_df = predict_race(bundle, frame)
        except Exception as exc:
            log.warning("Prediction failed for race %s: %s", race_id, exc)
            result[race_id] = RacePredictionSummary(top_horses=[])
            continue

        # 上位 top_n 馬を抽出（score 降順ですでにソート済み）
        top_rows = pred_df.head(top_n)

        # post_position は feature frame から取得する
        post_pos_map: dict[str, int | None] = {}
        if "post_position" in frame.columns:
            for _, frow in frame.iterrows():
                hid = str(frow.get("horse_id", ""))
                pp = frow.get("post_position")
                post_pos_map[hid] = int(pp) if pp is not None else None

        top_horses = []
        for _, row in top_rows.iterrows():
            hid = str(row["horse_id"])
            top_horses.append(TopHorse(
                post_position=post_pos_map.get(hid),
                horse_name=_horse_name(hid),
                win_prob=float(row["win_prob"]),
            ))

        result[race_id] = RacePredictionSummary(top_horses=top_horses)

    return BulkPredictionsResponse(predictions=result)


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

    frame = build_inference_frame_or_404(session, race_id)

    bundle = load_model_full(active_path)
    # bundle-aware: NN モデルでは top_features=[] が返る
    result_df = predict_race_with_shap(bundle, frame)

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
            bundle,
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
