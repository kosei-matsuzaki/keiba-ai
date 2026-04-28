"""Model management endpoints: list, detail, activate, train."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from keiba_ai.ai.registry import set_active
from keiba_ai.ai.train import train
from keiba_ai.api.deps import get_job_registry, get_session
from keiba_ai.api.jobs import JobRegistry
from keiba_ai.api.schemas import JobAccepted, ModelMeta, TrainRequest
from keiba_ai.db.models.model_run import ModelRun

router = APIRouter()


def _run_to_schema(run: ModelRun) -> ModelMeta:
    params: dict | None = None
    if run.params_json:
        with contextlib.suppress(json.JSONDecodeError):
            params = json.loads(run.params_json)

    metrics: dict | None = None
    if run.metrics_json:
        with contextlib.suppress(json.JSONDecodeError):
            metrics = json.loads(run.metrics_json)

    return ModelMeta(
        id=run.id,
        created_at=run.created_at,
        model_path=run.model_path,
        train_range=run.train_range,
        valid_range=run.valid_range,
        params=params,
        metrics=metrics,
        is_active=bool(run.is_active),
    )


@router.get("/models", response_model=list[ModelMeta])
def get_models(
    session: Annotated[Session, Depends(get_session)],
) -> list[ModelMeta]:
    runs = session.query(ModelRun).order_by(ModelRun.created_at.desc()).all()
    return [_run_to_schema(r) for r in runs]


@router.get("/models/{model_id}", response_model=ModelMeta)
def get_model(
    model_id: int,
    session: Annotated[Session, Depends(get_session)],
) -> ModelMeta:
    run = session.get(ModelRun, model_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return _run_to_schema(run)


@router.post("/models/{model_id}/activate", response_model=ModelMeta)
def activate_model(
    model_id: int,
    session: Annotated[Session, Depends(get_session)],
) -> ModelMeta:
    run = session.get(ModelRun, model_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")

    from pathlib import Path
    set_active(Path(run.model_path), session)
    # Refresh after flush so is_active reflects the change
    session.refresh(run)
    return _run_to_schema(run)


@router.post("/models/train", response_model=JobAccepted)
async def train_model(
    body: TrainRequest,
    session: Annotated[Session, Depends(get_session)],  # noqa: ARG001
    registry: Annotated[JobRegistry, Depends(get_job_registry)],
) -> JobAccepted:
    async def _coro() -> None:
        await asyncio.to_thread(
            train,
            train_end=body.train_end,
            valid_months=body.valid_months or 12,
            test_months=body.test_months or 6,
        )

    info = registry.start("train", _coro)
    return JobAccepted(
        job_id=info.job_id,
        status=info.status,
        started_at=info.started_at,
    )
