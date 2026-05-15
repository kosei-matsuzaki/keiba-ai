"""Model management endpoints: list, detail, activate, train."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ai.gbm.train import train
from ai.registry import delete_model_files, renumber_model_ids, set_active_by_id
from api.deps import get_job_registry, get_or_404, get_session
from api.jobs import JobRegistry
from api.schemas import JobAccepted, ModelMeta, TrainRequest, UpdateModelRequest
from db.models.model_run import ModelRun

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
        name=run.notes,
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
    runs = session.scalars(
        select(ModelRun).order_by(ModelRun.created_at.desc())
    ).all()
    return [_run_to_schema(r) for r in runs]


@router.get("/models/{model_id}", response_model=ModelMeta)
def get_model(
    model_id: int,
    session: Annotated[Session, Depends(get_session)],
) -> ModelMeta:
    run = get_or_404(session, ModelRun, model_id, label="Model")
    return _run_to_schema(run)


@router.post("/models/{model_id}/activate", response_model=ModelMeta)
def activate_model(
    model_id: int,
    session: Annotated[Session, Depends(get_session)],
) -> ModelMeta:
    run = get_or_404(session, ModelRun, model_id, label="Model")

    # id ベースで activate (パス比較は WSL/Windows でセパレータ差により壊れる)
    set_active_by_id(model_id, session)
    # Refresh after flush so is_active reflects the change
    session.refresh(run)
    return _run_to_schema(run)


@router.patch("/models/{model_id}", response_model=ModelMeta)
def update_model(
    model_id: int,
    body: UpdateModelRequest,
    session: Annotated[Session, Depends(get_session)],
) -> ModelMeta:
    """モデルの名称を更新する。空文字を渡すと名称をクリア (NULL) する。"""
    run = get_or_404(session, ModelRun, model_id, label="Model")
    if body.name is not None:
        run.notes = body.name.strip() or None
    session.flush()
    session.refresh(run)
    return _run_to_schema(run)


@router.post("/models/compact", status_code=204)
def compact_model_ids(
    session: Annotated[Session, Depends(get_session)],
) -> None:
    """ModelRun.id を created_at 昇順で 1..N に詰めて飛び番を解消する。

    削除時は自動で renumber されるが、過去の削除で残った飛び番を一括解消したい
    ときに手動で叩く。FK 参照されていないので安全。
    """
    renumber_model_ids(session)


@router.delete("/models/{model_id}", status_code=204)
def delete_model(
    model_id: int,
    session: Annotated[Session, Depends(get_session)],
) -> None:
    """モデルを削除する。Active モデルは削除不可 (まず別モデルを activate してから)。"""
    run = get_or_404(session, ModelRun, model_id, label="Model")
    if bool(run.is_active):
        raise HTTPException(
            status_code=409,
            detail="Active モデルは削除できません。先に別モデルを activate してください。",
        )
    delete_model_files(run.model_path)
    session.delete(run)
    session.flush()
    renumber_model_ids(session)


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
