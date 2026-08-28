"""Model management endpoints: list, detail, activate, train."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from ai.model.registry import delete_model_files, renumber_model_ids, set_active_by_id
from ai.training.train_nn import train_nn
from api.deps import get_job_registry, get_or_404, get_session, get_settings_store
from api.jobs import JobRegistry
from api.schemas import JobAccepted, ModelMeta, TrainRequest, UpdateModelRequest
from core.settings_store import SettingsStore, is_probability_model
from db.models.model_run import ModelRun

router = APIRouter()


def _run_to_schema(run: ModelRun, settings: dict | None = None) -> ModelMeta:
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
        is_probability_model=is_probability_model(run.model_path, settings or {}),
    )


@router.get("/models", response_model=list[ModelMeta])
def get_models(
    session: Annotated[Session, Depends(get_session)],
    store: Annotated[SettingsStore, Depends(get_settings_store)],
) -> list[ModelMeta]:
    runs = session.scalars(
        select(ModelRun).order_by(ModelRun.created_at.desc())
    ).all()
    settings = store.load()
    return [_run_to_schema(r, settings) for r in runs]


@router.get("/models/{model_id}", response_model=ModelMeta)
def get_model(
    model_id: int,
    session: Annotated[Session, Depends(get_session)],
    store: Annotated[SettingsStore, Depends(get_settings_store)],
) -> ModelMeta:
    run = get_or_404(session, ModelRun, model_id, label="Model")
    return _run_to_schema(run, store.load())


@router.post("/models/{model_id}/activate", response_model=ModelMeta)
def activate_model(
    model_id: int,
    session: Annotated[Session, Depends(get_session)],
    store: Annotated[SettingsStore, Depends(get_settings_store)],
) -> ModelMeta:
    run = get_or_404(session, ModelRun, model_id, label="Model")

    # id ベースで activate (パス比較は WSL/Windows でセパレータ差により壊れる)
    set_active_by_id(model_id, session)
    # Refresh after flush so is_active reflects the change
    session.refresh(run)
    return _run_to_schema(run, store.load())


@router.patch("/models/{model_id}", response_model=ModelMeta)
def update_model(
    model_id: int,
    body: UpdateModelRequest,
    session: Annotated[Session, Depends(get_session)],
    store: Annotated[SettingsStore, Depends(get_settings_store)],
) -> ModelMeta:
    """モデルの名称を更新する。空文字を渡すと名称をクリア (NULL) する。"""
    run = get_or_404(session, ModelRun, model_id, label="Model")
    if body.name is not None:
        run.notes = body.name.strip() or None
    session.flush()
    session.refresh(run)
    return _run_to_schema(run, store.load())


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
    store: Annotated[SettingsStore, Depends(get_settings_store)],
) -> None:
    """モデルを削除する。使用中のモデルは削除不可。

    使用中には 2 種類ある:
      * **active** (`model_runs.is_active`) — どの馬・買い目を選ぶかを決める
      * **確率モデル** (`settings.probability_model_path`) — 複勝の確信度と
        連系の確率を出す

    確率モデルを消せてしまうと、設定は存在しないパスを指したまま残り、推論側は
    警告ログを出して**黙って旧挙動に戻る**。画面上は何も起きていないように見える
    のに複勝の絞り込みと連系の確率が無効化されるので、ここで止める。
    """
    run = get_or_404(session, ModelRun, model_id, label="Model")
    if bool(run.is_active):
        raise HTTPException(
            status_code=409,
            detail="Active モデルは削除できません。先に別モデルを activate してください。",
        )
    if is_probability_model(run.model_path, store.load()):
        raise HTTPException(
            status_code=409,
            detail=(
                "確率モデルとして使用中のため削除できません。"
                "先に Settings で別のモデルを選ぶか、未設定にしてください。"
            ),
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
            train_nn,
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
