"""FastAPI dependency injection helpers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated, TypeVar

import pandas as pd
from fastapi import Depends, HTTPException, Request
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from api.jobs import JobRegistry
from core.settings_store import SettingsStore
from db.session import session_scope
from features.builder import build_inference_frame

T = TypeVar("T")


def get_engine(request: Request) -> Engine:
    return request.app.state.engine


def get_session(
    engine: Annotated[Engine, Depends(get_engine)],
) -> Iterator[Session]:
    with session_scope(engine) as s:
        yield s


def get_odds_engine(request: Request) -> Engine:
    """odds.db の Engine。lifespan が設定済みならそれを使い、無ければ遅延生成する。

    lifespan startup が（reload のタイミング等で）app.state.odds_engine を設定し
    損ねても 500 にならないよう、ここで idempotent に生成して app.state へキャッシュ
    する。create_app() を lifespan 無しで使うテスト経路にも耐える。
    """
    state = request.app.state
    engine = getattr(state, "odds_engine", None)
    if engine is None:
        from db.odds_db import init_odds_db, make_odds_engine

        engine = make_odds_engine()
        init_odds_db(engine)
        state.odds_engine = engine
    return engine


def get_odds_session(
    odds_engine: Annotated[Engine, Depends(get_odds_engine)],
) -> Iterator[Session]:
    """odds.db（実オッズ）の読み取り用セッション。"""
    from db.odds_db import odds_session_scope

    with odds_session_scope(odds_engine) as s:
        yield s


def get_settings_store(request: Request) -> SettingsStore:
    return request.app.state.settings_store


def get_job_registry(request: Request) -> JobRegistry:
    return request.app.state.job_registry


def get_or_404[T](session: Session, model: type[T], pk: object, label: str | None = None) -> T:
    """`session.get(model, pk)` の結果が None なら 404 を投げる。"""
    obj = session.get(model, pk)
    if obj is None:
        name = label or model.__name__
        raise HTTPException(status_code=404, detail=f"{name} {pk!r} not found")
    return obj


def build_inference_frame_or_404(session: Session, race_id: str) -> pd.DataFrame:
    """`build_inference_frame` の API ラッパ。Race 不在 / entry 0 を 404 に変換する。"""
    try:
        frame = build_inference_frame(session, race_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if frame.empty:
        raise HTTPException(status_code=404, detail=f"No entries found for race {race_id!r}")
    return frame
