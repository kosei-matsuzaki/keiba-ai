"""FastAPI dependency injection helpers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from keiba_ai.api.jobs import JobRegistry
from keiba_ai.core.settings_store import SettingsStore
from keiba_ai.db.session import session_scope


def get_engine(request: Request) -> Engine:
    return request.app.state.engine


def get_session(
    engine: Annotated[Engine, Depends(get_engine)],
) -> Iterator[Session]:
    with session_scope(engine) as s:
        yield s


def get_settings_store(request: Request) -> SettingsStore:
    return request.app.state.settings_store


def get_job_registry(request: Request) -> JobRegistry:
    return request.app.state.job_registry
