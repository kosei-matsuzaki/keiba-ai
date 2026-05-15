"""FastAPI application entry point.

Usage:
    uv run uvicorn main:app --host 127.0.0.1 --port 8765 --reload
    python -m main  (starts uvicorn directly)
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import db.models  # noqa: F401  Base.metadata に全モデルを登録
from api.jobs import JobRegistry
from api.routers import (
    bets,
    health,
    jobs,
    metrics,
    models,
    predictions,
    races,
    recommendations,
    scraper,
    settings,
    simulation,
)
from core.paths import db_path
from core.settings_store import SettingsStore
from db.base import Base
from db.session import make_engine

_DEFAULT_ORIGINS = [
    "http://localhost:5173",   # Vite dev server
    "http://127.0.0.1:5173",
]


def _cors_origins() -> list[str]:
    origins = list(_DEFAULT_ORIGINS)
    extra = os.getenv("KEIBA_CORS_EXTRA", "").strip()
    if extra:
        origins.extend(o.strip() for o in extra.split(",") if o.strip())
    return origins


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup
    engine = make_engine(db_path())
    # 既存 DB に未適用のテーブル (alembic を ユーザー手動で apply していない
    # ケース) を idempotent に作成する。
    # simulation_runs などの新テーブルが無くても落ちないようにする。
    Base.metadata.create_all(engine)
    app.state.engine = engine
    app.state.settings_store = SettingsStore()
    app.state.job_registry = JobRegistry()
    try:
        yield
    finally:
        # Shutdown
        engine.dispose()


def create_app() -> FastAPI:
    """Factory function — instantiates and configures the FastAPI app.

    Called by uvicorn (via module attribute) and by tests (via direct call).
    """
    application = FastAPI(
        title="KEIBA AI Backend",
        version="0.1.0",
        lifespan=_lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(health.router, prefix="/api")
    application.include_router(races.router, prefix="/api")
    application.include_router(bets.router, prefix="/api")
    application.include_router(predictions.router, prefix="/api")
    application.include_router(metrics.router, prefix="/api")
    application.include_router(models.router, prefix="/api")
    application.include_router(scraper.router, prefix="/api")
    application.include_router(jobs.router, prefix="/api")
    application.include_router(settings.router, prefix="/api")
    application.include_router(recommendations.router, prefix="/api")
    application.include_router(simulation.router, prefix="/api")

    return application


# Module-level app instance consumed by uvicorn
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.getenv("KEIBA_API_PORT", "8765")),
        log_level="info",
    )
