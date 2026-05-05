"""FastAPI application entry point.

Usage:
    uv run uvicorn keiba_ai.main:app --host 127.0.0.1 --port 8765 --reload
    python -m keiba_ai.main  (starts uvicorn directly)
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from keiba_ai.api.jobs import JobRegistry
from keiba_ai.api.routers import (
    bets,
    health,
    internal,
    jobs,
    metrics,
    models,
    predictions,
    races,
    scraper,
    settings,
)
from keiba_ai.core.paths import db_path
from keiba_ai.core.settings_store import SettingsStore
from keiba_ai.db.session import make_engine

_DEFAULT_ORIGINS = [
    "http://localhost:5173",   # Vite dev server
    "http://localhost:1420",   # Tauri dev default
    "tauri://localhost",       # Tauri production WebView (macOS/Linux)
    "http://tauri.localhost",  # Tauri production WebView (Windows)
    "https://tauri.localhost", # Tauri production WebView (Windows, https)
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
    application.include_router(internal.router, prefix="/api")

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
