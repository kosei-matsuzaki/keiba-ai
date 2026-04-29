"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

# Import all models so Base.metadata is fully populated
import keiba_ai.db.models  # noqa: F401
from keiba_ai.db.base import Base

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture()
def calendar_html() -> str:
    return (FIXTURES_DIR / "race_calendar_20241228.html").read_text(encoding="utf-8")


@pytest.fixture()
def race_result_html() -> str:
    return (FIXTURES_DIR / "race_result_202406010101.html").read_text(encoding="utf-8")


@pytest.fixture()
def robots_txt() -> str:
    return (FIXTURES_DIR / "robots.txt").read_text(encoding="utf-8")


# ── ORM fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture()
def in_memory_engine():
    """SQLite in-memory engine with all tables created."""
    engine = create_engine("sqlite:///:memory:", future=True)
    # Enable FK enforcement for in-memory DB

    @event.listens_for(engine, "connect")
    def _set_fk(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture()
def db_session(in_memory_engine):
    """ORM Session over in-memory engine. Each test gets a fresh session."""
    with Session(in_memory_engine) as session:
        yield session


# ── FastAPI test client fixtures ──────────────────────────────────────────────

@pytest.fixture()
def app_with_temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolated FastAPI app backed by a fresh temp-dir SQLite DB."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("KEIBA_DATA_DIR", str(data_dir))

    # Ensure a fresh engine is built from the patched env inside create_app()
    import keiba_ai.core.paths as _paths
    monkeypatch.setattr(_paths, "data_dir", lambda: data_dir)

    from keiba_ai.core.paths import db_path as _db_path
    from keiba_ai.db.session import make_engine

    engine = make_engine(_db_path())
    Base.metadata.create_all(engine)
    engine.dispose()

    # Reimport main so the lifespan sees the monkeypatched paths
    import importlib

    import keiba_ai.main as _main_mod

    importlib.reload(_main_mod)

    app = _main_mod.create_app()
    return app


@pytest.fixture()
def api_client(app_with_temp_db):
    """TestClient wrapping the isolated app."""
    with TestClient(app_with_temp_db) as client:
        yield client
