"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

# Import all models so Base.metadata is fully populated
import db.models  # noqa: F401
from db.base import Base

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
    # `data_dir()` は毎回この環境変数を読むので、これだけで隔離できる。
    monkeypatch.setenv("KEIBA_DATA_DIR", str(data_dir))

    # **`core.paths.data_dir` を差し替えてはいけない。**
    # 差し替えた状態で `core.settings_store` などが初めて import されると、
    # そのモジュールは `from core.paths import data_dir` で**その回の lambda を
    # 永久に掴む**。monkeypatch は core.paths 側しか戻さないので、以降の
    # テストは全部 1 つ目のテストの tmp ディレクトリを見る。
    # 実際に「別テストが PUT した設定が次のテストの GET に出る」で踏んだ。

    from core.paths import db_path as _db_path
    from db.session import make_engine

    engine = make_engine(_db_path())
    Base.metadata.create_all(engine)
    engine.dispose()

    # Reimport main so the lifespan sees the monkeypatched paths
    import importlib

    import main as _main_mod

    importlib.reload(_main_mod)

    app = _main_mod.create_app()
    # 設定ストアを**明示的に**この tmp_path 配下へ向ける。
    # create_app 内の SettingsStore() は data_dir() 経由で環境変数を読むので
    # 実際には隔離されているが、それは「呼び出し時に env が立っている」ことに
    # 依存した暗黙の隔離で、fixture の順序やインポート時刻が変わると崩れる。
    # 1 つのテストが書いた設定が次のテストに漏れると、原因の分かりにくい
    # 失敗になる (実際に「EV 閾値が 1.1 のはずが 1.2」で一度踏んだ)。
    from core.settings_store import SettingsStore
    app.state.settings_store = SettingsStore(data_dir / "settings.json")
    return app


@pytest.fixture()
def api_client(app_with_temp_db):
    """TestClient wrapping the isolated app."""
    with TestClient(app_with_temp_db) as client:
        yield client
