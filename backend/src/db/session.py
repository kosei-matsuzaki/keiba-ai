"""SQLAlchemy engine and session factory.

Replaces the sqlite3-based connect/transaction helpers from M2.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker


def make_engine(db_path: Path) -> Engine:
    """Create a SQLite engine with FK enforcement, WAL, and 30s busy_timeout.

    busy_timeout: 並行する書き込みジョブ (例: 長時間 ingest 中の simulation_runs
    INSERT) で 「database is locked」 になりがちなので、待機を 5 → 30 秒に
    伸ばして安定させる。

    pool sizing: bulk predictions など長時間 1 session を握るハンドラと、
    auto-shutuba / auto-odds 自動発射などの BackgroundTask が並走すると
    デフォルト (5+10) では枯渇するため余裕を持たせる。pool_pre_ping で
    死活確認し、reload 後の腐ったコネクションを掴まないようにする。
    SQLite + WAL は読み取り並行可能なのでサイズを上げても安全。
    """
    url = f"sqlite:///{db_path}"
    engine = create_engine(
        url,
        echo=False,
        future=True,
        pool_size=20,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=1800,
    )

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_conn, _connection_record):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")
        dbapi_conn.execute("PRAGMA journal_mode=WAL")
        dbapi_conn.execute("PRAGMA busy_timeout=30000")

    return engine


def _make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """Single-transaction Session: commit on success, rollback on error.

    Use one scope per logical unit of work (e.g. a single race ingest). Loaded
    attributes remain accessible after commit (expire_on_commit=False) but
    relationship lazy-loads outside the scope will fail — re-fetch in a new
    scope instead.
    """
    factory = _make_session_factory(engine)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
