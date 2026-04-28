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
    """Create a SQLite engine with FK enforcement and WAL journal mode."""
    url = f"sqlite:///{db_path}"
    engine = create_engine(url, echo=False, future=True)

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_conn, _connection_record):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")
        dbapi_conn.execute("PRAGMA journal_mode=WAL")

    return engine


def _make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """Context manager that provides a Session, committing on success and rolling back on error."""
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
