"""SQLite connection helpers."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with sensible defaults.

    - row_factory = sqlite3.Row  (column access by name)
    - PRAGMA foreign_keys = ON
    - PRAGMA journal_mode = WAL  (better concurrent read access)
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Generator[sqlite3.Connection, None, None]:
    """Context manager that commits on success and rolls back on exception."""
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
