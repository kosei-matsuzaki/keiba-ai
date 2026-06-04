"""Tests for jobs/backup_db.py — online SQLite backup + generation rotation."""

from __future__ import annotations

import sqlite3

import pytest

from jobs import backup_db


def _make_db(path, rows: int) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        conn.executemany("INSERT INTO t (v) VALUES (?)", [(f"r{i}",) for i in range(rows)])
        conn.commit()
    finally:
        conn.close()


@pytest.fixture()
def data_env(tmp_path, monkeypatch):
    """Point KEIBA_DATA_DIR at a temp dir and build a small keiba.db."""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    _make_db(tmp_path / "keiba.db", rows=5)
    return tmp_path


def test_backup_one_creates_consistent_copy(data_env):
    dest = backup_db.backup_one(data_env / "keiba.db", "keiba")
    assert dest is not None
    assert dest.exists()
    assert dest.parent == data_env / "backups"
    # backup is a valid sqlite db with the same row count
    conn = sqlite3.connect(dest)
    try:
        assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 5
    finally:
        conn.close()


def test_backup_one_missing_source_returns_none(data_env):
    assert backup_db.backup_one(data_env / "nope.db", "odds") is None


def test_rotation_keeps_newest_n(data_env):
    bdir = backup_db.backups_dir()
    # 10 fake generations with sortable timestamped names
    for i in range(10):
        (bdir / f"keiba-202601{i:02d}-000000.db").write_bytes(b"x")
    removed = backup_db._rotate("keiba", keep=7)
    remaining = sorted(p.name for p in bdir.glob("keiba-*.db"))
    assert len(remaining) == 7
    assert len(removed) == 3
    # oldest 3 removed, newest 7 kept
    assert remaining[0] == "keiba-20260103-000000.db"
    assert remaining[-1] == "keiba-20260109-000000.db"


def test_rotation_only_touches_matching_prefix(data_env):
    bdir = backup_db.backups_dir()
    (bdir / "keiba-20260101-000000.db").write_bytes(b"x")
    (bdir / "odds-20260101-000000.db").write_bytes(b"x")
    backup_db._rotate("keiba", keep=0)  # keep<=0 is a no-op guard
    assert (bdir / "odds-20260101-000000.db").exists()


def test_run_backup_both_targets(data_env):
    _make_db(data_env / "odds.db", rows=3)
    created = backup_db.run_backup(["keiba", "odds"], keep=7)
    assert len(created) == 2
    prefixes = sorted(p.name.split("-")[0] for p in created)
    assert prefixes == ["keiba", "odds"]


def test_run_backup_integrates_with_preexisting_generations(data_env):
    """Pre-existing keiba-*.db backups count toward the keep limit."""
    bdir = backup_db.backups_dir()
    for i in range(7):
        (bdir / f"keiba-202601{i:02d}-000000.db").write_bytes(b"x")
    # one fresh online backup → 8 total → rotate down to 7
    backup_db.run_backup(["keiba"], keep=7)
    assert len(list(bdir.glob("keiba-*.db"))) == 7
