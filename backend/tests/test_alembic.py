"""Alembic migration tests.

Verifies that:
  - upgrade head creates all expected tables and indexes
  - downgrade base drops all tables
  - re-upgrade works (idempotent round-trip)
  - autogen produces no diff (schema in code matches migration)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, event, inspect

import keiba_ai.db.models  # noqa: F401  (populate Base.metadata)
from keiba_ai.db.base import Base

BACKEND_DIR = Path(__file__).resolve().parent.parent
ALEMBIC_INI = BACKEND_DIR / "alembic.ini"

EXPECTED_TABLES = {
    "races",
    "horses",
    "jockeys",
    "trainers",
    "entries",
    "payouts",
    "scrape_log",
    "model_runs",
}

EXPECTED_INDEXES = {
    "ix_entries_race_id_horse_id",
    "ix_entries_horse_id_finish_position",
    "ix_payouts_race_id_bet_type",
    "ix_scrape_log_url_status",
    "ix_scrape_log_fetched_at",
}


def _make_alembic_cfg(db_url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", db_url)
    cfg.set_main_option("script_location", str(BACKEND_DIR / "migrations"))
    return cfg


@pytest.fixture()
def tmp_db(tmp_path):
    """Temporary SQLite file URL for Alembic migration tests."""
    db_file = tmp_path / "test_alembic.db"
    url = f"sqlite:///{db_file}"
    engine = create_engine(url, future=True)

    @event.listens_for(engine, "connect")
    def _set_fk(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    yield url, engine
    engine.dispose()


class TestAlembicMigrations:
    def test_upgrade_head_creates_all_tables(self, tmp_db):
        url, engine = tmp_db
        cfg = _make_alembic_cfg(url)
        command.upgrade(cfg, "head")

        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        # alembic_version is added by Alembic itself
        assert EXPECTED_TABLES.issubset(tables), f"Missing tables: {EXPECTED_TABLES - tables}"

    def test_upgrade_head_creates_indexes(self, tmp_db):
        url, engine = tmp_db
        cfg = _make_alembic_cfg(url)
        command.upgrade(cfg, "head")

        inspector = inspect(engine)
        all_indexes: set[str] = set()
        for table in EXPECTED_TABLES:
            for idx in inspector.get_indexes(table):
                all_indexes.add(idx["name"])

        assert EXPECTED_INDEXES.issubset(all_indexes), f"Missing indexes: {EXPECTED_INDEXES - all_indexes}"

    def test_downgrade_base_drops_all_tables(self, tmp_db):
        url, engine = tmp_db
        cfg = _make_alembic_cfg(url)
        command.upgrade(cfg, "head")
        command.downgrade(cfg, "base")

        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        remaining = tables - {"alembic_version"}
        assert remaining == set(), f"Tables still exist after downgrade: {remaining}"

    def test_upgrade_downgrade_upgrade_roundtrip(self, tmp_db):
        url, engine = tmp_db
        cfg = _make_alembic_cfg(url)
        command.upgrade(cfg, "head")
        command.downgrade(cfg, "base")
        command.upgrade(cfg, "head")  # must not raise

        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        assert EXPECTED_TABLES.issubset(tables)

    def test_autogen_produces_no_diff(self, tmp_db):
        """After upgrade head, autogen should find no schema differences."""
        url, engine = tmp_db
        cfg = _make_alembic_cfg(url)
        command.upgrade(cfg, "head")

        with engine.connect() as conn:
            migration_ctx = MigrationContext.configure(
                conn,
                opts={
                    "compare_type": True,
                    "render_as_batch": True,
                },
            )
            diffs = compare_metadata(migration_ctx, Base.metadata)

        # Filter out alembic_version table diffs if any
        meaningful_diffs = [d for d in diffs if not (
            isinstance(d, tuple) and len(d) > 1 and "alembic_version" in str(d)
        )]
        assert meaningful_diffs == [], f"Schema diff detected: {meaningful_diffs}"
