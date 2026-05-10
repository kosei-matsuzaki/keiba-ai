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
    "bet_records",
    "live_odds",
    "simulation_runs",
}

# races テーブルに含まれるべき列（0005 migration で name を追加）
EXPECTED_RACES_COLUMNS = {
    "race_id", "date", "course", "surface", "distance",
    "weather", "track_condition", "race_class", "n_runners",
    "payout_win", "payout_place", "name",
}

EXPECTED_INDEXES = {
    "ix_entries_race_id_horse_id",
    "ix_entries_horse_id_finish_position",
    "ix_payouts_race_id_bet_type",
    "ix_scrape_log_url_status",
    "ix_scrape_log_fetched_at",
    "ix_bet_records_race_id",
    "ix_bet_records_created_at",
    "ix_bet_records_settled_at",
    "ix_live_odds_race_id",
    "ix_live_odds_race_id_bet_type",
    "uq_live_odds_race_id_bet_type_combo",
    "ix_simulation_runs_created_at",
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

    def test_races_table_has_name_column(self, tmp_db):
        """0005 migration が races.name 列を追加する。"""
        url, engine = tmp_db
        cfg = _make_alembic_cfg(url)
        command.upgrade(cfg, "head")

        inspector = inspect(engine)
        cols = {c["name"] for c in inspector.get_columns("races")}
        assert EXPECTED_RACES_COLUMNS.issubset(cols), (
            f"races テーブルに不足している列: {EXPECTED_RACES_COLUMNS - cols}"
        )

    def test_migration_0005_up_down(self, tmp_db):
        """0005 migration の up → down が冪等に動作する。"""
        url, engine = tmp_db
        cfg = _make_alembic_cfg(url)

        # 0004 まで適用して name 列が存在しないことを確認
        command.upgrade(cfg, "0004")
        inspector = inspect(engine)
        cols_before = {c["name"] for c in inspector.get_columns("races")}
        assert "name" not in cols_before

        # 0005 を適用して name 列が追加されることを確認
        command.upgrade(cfg, "0005")
        inspector = inspect(engine)
        cols_after = {c["name"] for c in inspector.get_columns("races")}
        assert "name" in cols_after

        # downgrade で name 列が削除されることを確認
        command.downgrade(cfg, "0004")
        inspector = inspect(engine)
        cols_down = {c["name"] for c in inspector.get_columns("races")}
        assert "name" not in cols_down

    def test_migration_0006_up_down(self, tmp_db):
        """0006 migration の up → down が冪等に動作する。"""
        url, engine = tmp_db
        cfg = _make_alembic_cfg(url)

        # 0005 まで適用して live_odds テーブルが存在しないことを確認
        command.upgrade(cfg, "0005")
        inspector = inspect(engine)
        assert "live_odds" not in inspector.get_table_names()

        # 0006 を適用して live_odds テーブルが追加されることを確認
        command.upgrade(cfg, "0006")
        inspector = inspect(engine)
        assert "live_odds" in inspector.get_table_names()
        cols = {c["name"] for c in inspector.get_columns("live_odds")}
        assert {"id", "race_id", "bet_type", "combo", "odds", "odds_max", "popularity", "fetched_at"}.issubset(cols)

        # downgrade で live_odds テーブルが削除されることを確認
        command.downgrade(cfg, "0005")
        inspector = inspect(engine)
        assert "live_odds" not in inspector.get_table_names()

    def test_migration_0007_up_down(self, tmp_db):
        """0007 migration の up → down が冪等に動作する。"""
        url, engine = tmp_db
        cfg = _make_alembic_cfg(url)

        # 0006 まで適用して simulation_runs テーブルが存在しないことを確認
        command.upgrade(cfg, "0006")
        inspector = inspect(engine)
        assert "simulation_runs" not in inspector.get_table_names()

        # 0007 を適用して simulation_runs テーブルが追加されることを確認
        command.upgrade(cfg, "0007")
        inspector = inspect(engine)
        assert "simulation_runs" in inspector.get_table_names()
        cols = {c["name"] for c in inspector.get_columns("simulation_runs")}
        assert {
            "id",
            "created_at",
            "budget",
            "strategy",
            "model_path",
            "n_races",
            "final_bankroll",
            "summary_json",
            "bankroll_timeseries_json",
        }.issubset(cols)

        # downgrade で simulation_runs テーブルが削除されることを確認
        command.downgrade(cfg, "0006")
        inspector = inspect(engine)
        assert "simulation_runs" not in inspector.get_table_names()

    def test_migration_0007_idempotent_with_pre_existing_table(self, tmp_db):
        """0007 が「Base.metadata.create_all 由来で simulation_runs が既に存在する」
        DB に対して落ちないことを確認する。

        FastAPI lifespan の create_all (main.py:62) が新テーブルを先回りで作る
        ケースが実環境に存在し、その状態で alembic upgrade head を叩くと
        SQLite "table already exists" で失敗していた回帰テスト。
        """
        url, engine = tmp_db
        cfg = _make_alembic_cfg(url)

        # 0006 まで進めた上で、0007 が作るはずのテーブルを create_all 経由で
        # 先に作っておく（実ユーザの DB 状態を再現）。
        command.upgrade(cfg, "0006")
        Base.metadata.tables["simulation_runs"].create(engine)

        # 0007 が table-exists で落ちずに通ること（修正前はここで OperationalError）
        command.upgrade(cfg, "0007")

        inspector = inspect(engine)
        assert "simulation_runs" in inspector.get_table_names()
        idx_names = {ix["name"] for ix in inspector.get_indexes("simulation_runs")}
        assert "ix_simulation_runs_created_at" in idx_names

    def test_migration_0008_up_down(self, tmp_db):
        """0008 migration の up → down が動作する。

        upgrade: model_runs.model_type 列が追加される。
        downgrade: model_type 列が削除される。
        """
        url, engine = tmp_db
        cfg = _make_alembic_cfg(url)

        # 0007 まで適用して model_type 列が存在しないことを確認
        command.upgrade(cfg, "0007")
        inspector = inspect(engine)
        cols_before = {c["name"] for c in inspector.get_columns("model_runs")}
        assert "model_type" not in cols_before

        # 0008 を適用して model_type 列が追加されることを確認
        command.upgrade(cfg, "0008")
        inspector = inspect(engine)
        cols_after = {c["name"] for c in inspector.get_columns("model_runs")}
        assert "model_type" in cols_after

        # downgrade で model_type 列が削除されることを確認
        command.downgrade(cfg, "0007")
        inspector = inspect(engine)
        cols_down = {c["name"] for c in inspector.get_columns("model_runs")}
        assert "model_type" not in cols_down

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
