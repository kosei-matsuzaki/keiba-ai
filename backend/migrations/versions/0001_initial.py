"""Initial schema: all 8 tables with FK constraints and composite indexes.

Revision ID: 0001
Revises:
Create Date: 2026-04-28

Tables created (in dependency order):
  races, horses, jockeys, trainers, entries, payouts, scrape_log, model_runs

FK CASCADE policy:
  - entries.race_id → races: CASCADE (レース削除時に出走記録も連動削除)
  - entries.horse_id → horses: RESTRICT (馬の履歴を保持; entries を先に消す必要あり)
  - entries.jockey_id → jockeys: SET NULL (騎手引退でもエントリは残す)
  - entries.trainer_id → trainers: SET NULL (調教師も同様)
  - payouts.race_id → races: CASCADE (レースに付随)
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── races ────────────────────────────────────────────────────────────────
    op.create_table(
        "races",
        sa.Column("race_id", sa.String(), nullable=False),
        sa.Column("date", sa.String(), nullable=False),
        sa.Column("course", sa.String(), nullable=False),
        sa.Column("surface", sa.String(), nullable=False),
        sa.Column("distance", sa.Integer(), nullable=False),
        sa.Column("weather", sa.String(), nullable=True),
        sa.Column("track_condition", sa.String(), nullable=True),
        sa.Column("race_class", sa.String(), nullable=True),
        sa.Column("n_runners", sa.Integer(), nullable=True),
        sa.Column("payout_win", sa.Integer(), nullable=True),
        sa.Column("payout_place", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("race_id", name="pk_races"),
    )

    # ── horses ───────────────────────────────────────────────────────────────
    op.create_table(
        "horses",
        sa.Column("horse_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("sex", sa.String(), nullable=True),
        sa.Column("birth_date", sa.String(), nullable=True),
        sa.Column("sire", sa.String(), nullable=True),
        sa.Column("dam", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("horse_id", name="pk_horses"),
    )

    # ── jockeys ──────────────────────────────────────────────────────────────
    op.create_table(
        "jockeys",
        sa.Column("jockey_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("jockey_id", name="pk_jockeys"),
    )

    # ── trainers ─────────────────────────────────────────────────────────────
    op.create_table(
        "trainers",
        sa.Column("trainer_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("trainer_id", name="pk_trainers"),
    )

    # ── entries ──────────────────────────────────────────────────────────────
    op.create_table(
        "entries",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("race_id", sa.String(), nullable=False),
        sa.Column("horse_id", sa.String(), nullable=False),
        sa.Column("post_position", sa.Integer(), nullable=True),
        sa.Column("jockey_id", sa.String(), nullable=True),
        sa.Column("trainer_id", sa.String(), nullable=True),
        sa.Column("weight_carried", sa.Float(), nullable=True),
        sa.Column("age", sa.Integer(), nullable=True),
        sa.Column("sex", sa.String(), nullable=True),
        sa.Column("horse_weight", sa.Integer(), nullable=True),
        sa.Column("horse_weight_diff", sa.Integer(), nullable=True),
        sa.Column("odds_win", sa.Float(), nullable=True),
        sa.Column("popularity", sa.Integer(), nullable=True),
        sa.Column("finish_position", sa.Integer(), nullable=True),
        sa.Column("finish_time", sa.Float(), nullable=True),
        sa.Column("margin", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(
            ["race_id"], ["races.race_id"],
            name="fk_entries_race_id_races",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["horse_id"], ["horses.horse_id"],
            name="fk_entries_horse_id_horses",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["jockey_id"], ["jockeys.jockey_id"],
            name="fk_entries_jockey_id_jockeys",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["trainer_id"], ["trainers.trainer_id"],
            name="fk_entries_trainer_id_trainers",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_entries"),
        sa.UniqueConstraint("race_id", "horse_id", name="uq_entries_race_id_horse_id"),
    )
    op.create_index("ix_entries_race_id_horse_id", "entries", ["race_id", "horse_id"])
    op.create_index("ix_entries_horse_id_finish_position", "entries", ["horse_id", "finish_position"])

    # ── payouts ──────────────────────────────────────────────────────────────
    op.create_table(
        "payouts",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("race_id", sa.String(), nullable=False),
        sa.Column("bet_type", sa.String(), nullable=False),
        sa.Column("combo", sa.String(), nullable=False),
        sa.Column("amount", sa.Integer(), nullable=False),
        sa.Column("popularity", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["race_id"], ["races.race_id"],
            name="fk_payouts_race_id_races",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_payouts"),
    )
    op.create_index("ix_payouts_race_id_bet_type", "payouts", ["race_id", "bet_type"])

    # ── scrape_log ───────────────────────────────────────────────────────────
    op.create_table(
        "scrape_log",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("url", sa.String(), nullable=False),
        sa.Column("fetched_at", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("etag", sa.String(), nullable=True),
        sa.Column("content_hash", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_scrape_log"),
    )
    op.create_index("ix_scrape_log_url_status", "scrape_log", ["url", "status"])

    # ── model_runs ───────────────────────────────────────────────────────────
    op.create_table(
        "model_runs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("model_path", sa.String(), nullable=False),
        sa.Column("params_json", sa.String(), nullable=True),
        sa.Column("train_range", sa.String(), nullable=True),
        sa.Column("valid_range", sa.String(), nullable=True),
        sa.Column("metrics_json", sa.String(), nullable=True),
        sa.Column("notes", sa.String(), nullable=True),
        sa.Column("is_active", sa.Integer(), nullable=False, server_default="0"),
        sa.PrimaryKeyConstraint("id", name="pk_model_runs"),
    )


def downgrade() -> None:
    # Drop in reverse dependency order
    op.drop_table("model_runs")
    op.drop_index("ix_scrape_log_url_status", table_name="scrape_log")
    op.drop_table("scrape_log")
    op.drop_index("ix_payouts_race_id_bet_type", table_name="payouts")
    op.drop_table("payouts")
    op.drop_index("ix_entries_horse_id_finish_position", table_name="entries")
    op.drop_index("ix_entries_race_id_horse_id", table_name="entries")
    op.drop_table("entries")
    op.drop_table("trainers")
    op.drop_table("jockeys")
    op.drop_table("horses")
    op.drop_table("races")
