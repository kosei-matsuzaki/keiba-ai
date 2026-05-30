"""drop live_odds table

Revision ID: 0010
Revises: 0009
Create Date: 2026-05-29

live_odds サブシステムを全廃する。締切前オッズ1点+結果の現データではモデルは
オッズの複製にとどまり、closing-line(オッズの動き)も live_odds の上書き保持設計
では時系列を貯められず原理的に追えなかったため、サブシステムごと撤去する。
テーブルは未使用(0 行)でデータ損失はない。
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_index("uq_live_odds_race_id_bet_type_combo", table_name="live_odds")
    op.drop_index("ix_live_odds_race_id_bet_type", table_name="live_odds")
    op.drop_index("ix_live_odds_race_id", table_name="live_odds")
    op.drop_table("live_odds")


def downgrade() -> None:
    op.create_table(
        "live_odds",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("race_id", sa.String(), nullable=False),
        sa.Column("bet_type", sa.String(), nullable=False),
        sa.Column("combo", sa.String(), nullable=False),
        sa.Column("odds", sa.Float(), nullable=True),
        sa.Column("odds_max", sa.Float(), nullable=True),
        sa.Column("popularity", sa.Integer(), nullable=True),
        sa.Column("fetched_at", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(
            ["race_id"], ["races.race_id"],
            name="fk_live_odds_race_id_races",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_live_odds"),
    )
    op.create_index("ix_live_odds_race_id", "live_odds", ["race_id"], unique=False)
    op.create_index(
        "ix_live_odds_race_id_bet_type", "live_odds", ["race_id", "bet_type"], unique=False
    )
    op.create_index(
        "uq_live_odds_race_id_bet_type_combo",
        "live_odds",
        ["race_id", "bet_type", "combo"],
        unique=True,
    )
