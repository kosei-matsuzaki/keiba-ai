"""add live_odds table for current-day combination odds

Revision ID: 0006
Revises: 0005
Create Date: 2026-05-05

当日のリアルタイムオッズを保存するテーブルを追加する。
fetch_live_odds ジョブが race.netkeiba.com のオッズページを取得し、
各券種×組合せのオッズをここに書き込む。

EV 計算は live_odds が存在する場合に過去払戻平均（baseline）より優先して使用される。
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
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


def downgrade() -> None:
    op.drop_index("uq_live_odds_race_id_bet_type_combo", table_name="live_odds")
    op.drop_index("ix_live_odds_race_id_bet_type", table_name="live_odds")
    op.drop_index("ix_live_odds_race_id", table_name="live_odds")
    op.drop_table("live_odds")
