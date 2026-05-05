"""add bet_records table with indexes

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-05

ユーザが実際に購入した（または推奨どおり購入したと仮定した）ベット記録を
保存するテーブルを追加する。収支ボード (#129) の前提データとして使用する。

race_id → races は RESTRICT（races 削除前に bet_records を先に消す必要あり）。
recommendation_id は将来の recommendations テーブル追加時の FK 用整数列で、
現時点では外部キー制約なし。
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "bet_records",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.String(), nullable=False),
        sa.Column("race_id", sa.String(), nullable=False),
        sa.Column("bet_type", sa.String(), nullable=False),
        sa.Column("combo", sa.String(), nullable=False),
        sa.Column("stake", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("recommendation_id", sa.Integer(), nullable=True),
        sa.Column("settled_at", sa.String(), nullable=True),
        sa.Column("payout", sa.Integer(), nullable=True),
        sa.Column("profit", sa.Integer(), nullable=True),
        sa.Column("notes", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(
            ["race_id"], ["races.race_id"],
            name="fk_bet_records_race_id_races",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_bet_records"),
    )
    op.create_index("ix_bet_records_race_id", "bet_records", ["race_id"], unique=False)
    op.create_index("ix_bet_records_created_at", "bet_records", ["created_at"], unique=False)
    op.create_index("ix_bet_records_settled_at", "bet_records", ["settled_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_bet_records_settled_at", table_name="bet_records")
    op.drop_index("ix_bet_records_created_at", table_name="bet_records")
    op.drop_index("ix_bet_records_race_id", table_name="bet_records")
    op.drop_table("bet_records")
