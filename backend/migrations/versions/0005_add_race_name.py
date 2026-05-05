"""add races.name column

Revision ID: 0005
Revises: 0004
Create Date: 2026-05-05

races テーブルに name 列（レース名: "有馬記念" / "3歳未勝利" 等）を追加する。
既存行は NULL のままとし、refill_race_meta CLI で遡及的に埋める。
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("races") as batch_op:
        batch_op.add_column(sa.Column("name", sa.String(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("races") as batch_op:
        batch_op.drop_column("name")
