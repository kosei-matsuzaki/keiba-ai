"""add model_type column to model_runs

Revision ID: 0008
Revises: 0007
Create Date: 2026-05-10
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("model_runs") as batch_op:
        batch_op.add_column(
            sa.Column("model_type", sa.String(), nullable=False, server_default="gbdt")
        )


def downgrade() -> None:
    with op.batch_alter_table("model_runs") as batch_op:
        batch_op.drop_column("model_type")
