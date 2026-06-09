"""change model_runs.model_type server_default from gbdt to nn

Revision ID: 0011
Revises: 0010
Create Date: 2026-06-10

GBDT を撤去し NN 専用化したため、model_type のデフォルトを "nn" に揃える。
既存行はすべて model_type を明示設定済みのため値の書き換えは不要で、新規行の
デフォルトのみを切り替える。列自体は履歴互換のため残置する。
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("model_runs") as batch_op:
        batch_op.alter_column(
            "model_type",
            existing_type=sa.String(),
            server_default="nn",
            existing_nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("model_runs") as batch_op:
        batch_op.alter_column(
            "model_type",
            existing_type=sa.String(),
            server_default="gbdt",
            existing_nullable=False,
        )
