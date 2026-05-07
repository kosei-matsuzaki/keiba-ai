"""add simulation_runs table for persisted backtest results

Revision ID: 0007
Revises: 0006
Create Date: 2026-05-07

Ledger 「シミュレーション」 タブの実行結果を保存するテーブル。
実行ごとに 1 行 insert され、上限 50 件 (古い順に削除) を persistence helper
側で維持する。

idempotency:
  FastAPI lifespan の Base.metadata.create_all(engine) (main.py:62) が
  alembic より先にこのテーブルを作成しているケースがある（alembic を一度も
  実行していない既存ユーザの DB が該当）。その場合 op.create_table が
  "table already exists" で失敗するため、inspector でテーブル / インデックスの
  存在を確認してから作成する。今後の "新テーブル追加系" migration でも
  同じガードを置くことを推奨する。
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "simulation_runs" not in inspector.get_table_names():
        op.create_table(
            "simulation_runs",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("created_at", sa.String(), nullable=False),
            sa.Column("budget", sa.Integer(), nullable=False),
            sa.Column("strategy", sa.String(), nullable=False),
            sa.Column("window_start", sa.String(), nullable=True),
            sa.Column("window_end", sa.String(), nullable=True),
            sa.Column("model_path", sa.String(), nullable=False),
            sa.Column("n_races", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("n_settled_races", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("final_bankroll", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("peak_bankroll", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("summary_json", sa.String(), nullable=False),
            sa.Column("by_bet_type_json", sa.String(), nullable=False),
            sa.Column("by_race_class_json", sa.String(), nullable=False),
            sa.Column("by_course_json", sa.String(), nullable=False),
            sa.Column("bankroll_timeseries_json", sa.String(), nullable=False),
            sa.PrimaryKeyConstraint("id", name="pk_simulation_runs"),
        )

    existing_indexes = {ix["name"] for ix in inspector.get_indexes("simulation_runs")}
    if "ix_simulation_runs_created_at" not in existing_indexes:
        op.create_index(
            "ix_simulation_runs_created_at",
            "simulation_runs",
            ["created_at"],
            unique=False,
        )


def downgrade() -> None:
    op.drop_index("ix_simulation_runs_created_at", table_name="simulation_runs")
    op.drop_table("simulation_runs")
