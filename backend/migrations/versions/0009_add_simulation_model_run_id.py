"""tie simulation_runs to model_runs via model_run_id FK

Revision ID: 0009
Revises: 0008
Create Date: 2026-05-22

シミュレーションをモデルに紐づける。simulation_runs.model_run_id を追加し、
model_runs.id への FK (ON DELETE CASCADE / ON UPDATE CASCADE) を張る。

- ON DELETE CASCADE: モデル削除でそのバックテスト履歴も消える。
- ON UPDATE CASCADE: registry.renumber_model_ids が model_runs.id を振り直す
  際に子の参照が追従する。

既存 run は model_run_id を持たない (旧スキーマ)。ユーザー合意のもと移行時に
全削除してクリーンに作り直す (backfill しない)。これにより basename 突合の
脆さを避けつつ model_run_id を NOT NULL にできる。

idempotency: FastAPI lifespan の create_all が既に新スキーマ (model_run_id 付き)
でテーブルを作っているケースがあるため、カラム存在を確認してから追加する。
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    columns = {c["name"] for c in inspector.get_columns("simulation_runs")}
    if "model_run_id" in columns:
        # create_all が既に新スキーマで作成済み → 何もしない。
        return

    # 旧 run は紐づくモデルが不定なので全削除 (ユーザー合意済み)。
    # batch_alter_table が NOT NULL カラムを足せるよう、先に空にしておく。
    op.execute("DELETE FROM simulation_runs")

    # SQLite で FK 制約 (CASCADE 付き) を確実にスキーマへ刻むため batch recreate。
    with op.batch_alter_table("simulation_runs") as batch_op:
        batch_op.add_column(
            sa.Column(
                "model_run_id",
                sa.Integer(),
                sa.ForeignKey(
                    "model_runs.id",
                    name="fk_simulation_runs_model_run_id",
                    ondelete="CASCADE",
                    onupdate="CASCADE",
                ),
                nullable=False,
            )
        )

    existing_indexes = {ix["name"] for ix in inspector.get_indexes("simulation_runs")}
    if "ix_simulation_runs_model_run_id" not in existing_indexes:
        op.create_index(
            "ix_simulation_runs_model_run_id",
            "simulation_runs",
            ["model_run_id"],
            unique=False,
        )


def downgrade() -> None:
    op.drop_index(
        "ix_simulation_runs_model_run_id", table_name="simulation_runs"
    )
    with op.batch_alter_table("simulation_runs") as batch_op:
        batch_op.drop_column("model_run_id")
