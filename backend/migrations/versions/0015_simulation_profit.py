"""simulation_runs: 資産 → 1 レース予算と累計損益

Revision ID: 0015
Revises: 0014
Create Date: 2026-09-01

シミュレーションを RACE 画面と同じ仕組みに揃えた。入力は「1 レースに使う上限」
だけになり、初期資産・賭け金の決め方 (定額/複利)・戦略プリセットは無くなった。
結果も資産残高ではなく **0 から始まる累計損益** で持つ。

  budget                   → race_budget            (初期資産 → 1 レースの上限)
  final_bankroll           → final_profit           (残高 → 0 起点の損益)
  peak_bankroll            → peak_profit
  bankroll_timeseries_json → profit_timeseries_json
  strategy                 → 削除 (プリセットが無くなった)

**既存行は削除する。** 旧行は「初期資産 10 万〜1000 万・複利/定額・戦略プリセット」
という別ルールで走った結果で、列名だけ付け替えても数字の意味が変わらない
(final_bankroll 8,184,150 を「損益」として読むと嘘になる)。シミュレーションは
いつでも回し直せるので、誤読するデータを残すより消す方が安全と判断した。
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0015"
down_revision: str | None = "0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 旧ルールで走った結果は意味が変わるため捨てる (上の docstring 参照)
    op.execute(sa.text("DELETE FROM simulation_runs"))
    with op.batch_alter_table("simulation_runs") as batch:
        batch.alter_column("budget", new_column_name="race_budget")
        batch.alter_column("final_bankroll", new_column_name="final_profit")
        batch.alter_column("peak_bankroll", new_column_name="peak_profit")
        batch.alter_column(
            "bankroll_timeseries_json", new_column_name="profit_timeseries_json"
        )
        batch.drop_column("strategy")


def downgrade() -> None:
    op.execute(sa.text("DELETE FROM simulation_runs"))
    with op.batch_alter_table("simulation_runs") as batch:
        batch.add_column(
            sa.Column("strategy", sa.String(), nullable=False, server_default="balanced")
        )
        batch.alter_column("race_budget", new_column_name="budget")
        batch.alter_column("final_profit", new_column_name="final_bankroll")
        batch.alter_column("peak_profit", new_column_name="peak_bankroll")
        batch.alter_column(
            "profit_timeseries_json", new_column_name="bankroll_timeseries_json"
        )
