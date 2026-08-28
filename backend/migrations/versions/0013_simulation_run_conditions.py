"""add simulation_runs.conditions_json

Revision ID: 0013
Revises: 0012
Create Date: 2026-08-28

シミュレーションの保存レコードが「どの条件で走ったか」を残していなかった。
記録されていたのは budget / strategy / window / model_path だけで、

  * 確率モデルを使ったか (複勝の確信度・連系の確率がそのモデル由来か)
  * 複勝を買う確信度のしきい値
  * 履歴の無いレース (新馬戦など) を除外したか
  * 対象券種と券種ごとの 1 点あたり金額
  * 1 レースの上限と、買い目を組む頭数

はいずれも結果を大きく動かすのに保存されていない。設定を変えて回し直すと、
過去の run が何の条件だったのか分からなくなり、履歴どうしを比べられなかった。

列を個別に足さず JSON 1 本にしたのは、この種のノブが今後も増減するため
(実際 2026-08 だけで EV 閾値の廃止・確率モデルの追加・プリセットの再定義が起きた)。

既存行は NULL のまま = 「条件不明」。UI 側で「記録なし」と表示する。
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "simulation_runs",
        sa.Column("conditions_json", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("simulation_runs", "conditions_json")
