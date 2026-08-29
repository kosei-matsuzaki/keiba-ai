"""add bet_records.conditions_json

Revision ID: 0014
Revises: 0013
Create Date: 2026-08-29

購入記録が「どの条件で出た買い目か」を残していなかった。記録されていたのは
券種・買い目・金額・source だけで、

  * 買い目を決めたモデル (active) がどれか
  * 確からしさを出すモデル (確率モデル) を使っていたか
  * 複勝を買う確信度のしきい値
  * 単勝のオッズ下限・券種ごとの 1 点あたり金額

はいずれも買い目を左右するのに保存されない。モデルを差し替えたり設定を変えたり
すると、**過去の購入記録がどの条件で出たものか分からなくなり、実績を評価できない**。
シミュレーションには 0013 で同じ仕組みを入れており、実運用の記録にも要る。

**登録時点の設定を写す**方式にした (クライアントに送らせない)。推奨買目を見てから
記録するまでに設定が変わっていれば理論上ズレるが、クライアントの申告を信じるより
確実で、実運用では両者は数秒差。

既存行は NULL のまま = 「条件の記録なし」。
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0014"
down_revision: str | None = "0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "bet_records",
        sa.Column("conditions_json", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("bet_records", "conditions_json")
