"""add indexes on horses.sire / horses.dam

Revision ID: 0012
Revises: 0011
Create Date: 2026-06-14

系統特徴量 (features/extractors/pedigree.py の compute_pedigree_features) は
`WHERE horses.sire == ?` / `WHERE horses.dam == ?` で産駒勝率を集計する。
これらの列に index が無いと、1 クエリごとに entries (~50万行) を全件スキャンし、
1 レース推論 (15頭 × 父母 2 クエリ = 30 回) で 15 秒前後かかる。フロントの
HTTP クライアント (ky, 既定 10s) がタイムアウトし、AI 予想スコアが入らない /
推奨買い目が timeout する不具合の原因となっていた。

index 追加で `build_inference_frame` は 15.3s → 1.4s に短縮される。
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index("ix_horses_sire", "horses", ["sire"])
    op.create_index("ix_horses_dam", "horses", ["dam"])


def downgrade() -> None:
    op.drop_index("ix_horses_dam", table_name="horses")
    op.drop_index("ix_horses_sire", table_name="horses")
