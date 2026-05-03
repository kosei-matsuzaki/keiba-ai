"""add index on scrape_log.fetched_at

Revision ID: 0003
Revises: 0002
Create Date: 2026-05-03

UI の `/api/scraper/recent_activity` は `WHERE fetched_at >= cutoff` で
直近 N 分の行を引くが、scrape_log は ingest 中に高頻度 INSERT され
数万行に達するため、index がないと毎回 full scan になり、UI ポーリング
時に WAL ロック待機で UI が応答なしになる症状の主因の一つとなる。
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_scrape_log_fetched_at",
        "scrape_log",
        ["fetched_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_scrape_log_fetched_at", table_name="scrape_log")
