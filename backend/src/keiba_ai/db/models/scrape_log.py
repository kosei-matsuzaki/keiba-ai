"""ScrapeLog ORM model — スクレイピング実行ログ。"""

from __future__ import annotations

from sqlalchemy import Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from keiba_ai.db.base import Base


class ScrapeLog(Base):
    __tablename__ = "scrape_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(String, nullable=False)
    fetched_at: Mapped[str] = mapped_column(String, nullable=False)  # ISO 8601
    status: Mapped[str] = mapped_column(String, nullable=False)      # 'ok' | 'error' | 'skipped'
    etag: Mapped[str | None] = mapped_column(String)
    content_hash: Mapped[str | None] = mapped_column(String)         # SHA-256

    __table_args__ = (
        Index("ix_scrape_log_url_status", "url", "status"),
        # /api/scraper/recent_activity の WHERE fetched_at >= cutoff で
        # full scan を避けるための index (migration 0003)。
        Index("ix_scrape_log_fetched_at", "fetched_at"),
    )
