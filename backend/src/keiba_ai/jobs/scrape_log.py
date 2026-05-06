"""ScrapeLog 共通ヘルパ。ingest と ingest_shutuba で共有する。"""

from __future__ import annotations

import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from keiba_ai.db.models.scrape_log import ScrapeLog


def already_scraped(session: Session, url: str) -> bool:
    """同一 URL が status='ok' で記録済みなら True。"""
    row = session.execute(
        select(ScrapeLog).where(ScrapeLog.url == url, ScrapeLog.status == "ok").limit(1)
    ).first()
    return row is not None


def record_scrape_log(
    session: Session,
    url: str,
    status: str,
    content_hash: str | None = None,
) -> None:
    """fetched_at を「今 UTC」に固定して ScrapeLog 行を追加する。"""
    fetched_at = datetime.datetime.now(datetime.UTC).isoformat()
    session.add(
        ScrapeLog(url=url, fetched_at=fetched_at, status=status, content_hash=content_hash)
    )
