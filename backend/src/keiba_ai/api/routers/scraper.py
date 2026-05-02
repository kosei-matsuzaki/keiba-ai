"""Scraper management endpoints: status, recent activity, run, stop."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from typing import Annotated
from zoneinfo import ZoneInfo

# netkeiba の race_id は JST 基準で YYYYMMDD を含むので JST 起算の date を使う
_JST = ZoneInfo("Asia/Tokyo")

import httpx
from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from keiba_ai.api.deps import get_job_registry, get_session
from keiba_ai.api.jobs import JobRegistry
from keiba_ai.api.schemas import (
    JobAccepted,
    ScraperRecentActivity,
    ScraperRunRequest,
    ScraperStatus,
)
from keiba_ai.core.config import load_settings
from keiba_ai.core.paths import db_path
from keiba_ai.db.models.scrape_log import ScrapeLog
from keiba_ai.db.session import make_engine, session_scope
from keiba_ai.jobs.ingest import run_ingest
from keiba_ai.scraper import stop_flag
from keiba_ai.scraper.netkeiba import NetkeibaClient
from keiba_ai.scraper.rate_limiter import AsyncRateLimiter
from keiba_ai.scraper.robots import RobotsCache

router = APIRouter()

_RESULT_URL_PREFIX = "https://db.netkeiba.com/race/"
_RACE_ID_FROM_URL_RE = re.compile(r"/race/(\d{12})")


def _count_missing_dates(session: Session, days: int = 30) -> int:
    """Count days in the recent window that have no 'ok' scrape_log entries.

    A day is considered 'completed' if at least one result URL for that date's
    compact prefix (YYYYMMDD) has status='ok' in scrape_log.
    """
    today = datetime.now(_JST).date()
    start = today - timedelta(days=days - 1)  # inclusive range of `days` days

    # Collect distinct date prefixes that have at least one 'ok' entry.
    # Race result URLs are https://db.netkeiba.com/race/<YYYYMMDD><suffix>/
    # We extract the 8-char date prefix from urls matching the base prefix.
    rows = session.execute(
        select(ScrapeLog.url).where(
            ScrapeLog.url.like(f"{_RESULT_URL_PREFIX}%"),
            ScrapeLog.status == "ok",
        )
    ).scalars().all()

    completed_dates: set[str] = set()
    prefix_len = len(_RESULT_URL_PREFIX)
    for url in rows:
        # URL format: https://db.netkeiba.com/race/YYYYMMDDXXXX/
        suffix = url[prefix_len:]  # e.g. "202412280101/"
        if len(suffix) >= 8:
            date_str = suffix[:8]  # "20241228"
            try:
                d = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
                if start <= d <= today:
                    completed_dates.add(date_str)
            except ValueError:
                pass

    missing = days - len(completed_dates)
    return max(missing, 0)


@router.get("/scraper/status", response_model=ScraperStatus)
def get_scraper_status(
    session: Annotated[Session, Depends(get_session)],
    registry: Annotated[JobRegistry, Depends(get_job_registry)],
    range_days: int = Query(default=30, ge=1, le=365, alias="range"),
) -> ScraperStatus:
    # Latest fetched_at from scrape_log (status='ok')
    row = session.execute(
        select(func.max(ScrapeLog.fetched_at)).where(ScrapeLog.status == "ok")
    ).scalar()
    last_fetched = row if row else None

    missing = _count_missing_dates(session, days=range_days)

    return ScraperStatus(
        stopped=stop_flag.is_stopped(),
        last_fetched_date=last_fetched,
        missing_dates_count=missing,
        current_job_id=registry.current_ingest_job_id(),
    )


@router.get("/scraper/recent_activity", response_model=ScraperRecentActivity)
def recent_activity(
    session: Annotated[Session, Depends(get_session)],
    minutes: int = Query(default=10, ge=1, le=1440),
) -> ScraperRecentActivity:
    """Aggregate scrape_log over the last N minutes.

    Designed to surface CLI-driven ingest progress alongside UI-launched jobs.
    fetched_at is stored as UTC ISO 8601, so a string `>=` cutoff works.
    """
    cutoff = (datetime.now(UTC) - timedelta(minutes=minutes)).isoformat()

    rows = session.execute(
        select(ScrapeLog.url, ScrapeLog.status, ScrapeLog.fetched_at)
        .where(ScrapeLog.fetched_at >= cutoff)
        .order_by(ScrapeLog.fetched_at.desc())
    ).all()

    total = len(rows)
    ok = sum(1 for r in rows if r.status == "ok")
    err = sum(1 for r in rows if r.status == "error")
    skipped = sum(1 for r in rows if r.status == "skipped")

    latest_fetched_at = rows[0].fetched_at if rows else None
    latest_race_id: str | None = None
    if rows:
        m = _RACE_ID_FROM_URL_RE.search(rows[0].url)
        if m:
            latest_race_id = m.group(1)

    rate_per_min = ok / max(minutes, 1)

    return ScraperRecentActivity(
        window_minutes=minutes,
        total_fetched=total,
        ok_count=ok,
        error_count=err,
        skipped_count=skipped,
        rate_per_min=round(rate_per_min, 2),
        latest_fetched_at=latest_fetched_at,
        latest_race_id=latest_race_id,
    )


@router.post("/scraper/run", response_model=JobAccepted)
async def run_scraper(
    body: ScraperRunRequest,
    session: Annotated[Session, Depends(get_session)],  # noqa: ARG001
    registry: Annotated[JobRegistry, Depends(get_job_registry)],
) -> JobAccepted:
    date_str = body.date
    limit = body.limit

    async def _coro() -> None:
        settings = load_settings()
        rate_limiter = AsyncRateLimiter(settings)
        robots_cache = RobotsCache(settings.user_agent)
        engine = make_engine(db_path())

        async with httpx.AsyncClient() as http_client:
            client = NetkeibaClient(rate_limiter, robots_cache, http_client, settings)
            with session_scope(engine) as s:
                await run_ingest(date_str, client, s, limit=limit)

    info = registry.start("ingest", _coro)
    return JobAccepted(
        job_id=info.job_id,
        status=info.status,
        started_at=info.started_at,
    )


@router.post("/scraper/stop", status_code=200)
def stop_scraper() -> dict:
    stop_flag.set_stopped()
    return {"ok": True}
