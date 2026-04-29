"""Scraper management endpoints: status, run, stop."""

from __future__ import annotations

from typing import Annotated

import httpx
from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from keiba_ai.api.deps import get_job_registry, get_session
from keiba_ai.api.jobs import JobRegistry
from keiba_ai.api.schemas import JobAccepted, ScraperRunRequest, ScraperStatus
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


@router.get("/scraper/status", response_model=ScraperStatus)
def get_scraper_status(
    session: Annotated[Session, Depends(get_session)],
    registry: Annotated[JobRegistry, Depends(get_job_registry)],
) -> ScraperStatus:
    # Latest fetched_at from scrape_log (status='ok')
    row = session.execute(
        select(func.max(ScrapeLog.fetched_at)).where(ScrapeLog.status == "ok")
    ).scalar()
    last_fetched = row if row else None

    return ScraperStatus(
        stopped=stop_flag.is_stopped(),
        last_fetched_date=last_fetched,
        missing_dates_count=None,  # M9 で本格実装
        current_job_id=registry.current_ingest_job_id(),
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
