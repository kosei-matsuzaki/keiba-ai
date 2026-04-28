"""Integration test for the ingest job.

Uses monkeypatching to avoid any real HTTP requests.  The NetkeibaClient.fetch
method is replaced with a function that returns fixture HTML.

DB assertions use ORM (sqlalchemy.orm.Session) instead of sqlite3.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select

from keiba_ai.core.config import Settings
from keiba_ai.db.models.entry import Entry
from keiba_ai.db.models.race import Race
from keiba_ai.db.models.scrape_log import ScrapeLog
from keiba_ai.jobs.ingest import run_ingest
from keiba_ai.scraper.netkeiba import NetkeibaClient
from keiba_ai.scraper.rate_limiter import AsyncRateLimiter
from keiba_ai.scraper.robots import RobotsCache

FIXTURES = Path(__file__).parent / "fixtures"
CALENDAR_HTML = (FIXTURES / "race_calendar_20241228.html").read_text(encoding="utf-8")
RESULT_HTML = (FIXTURES / "race_result_202406010101.html").read_text(encoding="utf-8")

DATE = "2024-12-28"


def _build_fake_fetch(calendar_html: str, result_html: str):
    async def fake_fetch(url: str, *, use_cache: bool = True, cache_max_age_hours: float = 24 * 30) -> str:
        if "race_list" in url:
            return calendar_html
        return result_html
    return fake_fetch


@pytest.fixture()
def mock_client(monkeypatch) -> NetkeibaClient:
    import httpx
    settings = Settings(rate_min_seconds=0.0, rate_max_seconds=0.0)
    rate = AsyncRateLimiter(settings)
    robots = RobotsCache("TestAgent")
    http = httpx.AsyncClient()
    client = NetkeibaClient(rate, robots, http, settings)
    client.fetch = _build_fake_fetch(CALENDAR_HTML, RESULT_HTML)  # type: ignore[method-assign]
    return client


@pytest.mark.asyncio
async def test_ingest_inserts_races(db_session, mock_client, tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    counters = await run_ingest(DATE, mock_client, db_session, limit=2)

    assert counters["fetched"] == 2
    assert counters["errors"] == 0

    rows = db_session.execute(select(Race)).scalars().all()
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_ingest_inserts_entries(db_session, mock_client, tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    await run_ingest(DATE, mock_client, db_session, limit=1)

    entries = db_session.execute(select(Entry)).scalars().all()
    # fixture has 3 runners
    assert len(entries) == 3


@pytest.mark.asyncio
async def test_ingest_records_scrape_log(db_session, mock_client, tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    await run_ingest(DATE, mock_client, db_session, limit=1)

    logs = db_session.execute(select(ScrapeLog).where(ScrapeLog.status == "ok")).scalars().all()
    assert len(logs) >= 1


@pytest.mark.asyncio
async def test_ingest_skips_already_scraped(db_session, mock_client, tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    # Run once
    c1 = await run_ingest(DATE, mock_client, db_session, limit=2)
    assert c1["fetched"] == 2

    # Run again — both races should be skipped
    c2 = await run_ingest(DATE, mock_client, db_session, limit=2)
    assert c2["skipped"] == 2
    assert c2["fetched"] == 0


@pytest.mark.asyncio
async def test_ingest_date_stored_in_races(db_session, mock_client, tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    await run_ingest(DATE, mock_client, db_session, limit=1)

    row = db_session.execute(select(Race)).scalar_one()
    assert row.date == DATE


@pytest.mark.asyncio
async def test_ingest_stops_on_stop_flag(db_session, mock_client, tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("KEIBA_SCRAPER_STOP", "1")

    from keiba_ai.scraper.stop_flag import ScraperStopped
    with pytest.raises(ScraperStopped):
        await run_ingest(DATE, mock_client, db_session)
