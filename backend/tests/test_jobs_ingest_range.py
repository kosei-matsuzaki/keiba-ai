"""Tests for jobs/ingest_range.py — range ingest with resume support."""

from __future__ import annotations

import datetime

import pytest
from sqlalchemy import select

from keiba_ai.core.config import Settings
from keiba_ai.db.models.scrape_log import ScrapeLog
from keiba_ai.jobs.ingest_range import is_date_completed, run_range
from keiba_ai.scraper.netkeiba import NetkeibaClient
from keiba_ai.scraper.rate_limiter import AsyncRateLimiter
from keiba_ai.scraper.robots import RobotsCache
from tests.conftest import FIXTURES_DIR

CALENDAR_HTML = (FIXTURES_DIR / "race_calendar_20241228.html").read_text(encoding="utf-8")
RESULT_HTML = (FIXTURES_DIR / "race_result_202406010101.html").read_text(encoding="utf-8")


def _build_fake_fetch(calendar_html: str, result_html: str):
    async def fake_fetch(url: str, *, use_cache: bool = True, cache_max_age_hours: float = 24) -> str:
        if "/race/list/" in url:
            return calendar_html
        if "/horse/" in url:
            raise RuntimeError(f"basic mock has no horse fixture for {url}")
        return result_html
    return fake_fetch


@pytest.fixture()
def mock_client() -> NetkeibaClient:
    import httpx
    settings = Settings(rate_min_seconds=0.0, rate_max_seconds=0.0)
    rate = AsyncRateLimiter(settings)
    robots = RobotsCache("TestAgent")
    http = httpx.AsyncClient()
    client = NetkeibaClient(rate, robots, http, settings)
    client.fetch = _build_fake_fetch(CALENDAR_HTML, RESULT_HTML)  # type: ignore[method-assign]
    return client


def test_is_date_completed_false_when_no_logs(db_session):
    """Fresh DB has no scrape logs; every date is incomplete."""
    assert is_date_completed(db_session, "2024-12-28") is False


def test_is_date_completed_true_when_ok_log_exists(db_session):
    """A date is complete when at least one ok entry exists for that date prefix."""
    db_session.add(ScrapeLog(
        url="https://db.netkeiba.com/race/202412280101/",
        fetched_at="2024-12-28T10:00:00+00:00",
        status="ok",
    ))
    db_session.commit()
    assert is_date_completed(db_session, "2024-12-28") is True


def test_is_date_completed_false_when_only_error_log(db_session):
    """An error-status log does not count as completed."""
    db_session.add(ScrapeLog(
        url="https://db.netkeiba.com/race/202412280101/",
        fetched_at="2024-12-28T10:00:00+00:00",
        status="error",
    ))
    db_session.commit()
    assert is_date_completed(db_session, "2024-12-28") is False


@pytest.mark.asyncio
async def test_run_range_ingests_all_dates(in_memory_engine, mock_client, tmp_path, monkeypatch):
    """run_range over 3 consecutive dates processes each date (no resume skips on first run)."""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))

    today = datetime.date.today()
    # Use a range 10 days ago to avoid overlap with is_date_completed clock issues
    base = today - datetime.timedelta(days=10)
    start = base.isoformat()
    end = (base + datetime.timedelta(days=2)).isoformat()

    totals = await run_range(start, end, mock_client, in_memory_engine, limit_per_day=1)

    # All 3 dates should be attempted (none skipped via resume logic)
    assert totals["dates_done"] == 3
    assert totals["dates_skipped"] == 0

    # At least 1 ok log entry exists (first date's fetch; subsequent dates skip already-scraped race)
    from sqlalchemy.orm import Session
    with Session(in_memory_engine) as s:
        logs = s.execute(select(ScrapeLog).where(ScrapeLog.status == "ok")).scalars().all()
    assert len(logs) >= 1


@pytest.mark.asyncio
async def test_run_range_clears_misc_cache_after_each_day(
    in_memory_engine, mock_client, tmp_path, monkeypatch
):
    """misc/ キャッシュは各日完了後に自動削除される (default 動作)。"""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("KEIBA_KEEP_MISC_CACHE", raising=False)

    from keiba_ai.scraper import cache as cache_module

    # 事前に misc/ にダミーファイルを置いておく (前回 ingest の残骸を模擬)
    cache_module.write_cache("https://db.netkeiba.com/horse/2019105293/", "leftover")
    misc_dir = tmp_path / "raw" / "misc"
    assert any(misc_dir.iterdir()), "precondition: misc/ should have leftover"

    base = datetime.date.today() - datetime.timedelta(days=10)
    start = base.isoformat()
    end = base.isoformat()  # 1 day only

    await run_range(start, end, mock_client, in_memory_engine, limit_per_day=1)

    # 各日完了後の cleanup により misc/ は空 (ディレクトリ自体は残る)
    leftover_files = [f for f in misc_dir.iterdir() if f.is_file()] if misc_dir.exists() else []
    assert leftover_files == [], f"misc/ should be cleared, found: {leftover_files}"


@pytest.mark.asyncio
async def test_run_range_keeps_misc_cache_when_opt_out(
    in_memory_engine, mock_client, tmp_path, monkeypatch
):
    """KEIBA_KEEP_MISC_CACHE=1 で misc/ 自動削除を無効化できる。"""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("KEIBA_KEEP_MISC_CACHE", "1")

    from keiba_ai.scraper import cache as cache_module

    cache_module.write_cache("https://db.netkeiba.com/horse/2019105293/", "keep me")
    misc_dir = tmp_path / "raw" / "misc"

    base = datetime.date.today() - datetime.timedelta(days=10)
    start = base.isoformat()
    end = base.isoformat()

    await run_range(start, end, mock_client, in_memory_engine, limit_per_day=1)

    leftover_files = [f for f in misc_dir.iterdir() if f.is_file()]
    assert len(leftover_files) >= 1, "misc/ should be retained when KEIBA_KEEP_MISC_CACHE=1"


@pytest.mark.asyncio
async def test_run_range_skips_completed_dates(in_memory_engine, mock_client, tmp_path, monkeypatch):
    """Dates that already have ok scrape_log entries are skipped on re-run."""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))

    base = datetime.date.today() - datetime.timedelta(days=10)
    dates = [base + datetime.timedelta(days=i) for i in range(3)]
    start = dates[0].isoformat()
    end = dates[-1].isoformat()

    # Pre-seed scrape_log with ok entries for each date using matching race_id prefix
    from sqlalchemy.orm import Session
    with Session(in_memory_engine) as s:
        for d in dates:
            date_compact = d.strftime("%Y%m%d")
            s.add(ScrapeLog(
                url=f"https://db.netkeiba.com/race/{date_compact}0101/",
                fetched_at=f"{d.isoformat()}T10:00:00+00:00",
                status="ok",
            ))
        s.commit()

    # All 3 dates should be skipped because is_date_completed returns True for each
    t = await run_range(start, end, mock_client, in_memory_engine, limit_per_day=1)
    assert t["dates_skipped"] == 3
    assert t["dates_done"] == 0
