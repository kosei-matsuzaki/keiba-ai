"""Tests for ScraperStatus.missing_dates_count."""

from __future__ import annotations

import datetime

from fastapi.testclient import TestClient

from db.models.scrape_log import ScrapeLog
from scraper import stop_flag


def _add_ok_log(session, date_str: str, race_suffix: str = "0101") -> None:
    date_compact = date_str.replace("-", "")
    url = f"https://db.netkeiba.com/race/{date_compact}{race_suffix}/"
    session.add(ScrapeLog(
        url=url,
        fetched_at=f"{date_str}T10:00:00+00:00",
        status="ok",
    ))
    session.commit()


def test_missing_dates_count_all_missing(api_client: TestClient) -> None:
    """With no scrape_log entries, missing_dates_count equals the default range (30)."""
    stop_flag.clear_stopped()
    resp = api_client.get("/api/scraper/status")
    assert resp.status_code == 200
    data = resp.json()
    # All 30 days are missing
    assert data["missing_dates_count"] == 30


def test_missing_dates_count_with_recent_entries(
    app_with_temp_db,
    tmp_path,
) -> None:
    """5 recent ok log entries should reduce missing_dates_count from 30 to 25."""
    from core.paths import db_path
    from db.session import make_engine, session_scope

    engine = make_engine(db_path())
    today = datetime.date.today()

    with session_scope(engine) as session:
        for i in range(5):
            d = today - datetime.timedelta(days=i)
            _add_ok_log(session, d.isoformat())

    stop_flag.clear_stopped()
    with TestClient(app_with_temp_db) as client:
        resp = client.get("/api/scraper/status")

    assert resp.status_code == 200
    data = resp.json()
    # 5 days have ok entries → 30 - 5 = 25 missing
    assert data["missing_dates_count"] == 25


def test_missing_dates_count_custom_range(
    app_with_temp_db,
    tmp_path,
) -> None:
    """?range=7 uses a 7-day window."""
    from core.paths import db_path
    from db.session import make_engine, session_scope

    engine = make_engine(db_path())
    today = datetime.date.today()

    with session_scope(engine) as session:
        # Add ok entries for 2 of the last 7 days
        for i in range(2):
            d = today - datetime.timedelta(days=i)
            _add_ok_log(session, d.isoformat())

    stop_flag.clear_stopped()
    with TestClient(app_with_temp_db) as client:
        resp = client.get("/api/scraper/status?range=7")

    assert resp.status_code == 200
    data = resp.json()
    # 2 days completed → 7 - 2 = 5 missing
    assert data["missing_dates_count"] == 5


def test_missing_dates_count_error_logs_dont_count(
    app_with_temp_db,
    tmp_path,
) -> None:
    """Error-status scrape_log entries do not count as completed days."""
    from core.paths import db_path
    from db.session import make_engine, session_scope

    engine = make_engine(db_path())
    today = datetime.date.today()
    date_compact = today.strftime("%Y%m%d")

    with session_scope(engine) as session:
        session.add(ScrapeLog(
            url=f"https://db.netkeiba.com/race/{date_compact}0101/",
            fetched_at=f"{today.isoformat()}T10:00:00+00:00",
            status="error",
        ))
        session.commit()

    stop_flag.clear_stopped()
    with TestClient(app_with_temp_db) as client:
        resp = client.get("/api/scraper/status?range=7")

    assert resp.status_code == 200
    data = resp.json()
    # Error entry does not count → still 7 missing
    assert data["missing_dates_count"] == 7
