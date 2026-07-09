"""Tests for jobs.ingest_results — 期間指定で確定レース(レース本体＋結果＋確定オッズ)を取込 (no network)."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from sqlalchemy import select

from core.config import Settings
from db.models.entry import Entry
from db.models.race import Race
from db.odds_db import init_odds_db, load_race_odds, make_odds_engine, odds_session_scope
from jobs.ingest_results import run_ingest_recent_results
from scraper.netkeiba import NetkeibaClient
from scraper.rate_limiter import AsyncRateLimiter
from scraper.robots import RobotsCache

FIXTURES = Path(__file__).parent / "fixtures"
RESULT_HTML = (FIXTURES / "race_result_202406010101.html").read_text(encoding="utf-8")

TODAY = datetime.date(2026, 6, 15)
YESTERDAY = "2026-06-14"
CENTRAL_RID = "202405010101"  # track code 05 (東京) = 中央
CAL_ONE = f'<html><body><a href="/race/{CENTRAL_RID}/">x</a></body></html>'
CAL_EMPTY = "<html><body>no races</body></html>"


def _odds_json(type_code: int) -> str:
    if type_code == 1:
        odds = {"1": {"01": ["2.0", "0.0", "1"]}, "2": {"01": ["1.1", "1.3", "1"]}}
    else:
        odds = {str(type_code): {"0102": ["5.0", "0.0", "1"]}}
    return json.dumps({"status": "result", "data": {"official_datetime": "2026-x", "odds": odds}})


def _build_fake_fetch(calendar_html: str, result_html: str):
    async def fake_fetch(
        url: str,
        *,
        use_cache: bool = True,
        write_to_cache: bool = True,
        cache_max_age_hours: float = 24 * 30,
    ) -> str:
        if "/race/list/" in url:
            return calendar_html
        if "api_get_jra_odds" in url:
            t = int(parse_qs(urlparse(url).query)["type"][0])
            return _odds_json(t)
        return result_html  # 結果ページ / horse detail 等

    return fake_fetch


def _client(calendar_html: str = CAL_ONE, result_html: str = RESULT_HTML) -> NetkeibaClient:
    s = Settings(rate_min_seconds=0.0, rate_max_seconds=0.0)
    c = NetkeibaClient(AsyncRateLimiter(s), RobotsCache("T"), httpx.AsyncClient(), s)
    c.fetch = _build_fake_fetch(calendar_html, result_html)  # type: ignore[method-assign]
    return c


@pytest.fixture
def odds_engine(tmp_path):
    e = make_odds_engine(tmp_path / "odds.db")
    init_odds_db(e)
    return e


@pytest.mark.asyncio
async def test_discovers_and_ingests_race_results_and_odds(
    db_session, odds_engine, tmp_path, monkeypatch
):
    """カレンダー発見で keiba.db 未登録のレースを作成し、結果＋確定オッズを取り込む。"""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    counters = await run_ingest_recent_results(
        _client(), db_session, odds_engine, start=YESTERDAY, end=YESTERDAY, today=TODAY
    )

    assert counters["results"] == 1
    # レース本体が作成される
    race = db_session.get(Race, CENTRAL_RID)
    assert race is not None
    entries = db_session.execute(select(Entry).where(Entry.race_id == CENTRAL_RID)).scalars().all()
    assert entries and any(e.finish_position is not None for e in entries)
    # 確定オッズ
    with odds_session_scope(odds_engine) as s:
        assert "単勝" in load_race_odds(s, CENTRAL_RID)


@pytest.mark.asyncio
async def test_includes_db_races_when_calendar_empty(
    db_session, odds_engine, tmp_path, monkeypatch
):
    """カレンダーが空でも keiba.db 既存レースは対象になる。"""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    db_session.add(Race(race_id="202603010101", date=YESTERDAY, course="東京", surface="芝", distance=2000))
    db_session.commit()

    counters = await run_ingest_recent_results(
        _client(calendar_html=CAL_EMPTY), db_session, odds_engine,
        start=YESTERDAY, end=YESTERDAY, today=TODAY,
    )
    assert counters["races"] == 1
    assert counters["results"] == 1


@pytest.mark.asyncio
async def test_skips_already_ingested(db_session, odds_engine, tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    await run_ingest_recent_results(
        _client(), db_session, odds_engine, start=YESTERDAY, end=YESTERDAY, today=TODAY
    )
    c2 = await run_ingest_recent_results(
        _client(), db_session, odds_engine, start=YESTERDAY, end=YESTERDAY, today=TODAY
    )
    assert c2["results"] == 0  # 結果は取得済み


@pytest.mark.asyncio
async def test_pending_when_results_not_archived(db_session, odds_engine, tmp_path, monkeypatch):
    """結果ページ未掲載（未アーカイブ）なら pending・エラーにせず、確定オッズは取得する。"""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    counters = await run_ingest_recent_results(
        _client(result_html="<html><body>not archived</body></html>"),
        db_session, odds_engine, start=YESTERDAY, end=YESTERDAY, today=TODAY,
    )
    assert counters["results"] == 0
    assert counters["pending"] == 1
    assert counters["errors"] == 0
    assert counters["odds"] == 1


@pytest.mark.asyncio
async def test_excludes_today(db_session, odds_engine, tmp_path, monkeypatch):
    """今日は未確定のため対象外（範囲は昨日までにクランプ）。"""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    counters = await run_ingest_recent_results(
        _client(calendar_html=CAL_EMPTY), db_session, odds_engine,
        start=TODAY.isoformat(), end=TODAY.isoformat(), today=TODAY,
    )
    assert counters["races"] == 0
