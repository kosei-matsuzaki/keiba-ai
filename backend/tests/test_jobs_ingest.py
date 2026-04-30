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
from keiba_ai.db.models.horse import Horse
from keiba_ai.db.models.jockey import Jockey
from keiba_ai.db.models.race import Race
from keiba_ai.db.models.scrape_log import ScrapeLog
from keiba_ai.db.models.trainer import Trainer
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


@pytest.mark.asyncio
async def test_ingest_saves_horse_names(db_session, mock_client, tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    await run_ingest(DATE, mock_client, db_session, limit=1)

    horses = db_session.execute(select(Horse)).scalars().all()
    names = {h.name for h in horses}
    assert "ドウデュース" in names
    assert "タスティエーラ" in names
    assert "シャフリヤール" in names


@pytest.mark.asyncio
async def test_ingest_saves_jockey_names(db_session, mock_client, tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    await run_ingest(DATE, mock_client, db_session, limit=1)

    jockeys = db_session.execute(select(Jockey)).scalars().all()
    names = {j.name for j in jockeys}
    assert "武豊" in names


@pytest.mark.asyncio
async def test_ingest_saves_trainer_names(db_session, mock_client, tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    await run_ingest(DATE, mock_client, db_session, limit=1)

    trainers = db_session.execute(select(Trainer)).scalars().all()
    names = {t.name for t in trainers}
    assert "友道康夫" in names


@pytest.mark.asyncio
async def test_ingest_saves_agari_3f(db_session, mock_client, tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    await run_ingest(DATE, mock_client, db_session, limit=1)

    entries = db_session.execute(select(Entry)).scalars().all()
    agari_values = [e.agari_3f for e in entries]
    assert any(v is not None for v in agari_values)
    # first entry (finish_position=1) has agari_3f=35.1 in fixture
    first = next(e for e in entries if e.finish_position == 1)
    assert abs(first.agari_3f - 35.1) < 0.01


@pytest.mark.asyncio
async def test_ingest_saves_passing(db_session, mock_client, tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    await run_ingest(DATE, mock_client, db_session, limit=1)

    entries = db_session.execute(select(Entry)).scalars().all()
    first = next(e for e in entries if e.finish_position == 1)
    assert first.passing == "2-2"


def test_ensure_masters_coalesce_preserves_existing_name_when_new_is_none(db_session):
    """COALESCE(excluded.name, Horse.name) は新規 None で既存値を維持する。

    回帰防止: PR-A code-reviewer が「既存 'X' + 新規 None で NULL になる」と
    誤認したため、SQL COALESCE の挙動を直接ロックする。
    """
    from keiba_ai.jobs.ingest import _ensure_masters
    from keiba_ai.scraper.parsers.race_result import ParsedEntry, ParsedRaceResult

    # 1) 既存馬を name 入りで insert
    db_session.add(Horse(horse_id="HORSE_X", name="ExistingName"))
    db_session.commit()

    # 2) horse_name=None の Entry を持つ ParsedRaceResult で再 ingest
    parsed = ParsedRaceResult(
        race_id="R1",
        date="2024-01-01",
        course="中山",
        surface="芝",
        distance=1600,
        entries=[ParsedEntry(race_id="R1", horse_id="HORSE_X", horse_name=None)],
    )
    _ensure_masters(db_session, parsed)
    db_session.commit()

    # 既存 name は維持される
    horse = db_session.get(Horse, "HORSE_X")
    assert horse.name == "ExistingName"


def test_ensure_masters_coalesce_fills_missing_name(db_session):
    """既存 NULL + 新規 'X' で name が補完される。"""
    from keiba_ai.jobs.ingest import _ensure_masters
    from keiba_ai.scraper.parsers.race_result import ParsedEntry, ParsedRaceResult

    db_session.add(Horse(horse_id="HORSE_Y", name=None))
    db_session.commit()

    parsed = ParsedRaceResult(
        race_id="R2",
        date="2024-01-01",
        course="中山",
        surface="芝",
        distance=1600,
        entries=[ParsedEntry(race_id="R2", horse_id="HORSE_Y", horse_name="NewName")],
    )
    _ensure_masters(db_session, parsed)
    db_session.commit()

    horse = db_session.get(Horse, "HORSE_Y")
    assert horse.name == "NewName"


def test_ensure_masters_coalesce_overwrites_when_both_have_value(db_session):
    """既存 'X' + 新規 'Y' は上書き（COALESCE 第一引数優先のため）。

    現仕様: netkeiba 側の表記揺れがあった場合、最新を採用する。
    将来「name は一度入ったら不変」に変えたい場合は引数順を反転する。
    """
    from keiba_ai.jobs.ingest import _ensure_masters
    from keiba_ai.scraper.parsers.race_result import ParsedEntry, ParsedRaceResult

    db_session.add(Horse(horse_id="HORSE_Z", name="OldName"))
    db_session.commit()

    parsed = ParsedRaceResult(
        race_id="R3",
        date="2024-01-01",
        course="中山",
        surface="芝",
        distance=1600,
        entries=[ParsedEntry(race_id="R3", horse_id="HORSE_Z", horse_name="NewName")],
    )
    _ensure_masters(db_session, parsed)
    db_session.commit()

    horse = db_session.get(Horse, "HORSE_Z")
    assert horse.name == "NewName"
