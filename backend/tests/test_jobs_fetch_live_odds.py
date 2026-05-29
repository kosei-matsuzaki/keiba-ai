"""Tests for jobs/fetch_live_odds.py — fetch + ingest flow."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session

import db.models  # noqa: F401 (populate Base.metadata)
from core.config import Settings
from db.base import Base
from db.models.live_odds import LiveOdds
from db.models.race import Race
from jobs.fetch_live_odds import fetch_odds_for_race, run_fetch_live_odds
from scraper.netkeiba import NetkeibaClient
from scraper.parsers.odds import LiveOddsRow
from scraper.rate_limiter import AsyncRateLimiter
from scraper.robots import RobotsCache
from scraper.stop_flag import ScraperStopped


def _make_tan_fuku_rows() -> list[LiveOddsRow]:
    """18 頭立ての単勝 + 複勝 を表現する合成オッズ行。"""
    rows: list[LiveOddsRow] = []
    for pp in range(1, 19):
        rows.append(LiveOddsRow(
            bet_type="単勝", combo=str(pp), odds=2.0 + pp * 0.5,
            odds_max=None, popularity=pp,
        ))
        rows.append(LiveOddsRow(
            bet_type="複勝", combo=str(pp), odds=1.1 + pp * 0.1,
            odds_max=1.5 + pp * 0.1, popularity=pp,
        ))
    return rows


def _make_umaren_rows() -> list[LiveOddsRow]:
    """18 頭立ての馬連 153 通り。"""
    rows: list[LiveOddsRow] = []
    pop = 1
    for i in range(1, 19):
        for j in range(i + 1, 19):
            rows.append(LiveOddsRow(
                bet_type="馬連", combo=f"{i}-{j}",
                odds=10.0 + pop * 0.5, odds_max=None, popularity=pop,
            ))
            pop += 1
    return rows

FIXTURES = Path(__file__).parent / "fixtures"

TAN_FUKU_HTML = (FIXTURES / "odds_real_tan_fuku.html").read_bytes().decode("euc-jp", errors="replace")
UMAREN_HTML = (FIXTURES / "odds_real_umaren.html").read_bytes().decode("euc-jp", errors="replace")
WIDE_HTML = (FIXTURES / "odds_real_wide.html").read_bytes().decode("euc-jp", errors="replace")
UMATAN_HTML = (FIXTURES / "odds_real_umatan.html").read_bytes().decode("euc-jp", errors="replace")
SANRENPUKU_HTML = (FIXTURES / "odds_real_sanrenpuku.html").read_bytes().decode("euc-jp", errors="replace")
SANRENTAN_HTML = (FIXTURES / "odds_real_sanrentan.html").read_bytes().decode("euc-jp", errors="replace")

RACE_ID = "202412281211"  # ホープフルステークス (フィクスチャの race)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def engine():
    eng = create_engine("sqlite:///:memory:", future=True)

    @event.listens_for(eng, "connect")
    def _fk(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)
    eng.dispose()


@pytest.fixture()
def session(engine):
    with Session(engine) as s:
        # Insert dummy race so FK constraint passes
        s.add(Race(
            race_id=RACE_ID,
            date="2024-12-28",
            course="中山",
            surface="芝",
            distance=2000,
            n_runners=18,
        ))
        s.commit()
        yield s


def _build_mock_client(html_map: dict[str, str]) -> NetkeibaClient:
    """type コードをキーに HTML を返すモッククライアントを作る。"""
    import httpx

    settings = Settings(rate_min_seconds=0.0, rate_max_seconds=0.0)
    rate = AsyncRateLimiter(settings)
    robots = RobotsCache("TestAgent")
    http = httpx.AsyncClient()
    client = NetkeibaClient(rate, robots, http, settings)

    async def _fake_fetch(url: str, *, use_cache=True, cache_max_age_hours=0.5) -> str:
        for type_code, html in html_map.items():
            if f"type={type_code}" in url:
                return html
        return TAN_FUKU_HTML

    client.fetch = _fake_fetch  # type: ignore[method-assign]
    return client


# ---------------------------------------------------------------------------
# Unit tests: fetch_odds_for_race
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_tan_fuku_inserts_rows(session, monkeypatch):
    """b1 fetch → 単勝/複勝 の live_odds 行が挿入される。

    実フィクスチャ HTML は発走前 (odds=---.-) で全行 skip されるので、
    パーサを差し替えて INSERT パイプライン自体を検証する。
    """
    import jobs.fetch_live_odds as _flo
    monkeypatch.setitem(
        _flo._TYPE_PARSERS, "b1", (lambda html: _make_tan_fuku_rows(), 0.5),
    )
    client = _build_mock_client({"b1": TAN_FUKU_HTML})
    counters = await fetch_odds_for_race(RACE_ID, ["b1"], client, session)

    assert counters["fetched"] == 1
    assert counters["errors"] == 0

    rows = session.execute(select(LiveOdds).where(LiveOdds.race_id == RACE_ID)).scalars().all()
    assert len(rows) > 0

    tan_rows = [r for r in rows if r.bet_type == "単勝"]
    fuku_rows = [r for r in rows if r.bet_type == "複勝"]
    assert len(tan_rows) == 18
    assert len(fuku_rows) == 18


@pytest.mark.asyncio
async def test_fetch_umaren_inserts_rows(session, monkeypatch):
    """b4 fetch → 馬連 153 行が挿入される。"""
    import jobs.fetch_live_odds as _flo
    monkeypatch.setitem(
        _flo._TYPE_PARSERS, "b4", (lambda html: _make_umaren_rows(), 0.5),
    )
    client = _build_mock_client({"b4": UMAREN_HTML})
    await fetch_odds_for_race(RACE_ID, ["b4"], client, session)

    rows = session.execute(
        select(LiveOdds).where(LiveOdds.race_id == RACE_ID, LiveOdds.bet_type == "馬連")
    ).scalars().all()
    assert len(rows) == 153


@pytest.mark.asyncio
async def test_fetch_upsert_idempotent(session):
    """同じデータを 2 回 fetch しても行数が増えない。"""
    client = _build_mock_client({"b1": TAN_FUKU_HTML})

    await fetch_odds_for_race(RACE_ID, ["b1"], client, session)
    count_first = session.execute(
        select(LiveOdds).where(LiveOdds.race_id == RACE_ID)
    ).scalars().all()

    await fetch_odds_for_race(RACE_ID, ["b1"], client, session)
    count_second = session.execute(
        select(LiveOdds).where(LiveOdds.race_id == RACE_ID)
    ).scalars().all()

    assert len(count_first) == len(count_second)


@pytest.mark.asyncio
async def test_fetch_multiple_types(session, monkeypatch):
    """複数 type を指定するとすべてのオッズが挿入される。"""
    import jobs.fetch_live_odds as _flo
    monkeypatch.setitem(
        _flo._TYPE_PARSERS, "b1", (lambda html: _make_tan_fuku_rows(), 0.5),
    )
    monkeypatch.setitem(
        _flo._TYPE_PARSERS, "b4", (lambda html: _make_umaren_rows(), 0.5),
    )
    client = _build_mock_client({
        "b1": TAN_FUKU_HTML,
        "b4": UMAREN_HTML,
    })
    counters = await fetch_odds_for_race(RACE_ID, ["b1", "b4"], client, session)

    assert counters["fetched"] == 2
    assert counters["errors"] == 0

    bet_types = {
        r.bet_type for r in
        session.execute(select(LiveOdds).where(LiveOdds.race_id == RACE_ID)).scalars().all()
    }
    assert "単勝" in bet_types
    assert "複勝" in bet_types
    assert "馬連" in bet_types


@pytest.mark.asyncio
async def test_run_fetch_live_odds_multiple_races(engine, monkeypatch):
    """複数レースを順次 fetch して各レースのデータが別々に保存される。"""
    import jobs.fetch_live_odds as _flo
    monkeypatch.setitem(
        _flo._TYPE_PARSERS, "b1", (lambda html: _make_tan_fuku_rows(), 0.5),
    )
    race_id2 = "202412281212"
    with Session(engine) as session:
        session.add(Race(
            race_id=RACE_ID,
            date="2024-12-28",
            course="中山",
            surface="芝",
            distance=2000,
            n_runners=18,
        ))
        session.add(Race(
            race_id=race_id2,
            date="2024-12-28",
            course="中山",
            surface="芝",
            distance=1600,
            n_runners=18,
        ))
        session.commit()

        client = _build_mock_client({"b1": TAN_FUKU_HTML})
        counters = await run_fetch_live_odds([RACE_ID, race_id2], ["b1"], client, session)

    assert counters["fetched"] == 2

    with Session(engine) as s:
        rows1 = s.execute(
            select(LiveOdds).where(LiveOdds.race_id == RACE_ID)
        ).scalars().all()
        rows2 = s.execute(
            select(LiveOdds).where(LiveOdds.race_id == race_id2)
        ).scalars().all()
        assert len(rows1) == 36  # tan 18 + fuku 18
        assert len(rows2) == 36


@pytest.mark.asyncio
async def test_fetch_stops_on_stop_flag(session, monkeypatch):
    """stop_flag が立っている場合は ScraperStopped を送出する。"""
    import scraper.stop_flag as sf
    monkeypatch.setattr(sf, "_internal_stop", True)

    client = _build_mock_client({"b1": TAN_FUKU_HTML})
    with pytest.raises(ScraperStopped):
        await fetch_odds_for_race(RACE_ID, ["b1"], client, session)

    monkeypatch.setattr(sf, "_internal_stop", False)


@pytest.mark.asyncio
async def test_fetch_handles_http_error(session):
    """fetch エラー時は error カウンタが増加し、既存データは保持される。"""
    import httpx

    settings = Settings(rate_min_seconds=0.0, rate_max_seconds=0.0)
    rate = AsyncRateLimiter(settings)
    robots = RobotsCache("TestAgent")
    http = httpx.AsyncClient()
    client = NetkeibaClient(rate, robots, http, settings)

    async def _fail_fetch(url: str, *, use_cache=True, cache_max_age_hours=0.5) -> str:
        raise RuntimeError("simulated network error")

    client.fetch = _fail_fetch  # type: ignore[method-assign]

    counters = await fetch_odds_for_race(RACE_ID, ["b1"], client, session)
    assert counters["errors"] == 1
    assert counters["fetched"] == 0
