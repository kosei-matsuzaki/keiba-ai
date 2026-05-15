"""Integration test for the ingest job.

Uses monkeypatching to avoid any real HTTP requests.  The NetkeibaClient.fetch
method is replaced with a function that returns fixture HTML.

DB assertions use ORM (sqlalchemy.orm.Session) instead of sqlite3.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select

from core.config import Settings
from db.models.entry import Entry
from db.models.horse import Horse
from db.models.jockey import Jockey
from db.models.payout import Payout
from db.models.race import Race
from db.models.scrape_log import ScrapeLog
from db.models.trainer import Trainer
from jobs.ingest import run_ingest
from scraper.netkeiba import NetkeibaClient
from scraper.rate_limiter import AsyncRateLimiter
from scraper.robots import RobotsCache

FIXTURES = Path(__file__).parent / "fixtures"
CALENDAR_HTML = (FIXTURES / "race_calendar_20241228.html").read_text(encoding="utf-8")
RESULT_HTML = (FIXTURES / "race_result_202406010101.html").read_text(encoding="utf-8")

DATE = "2024-12-28"


def _build_fake_fetch(calendar_html: str, result_html: str):
    """Basic fake — calendar + race result only; /horse/ raises so detail fetch
    is exercised through its existing graceful-failure path (logged warning,
    horse_kwargs stays at race_result-derived name).
    """
    async def fake_fetch(url: str, *, use_cache: bool = True, cache_max_age_hours: float = 24 * 30) -> str:
        if "/race/list/" in url:
            return calendar_html
        if "/horse/" in url:
            raise RuntimeError(f"basic mock has no horse fixture for {url}")
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

    from scraper.stop_flag import ScraperStopped
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


@pytest.mark.asyncio
async def test_ensure_masters_coalesce_preserves_existing_name_when_new_is_none(db_session):
    """COALESCE(excluded.name, Horse.name) は新規 None で既存値を維持する。

    回帰防止: PR-A code-reviewer が「既存 'X' + 新規 None で NULL になる」と
    誤認したため、SQL COALESCE の挙動を直接ロックする。
    """
    from jobs.ingest import _ensure_masters
    from scraper.parsers.race_result import ParsedEntry, ParsedRaceResult

    # 1) 既存馬を name 入りで insert
    db_session.add(Horse(horse_id="HORSE_X", name="ExistingName"))
    db_session.commit()

    # 2) horse_name=None の Entry を持つ ParsedRaceResult で再 ingest（client=None → detail fetch なし）
    parsed = ParsedRaceResult(
        race_id="R1",
        date="2024-01-01",
        course="中山",
        surface="芝",
        distance=1600,
        entries=[ParsedEntry(race_id="R1", horse_id="HORSE_X", horse_name=None)],
    )
    await _ensure_masters(db_session, parsed)
    db_session.commit()

    # 既存 name は維持される
    horse = db_session.get(Horse, "HORSE_X")
    assert horse.name == "ExistingName"


@pytest.mark.asyncio
async def test_ensure_masters_coalesce_fills_missing_name(db_session):
    """既存 NULL + 新規 'X' で name が補完される。"""
    from jobs.ingest import _ensure_masters
    from scraper.parsers.race_result import ParsedEntry, ParsedRaceResult

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
    await _ensure_masters(db_session, parsed)
    db_session.commit()

    horse = db_session.get(Horse, "HORSE_Y")
    assert horse.name == "NewName"


@pytest.mark.asyncio
async def test_ensure_masters_coalesce_overwrites_when_both_have_value(db_session):
    """既存 'X' + 新規 'Y' は上書き（COALESCE 第一引数優先のため）。

    現仕様: netkeiba 側の表記揺れがあった場合、最新を採用する。
    将来「name は一度入ったら不変」に変えたい場合は引数順を反転する。
    """
    from jobs.ingest import _ensure_masters
    from scraper.parsers.race_result import ParsedEntry, ParsedRaceResult

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
    await _ensure_masters(db_session, parsed)
    db_session.commit()

    horse = db_session.get(Horse, "HORSE_Z")
    assert horse.name == "NewName"


# ── PR-B: horse_detail / horse_pedigree fetch tests ──────────────────────────

DETAIL_HTML = (FIXTURES / "horse_detail_2022104732.html").read_text(encoding="utf-8")
PEDIGREE_HTML = (FIXTURES / "horse_pedigree_2022104732.html").read_text(encoding="utf-8")


def _build_fake_fetch_with_detail(
    calendar_html: str,
    result_html: str,
    detail_html: str,
    pedigree_html: str,
):
    """Fake fetch that routes horse/ped URLs to fixture HTML."""
    async def fake_fetch(url: str, *, use_cache: bool = True, cache_max_age_hours: float = 24 * 30) -> str:
        if "/race/list/" in url:
            return calendar_html
        if "/horse/ped/" in url:
            return pedigree_html
        if "/horse/" in url and "/race/" not in url:
            return detail_html
        return result_html
    return fake_fetch


@pytest.fixture()
def mock_client_with_detail() -> NetkeibaClient:
    import httpx
    settings = Settings(rate_min_seconds=0.0, rate_max_seconds=0.0)
    rate = AsyncRateLimiter(settings)
    robots = RobotsCache("TestAgent")
    http = httpx.AsyncClient()
    client = NetkeibaClient(rate, robots, http, settings)
    client.fetch = _build_fake_fetch_with_detail(  # type: ignore[method-assign]
        CALENDAR_HTML, RESULT_HTML, DETAIL_HTML, PEDIGREE_HTML
    )
    return client


@pytest.mark.asyncio
async def test_ingest_skips_horse_detail_when_existing_name_present(db_session):
    """既存 horses.name が埋まっている馬は detail/ped fetch をしない。

    client.fetch の呼び出しを記録し、/horse/<id>/ と /horse/ped/<id>/ への
    リクエストが発生していないことを確認する。
    """
    import httpx

    from jobs.ingest import _ensure_masters
    from scraper.parsers.race_result import ParsedEntry, ParsedRaceResult

    settings = Settings(rate_min_seconds=0.0, rate_max_seconds=0.0)
    rate = AsyncRateLimiter(settings)
    robots = RobotsCache("TestAgent")
    http = httpx.AsyncClient()
    client = NetkeibaClient(rate, robots, http, settings)

    fetched_urls: list[str] = []

    async def recording_fetch(url: str, *, use_cache: bool = True, cache_max_age_hours: float = 24 * 30) -> str:
        fetched_urls.append(url)
        return RESULT_HTML

    client.fetch = recording_fetch  # type: ignore[method-assign]

    # 既存馬（name 埋まり）を事前 insert
    db_session.add(Horse(horse_id="2019105293", name="ドウデュース"))
    db_session.commit()

    parsed = ParsedRaceResult(
        race_id="R1",
        date="2024-01-01",
        course="東京",
        surface="芝",
        distance=2400,
        entries=[ParsedEntry(race_id="R1", horse_id="2019105293", horse_name="ドウデュース")],
    )
    await _ensure_masters(db_session, parsed, client=client)
    db_session.commit()

    # detail / ped fetch は一切発生しないはず
    assert all("/horse/" not in url for url in fetched_urls), (
        f"Unexpected detail/ped fetch for horse with existing name: {fetched_urls}"
    )


@pytest.mark.asyncio
async def test_ingest_fetches_horse_detail_and_pedigree_for_new_horse(
    db_session, mock_client_with_detail, tmp_path, monkeypatch
):
    """新規馬（DBにレコードなし）は詳細ページ + 血統ページをフェッチして sire/dam が埋まる。

    race_result fixture 内の馬（ドウデュース等）を対象にフル ingest して確認する。
    """
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    await run_ingest(DATE, mock_client_with_detail, db_session, limit=1)

    horses = db_session.execute(select(Horse)).scalars().all()
    # 少なくとも1頭はフェッチされ、sire と dam が埋まっている
    horses_with_sire = [h for h in horses if h.sire is not None]
    assert len(horses_with_sire) >= 1, "At least one horse should have sire populated"
    horse = horses_with_sire[0]
    assert horse.sire == "ロードカナロア"
    assert horse.dam == "スターハイネス"


@pytest.mark.asyncio
async def test_ingest_continues_on_horse_detail_fetch_failure(
    db_session, tmp_path, monkeypatch
):
    """horse_detail fetch が例外を出しても全 race ingest は失敗しない。"""
    import httpx

    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))

    settings = Settings(rate_min_seconds=0.0, rate_max_seconds=0.0)
    rate = AsyncRateLimiter(settings)
    robots = RobotsCache("TestAgent")
    http = httpx.AsyncClient()
    client = NetkeibaClient(rate, robots, http, settings)

    async def failing_detail_fetch(url: str, *, use_cache: bool = True, cache_max_age_hours: float = 24 * 30) -> str:
        if "/race/list/" in url:
            return CALENDAR_HTML
        if "/horse/" in url and "/race/" not in url:
            raise RuntimeError("simulated network failure")
        return RESULT_HTML

    client.fetch = failing_detail_fetch  # type: ignore[method-assign]

    counters = await run_ingest(DATE, client, db_session, limit=1)

    # detail fetch が失敗しても race ingest 自体は成功する
    assert counters["fetched"] == 1
    assert counters["errors"] == 0

    # 馬は name のみで登録される（detail fetch 失敗のため sire/dam は None）
    horses = db_session.execute(select(Horse)).scalars().all()
    assert len(horses) >= 1


# ── payouts ingest テスト ─────────────────────────────────────────────────────

ALL_PAYOUT_HTML = (FIXTURES / "race_result_all_payout_types.html").read_text(encoding="utf-8")
ALL_PAYOUT_DATE = "2024-06-01"


def _build_fake_fetch_all_payouts(calendar_html: str, result_html: str):
    """全 bet_type フィクスチャを返す fake fetch。"""
    async def fake_fetch(url: str, *, use_cache: bool = True, cache_max_age_hours: float = 24 * 30) -> str:
        if "/race/list/" in url:
            return calendar_html
        if "/horse/" in url:
            raise RuntimeError(f"no horse fixture for {url}")
        return result_html
    return fake_fetch


@pytest.fixture()
def mock_client_all_payouts(monkeypatch) -> NetkeibaClient:
    import httpx
    settings = Settings(rate_min_seconds=0.0, rate_max_seconds=0.0)
    rate = AsyncRateLimiter(settings)
    robots = RobotsCache("TestAgent")
    http = httpx.AsyncClient()
    client = NetkeibaClient(rate, robots, http, settings)
    client.fetch = _build_fake_fetch_all_payouts(CALENDAR_HTML, ALL_PAYOUT_HTML)  # type: ignore[method-assign]
    return client


@pytest.mark.asyncio
async def test_ingest_inserts_payouts(db_session, mock_client_all_payouts, tmp_path, monkeypatch):
    """ingest 後に payouts テーブルに全 8 bet_type の行が挿入される。"""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    counters = await run_ingest(ALL_PAYOUT_DATE, mock_client_all_payouts, db_session, limit=1)
    assert counters["fetched"] == 1

    payouts = db_session.execute(select(Payout)).scalars().all()
    bet_types = {p.bet_type for p in payouts}
    expected = {"単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "三連複", "三連単"}
    assert expected == bet_types


@pytest.mark.asyncio
async def test_ingest_payouts_amounts(db_session, mock_client_all_payouts, tmp_path, monkeypatch):
    """払戻金が正しく DB に保存される。"""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    await run_ingest(ALL_PAYOUT_DATE, mock_client_all_payouts, db_session, limit=1)

    payouts = db_session.execute(select(Payout)).scalars().all()
    by_type: dict[str, list[Payout]] = {}
    for p in payouts:
        by_type.setdefault(p.bet_type, []).append(p)

    tan = by_type["単勝"]
    assert len(tan) == 1
    assert tan[0].combo == "3"
    assert tan[0].amount == 520
    assert tan[0].popularity == 1

    srt = by_type["三連単"]
    assert len(srt) == 1
    assert srt[0].combo == "3→5→9"
    assert srt[0].amount == 52300


@pytest.mark.asyncio
async def test_ingest_payouts_wide_three_rows(db_session, mock_client_all_payouts, tmp_path, monkeypatch):
    """ワイドは 3 コンボそれぞれが独立した payouts 行になる。"""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    await run_ingest(ALL_PAYOUT_DATE, mock_client_all_payouts, db_session, limit=1)

    payouts = db_session.execute(select(Payout)).scalars().all()
    wide = [p for p in payouts if p.bet_type == "ワイド"]
    assert len(wide) == 3
    combos = {p.combo for p in wide}
    assert combos == {"3-5", "3-9", "5-9"}


@pytest.mark.asyncio
async def test_ingest_payouts_idempotent_on_reingest(
    db_session, mock_client_all_payouts, tmp_path, monkeypatch
):
    """同一レースを再 ingest しても payouts 行が重複しない（DELETE → INSERT の冪等性）。"""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))

    # 初回 ingest
    await run_ingest(ALL_PAYOUT_DATE, mock_client_all_payouts, db_session, limit=1)
    count_first = db_session.execute(select(Payout)).scalars().all()

    # 再 ingest のために scrape_log を消去してスキップを回避
    from db.models.scrape_log import ScrapeLog
    db_session.execute(ScrapeLog.__table__.delete())
    db_session.commit()

    await run_ingest(ALL_PAYOUT_DATE, mock_client_all_payouts, db_session, limit=1)
    count_second = db_session.execute(select(Payout)).scalars().all()

    assert len(count_first) == len(count_second), (
        f"payouts duplicated on re-ingest: {len(count_first)} -> {len(count_second)}"
    )
