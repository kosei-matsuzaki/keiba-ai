"""Tests for the shutuba ingest job.

冪等性テスト:
  - finish_position が確定した entry の odds_win は上書きしない
  - finish_position が NULL の entry の odds_win / popularity は最新値で上書きする
  - stop_flag が立っている場合は ScraperStopped を送出する
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from keiba_ai.core.config import Settings
from keiba_ai.db.models.entry import Entry
from keiba_ai.db.models.horse import Horse
from keiba_ai.db.models.jockey import Jockey
from keiba_ai.db.models.race import Race
from keiba_ai.db.models.scrape_log import ScrapeLog
from keiba_ai.db.models.trainer import Trainer
from keiba_ai.jobs.ingest_shutuba import run_ingest_shutuba
from keiba_ai.scraper.netkeiba import NetkeibaClient
from keiba_ai.scraper.rate_limiter import AsyncRateLimiter
from keiba_ai.scraper.robots import RobotsCache

FIXTURES = Path(__file__).parent / "fixtures"

CARD_CALENDAR_HTML = (FIXTURES / "race_card_calendar_20260505.html").read_text(encoding="utf-8")
SHUTUBA_HTML = (FIXTURES / "shutuba_202406010111.html").read_text(encoding="utf-8")

DATE = "2026-05-05"


def _build_fake_fetch(card_calendar_html: str, shutuba_html: str):
    async def fake_fetch(
        url: str,
        *,
        use_cache: bool = True,
        cache_max_age_hours: float = 24 * 30,
    ) -> str:
        if "race_list" in url:
            return card_calendar_html
        return shutuba_html
    return fake_fetch


@pytest.fixture()
def mock_client(monkeypatch) -> NetkeibaClient:
    import httpx
    settings = Settings(rate_min_seconds=0.0, rate_max_seconds=0.0)
    rate = AsyncRateLimiter(settings)
    robots = RobotsCache("TestAgent")
    http = httpx.AsyncClient()
    client = NetkeibaClient(rate, robots, http, settings)
    client.fetch = _build_fake_fetch(CARD_CALENDAR_HTML, SHUTUBA_HTML)  # type: ignore[method-assign]
    return client


@pytest.mark.asyncio
async def test_ingest_shutuba_inserts_races(db_session, mock_client, tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    counters = await run_ingest_shutuba(DATE, mock_client, db_session, limit=2)

    assert counters["fetched"] == 2
    assert counters["errors"] == 0

    rows = db_session.execute(select(Race)).scalars().all()
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_ingest_shutuba_inserts_entries(db_session, mock_client, tmp_path, monkeypatch):
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    await run_ingest_shutuba(DATE, mock_client, db_session, limit=1)

    entries = db_session.execute(select(Entry)).scalars().all()
    # フィクスチャは 16 頭立て
    assert len(entries) == 16


@pytest.mark.asyncio
async def test_ingest_shutuba_entries_have_null_finish_position(
    db_session, mock_client, tmp_path, monkeypatch
):
    """出馬表 ingest 後の entry は finish_position が NULL であること。"""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    await run_ingest_shutuba(DATE, mock_client, db_session, limit=1)

    entries = db_session.execute(select(Entry)).scalars().all()
    for e in entries:
        assert e.finish_position is None


@pytest.mark.asyncio
async def test_ingest_shutuba_records_scrape_log(
    db_session, mock_client, tmp_path, monkeypatch
):
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    await run_ingest_shutuba(DATE, mock_client, db_session, limit=1)

    logs = db_session.execute(
        select(ScrapeLog).where(ScrapeLog.status == "ok")
    ).scalars().all()
    assert len(logs) >= 1


@pytest.mark.asyncio
async def test_ingest_shutuba_date_stored_in_races(
    db_session, mock_client, tmp_path, monkeypatch
):
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    await run_ingest_shutuba(DATE, mock_client, db_session, limit=1)

    row = db_session.execute(select(Race)).scalars().first()
    assert row.date == DATE


@pytest.mark.asyncio
async def test_ingest_shutuba_saves_horse_names(
    db_session, mock_client, tmp_path, monkeypatch
):
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    await run_ingest_shutuba(DATE, mock_client, db_session, limit=1)

    horses = db_session.execute(select(Horse)).scalars().all()
    names = {h.name for h in horses}
    assert "ドウデュース" in names
    assert "タスティエーラ" in names


@pytest.mark.asyncio
async def test_ingest_shutuba_saves_jockey_names(
    db_session, mock_client, tmp_path, monkeypatch
):
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    await run_ingest_shutuba(DATE, mock_client, db_session, limit=1)

    jockeys = db_session.execute(select(Jockey)).scalars().all()
    names = {j.name for j in jockeys}
    assert "武豊" in names


@pytest.mark.asyncio
async def test_ingest_shutuba_saves_trainer_names(
    db_session, mock_client, tmp_path, monkeypatch
):
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    await run_ingest_shutuba(DATE, mock_client, db_session, limit=1)

    trainers = db_session.execute(select(Trainer)).scalars().all()
    names = {t.name for t in trainers}
    assert "友道康夫" in names


# ── 冪等性テスト ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_idempotency_does_not_overwrite_confirmed_finish_position(
    db_session, tmp_path, monkeypatch
):
    """finish_position が確定した entry は shutuba ingest で上書きされないこと。

    シナリオ:
      1. 結果 ingest 済み entry (finish_position=1, odds_win=3.1) が DB に存在する
      2. shutuba ingest で odds_win=99.9 (変動後) が来ても odds_win は変わらない
    """
    import httpx
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))

    RACE_ID = "202406010111"
    HORSE_ID = "2019105293"

    # 1. 事前: race / horse / entry (finish_position 確定) を直接 INSERT
    db_session.add(Race(
        race_id=RACE_ID,
        date="2024-06-01",
        course="札幌",
        surface="芝",
        distance=2000,
    ))
    db_session.add(Horse(horse_id=HORSE_ID, name="ドウデュース"))
    db_session.add(Jockey(jockey_id="01167", name="武豊"))
    db_session.add(Trainer(trainer_id="01096", name="友道康夫"))
    db_session.add(Entry(
        race_id=RACE_ID,
        horse_id=HORSE_ID,
        post_position=1,
        jockey_id="01167",
        trainer_id="01096",
        finish_position=1,  # 確定済み
        odds_win=3.1,
        popularity=1,
    ))
    db_session.commit()

    # 2. shutuba ingest 実行（同レースの odds が 99.9 に変動した想定 HTML）
    settings = Settings(rate_min_seconds=0.0, rate_max_seconds=0.0)
    rate = AsyncRateLimiter(settings)
    robots = RobotsCache("TestAgent")
    http = httpx.AsyncClient()
    client = NetkeibaClient(rate, robots, http, settings)
    client.fetch = _build_fake_fetch(CARD_CALENDAR_HTML, SHUTUBA_HTML)  # type: ignore[method-assign]

    await run_ingest_shutuba("2026-05-05", client, db_session, limit=1)

    # 3. finish_position=1 の entry の odds_win は変わっていないこと
    entry = db_session.execute(
        select(Entry).where(Entry.race_id == RACE_ID, Entry.horse_id == HORSE_ID)
    ).scalar_one()
    assert entry.finish_position == 1
    assert entry.odds_win == pytest.approx(3.1), (
        f"finish_position 確定済み entry の odds_win が上書きされた: {entry.odds_win}"
    )


@pytest.mark.asyncio
async def test_idempotency_updates_odds_for_pending_entry(
    db_session, tmp_path, monkeypatch
):
    """finish_position が NULL の entry は shutuba ingest で odds_win が更新されること。

    シナリオ:
      1. 出馬表から初回 ingest (odds_win=99.9)
      2. 再度 shutuba ingest — フィクスチャの odds_win=3.1 で上書きされる
    """
    import httpx
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))

    RACE_ID = "202406010111"
    HORSE_ID = "2019105293"

    # 1. 事前: race / horse / entry (finish_position=None, 古い odds) を INSERT
    db_session.add(Race(
        race_id=RACE_ID,
        date="2024-06-01",
        course="札幌",
        surface="芝",
        distance=2000,
    ))
    db_session.add(Horse(horse_id=HORSE_ID, name="ドウデュース"))
    db_session.add(Jockey(jockey_id="01167", name="武豊"))
    db_session.add(Trainer(trainer_id="01096", name="友道康夫"))
    db_session.add(Entry(
        race_id=RACE_ID,
        horse_id=HORSE_ID,
        post_position=1,
        jockey_id="01167",
        trainer_id="01096",
        finish_position=None,  # 未確定
        odds_win=99.9,         # 古いオッズ
        popularity=5,
    ))
    db_session.commit()

    # 2. shutuba ingest 実行 (フィクスチャ: odds_win=3.1, popularity=1)
    settings = Settings(rate_min_seconds=0.0, rate_max_seconds=0.0)
    rate = AsyncRateLimiter(settings)
    robots = RobotsCache("TestAgent")
    http = httpx.AsyncClient()
    client = NetkeibaClient(rate, robots, http, settings)
    client.fetch = _build_fake_fetch(CARD_CALENDAR_HTML, SHUTUBA_HTML)  # type: ignore[method-assign]

    await run_ingest_shutuba("2026-05-05", client, db_session, limit=1)

    # 3. odds_win と popularity が最新値に更新されていること
    entry = db_session.execute(
        select(Entry).where(Entry.race_id == RACE_ID, Entry.horse_id == HORSE_ID)
    ).scalar_one()
    assert entry.finish_position is None
    assert entry.odds_win == pytest.approx(3.1), (
        f"finish_position=None の entry の odds_win が更新されなかった: {entry.odds_win}"
    )
    assert entry.popularity == 1


@pytest.mark.asyncio
async def test_ingest_shutuba_stops_on_stop_flag(
    db_session, mock_client, tmp_path, monkeypatch
):
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("KEIBA_SCRAPER_STOP", "1")

    from keiba_ai.scraper.stop_flag import ScraperStopped
    with pytest.raises(ScraperStopped):
        await run_ingest_shutuba(DATE, mock_client, db_session)


@pytest.mark.asyncio
async def test_ingest_shutuba_limit(db_session, mock_client, tmp_path, monkeypatch):
    """--limit で取得レース数を制限できること。"""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    counters = await run_ingest_shutuba(DATE, mock_client, db_session, limit=1)

    assert counters["fetched"] == 1
    rows = db_session.execute(select(Race)).scalars().all()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_confirmed_entry_post_position_not_overwritten(
    db_session, tmp_path, monkeypatch
):
    """finish_position 確定済みの entry は post_position を含む全 shutuba カラムが
    上書きされないこと（High 指摘 #2 の回帰テスト）。

    シナリオ:
      1. finish_position=1, post_position=3 の確定済み entry が DB に存在する
      2. shutuba ingest で post_position=1 (フィクスチャ値) が来ても
         post_position は 3 のまま変わらない
    """
    import httpx
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))

    RACE_ID = "202406010111"
    HORSE_ID = "2019105293"
    ORIGINAL_POST_POS = 3
    ORIGINAL_JOCKEY_ID = "99999"  # フィクスチャと異なる騎手

    db_session.add(Race(
        race_id=RACE_ID,
        date="2024-06-01",
        course="札幌",
        surface="芝",
        distance=2000,
    ))
    db_session.add(Horse(horse_id=HORSE_ID, name="ドウデュース"))
    db_session.add(Jockey(jockey_id=ORIGINAL_JOCKEY_ID, name="仮騎手"))
    db_session.add(Trainer(trainer_id="01096", name="友道康夫"))
    db_session.add(Entry(
        race_id=RACE_ID,
        horse_id=HORSE_ID,
        post_position=ORIGINAL_POST_POS,
        jockey_id=ORIGINAL_JOCKEY_ID,
        trainer_id="01096",
        finish_position=1,  # 確定済み
        odds_win=3.1,
        popularity=1,
    ))
    db_session.commit()

    settings = Settings(rate_min_seconds=0.0, rate_max_seconds=0.0)
    rate = AsyncRateLimiter(settings)
    robots = RobotsCache("TestAgent")
    http = httpx.AsyncClient()
    client = NetkeibaClient(rate, robots, http, settings)
    client.fetch = _build_fake_fetch(CARD_CALENDAR_HTML, SHUTUBA_HTML)  # type: ignore[method-assign]

    await run_ingest_shutuba("2026-05-05", client, db_session, limit=1)

    entry = db_session.execute(
        select(Entry).where(Entry.race_id == RACE_ID, Entry.horse_id == HORSE_ID)
    ).scalar_one()
    assert entry.finish_position == 1, "finish_position が消えた"
    assert entry.post_position == ORIGINAL_POST_POS, (
        f"finish_position 確定済み entry の post_position が上書きされた: {entry.post_position}"
    )
    assert entry.jockey_id == ORIGINAL_JOCKEY_ID, (
        f"finish_position 確定済み entry の jockey_id が上書きされた: {entry.jockey_id}"
    )
    assert entry.odds_win == pytest.approx(3.1), (
        f"finish_position 確定済み entry の odds_win が上書きされた: {entry.odds_win}"
    )


@pytest.mark.asyncio
async def test_idempotency_same_date_multiple_runs(
    db_session, mock_client, tmp_path, monkeypatch
):
    """同一日付の shutuba ingest を複数回実行しても冪等であること（Suggestion #4）。

    シナリオ:
      1. shutuba ingest を 1 回目実行 → races/entries が INSERT される
      2. 同じ日付で 2 回目実行 → race/entry 数が変わらない、finish_position は NULL のまま
    """
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))

    await run_ingest_shutuba(DATE, mock_client, db_session, limit=1)
    races_after_first = db_session.execute(select(Race)).scalars().all()
    entries_after_first = db_session.execute(select(Entry)).scalars().all()

    await run_ingest_shutuba(DATE, mock_client, db_session, limit=1)
    races_after_second = db_session.execute(select(Race)).scalars().all()
    entries_after_second = db_session.execute(select(Entry)).scalars().all()

    assert len(races_after_second) == len(races_after_first), (
        "2 回目 ingest で race 行数が変わった"
    )
    assert len(entries_after_second) == len(entries_after_first), (
        "2 回目 ingest で entry 行数が変わった"
    )
    for e in entries_after_second:
        assert e.finish_position is None, (
            f"冪等実行後に finish_position が NULL でなくなった: {e}"
        )
