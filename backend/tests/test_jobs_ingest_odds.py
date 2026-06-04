"""Tests for jobs.ingest_odds resume / no-odds handling (no network)."""

from __future__ import annotations

import json
from urllib.parse import parse_qs, urlparse

import pytest

from db.odds_db import (
    fetched_bet_types,
    init_odds_db,
    load_race_odds,
    make_odds_engine,
    odds_session_scope,
    upsert_race_odds,
)
from jobs.ingest_odds import _ingest_race
from scraper.parsers.odds import parse_odds_payload

RID = "201906010101"


def _full_feed() -> dict[int, dict]:
    """Canned JSON keyed by netkeiba type for a 3-horse race."""
    def wrap(odds: dict) -> dict:
        return {"status": "result", "data": {"official_datetime": "2019-x", "odds": odds}}

    return {
        1: wrap({"1": {"01": ["2.0", "0.0", "1"]}, "2": {"01": ["1.1", "1.3", "1"]}}),
        3: wrap({"3": {"0102": ["3.0", "0.0", "1"]}}),
        4: wrap({"4": {"0102": ["5.0", "0.0", "1"]}}),
        5: wrap({"5": {"0102": ["2.0", "3.0", "1"]}}),
        6: wrap({"6": {"0102": ["8.0", "0.0", "1"]}}),
        7: wrap({"7": {"010203": ["20.0", "0.0", "1"]}}),
        8: wrap({"8": {"010203": ["50.0", "0.0", "1"]}}),
    }


class FakeClient:
    def __init__(self, feed: dict[int, dict]) -> None:
        self._feed = feed
        self.calls: list[int] = []

    async def fetch(self, url: str, **_kw) -> str:
        type_code = int(parse_qs(urlparse(url).query)["type"][0])
        self.calls.append(type_code)
        return json.dumps(self._feed[type_code])


@pytest.fixture
def engine(tmp_path):
    eng = make_odds_engine(tmp_path / "odds.db")
    init_odds_db(eng)
    return eng


async def test_full_ingest_stores_all_bet_types(engine) -> None:
    client = FakeClient(_full_feed())
    status = await _ingest_race(client, engine, RID)
    assert status == "done"
    with odds_session_scope(engine) as s:
        loaded = load_race_odds(s, RID)
    assert set(loaded) == {"単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "三連複", "三連単"}
    assert loaded["三連単"] == {"1→2→3": [50.0, 0.0, 1]}
    assert client.calls == [1, 3, 4, 5, 6, 7, 8]


async def test_second_run_skips_without_fetching(engine) -> None:
    await _ingest_race(FakeClient(_full_feed()), engine, RID)
    client2 = FakeClient(_full_feed())
    status = await _ingest_race(client2, engine, RID)
    assert status == "resumed_skip"
    assert client2.calls == []  # nothing re-fetched


async def test_partial_resume_only_fetches_missing(engine) -> None:
    # Simulate an interrupted run where only 単複 (type=1) got stored.
    _, parsed = parse_odds_payload(_full_feed()[1])
    with odds_session_scope(engine) as s:
        for bt, combos in parsed.items():
            upsert_race_odds(s, RID, bt, "dt", combos)

    client = FakeClient(_full_feed())
    status = await _ingest_race(client, engine, RID)
    assert status == "done"
    # type=1 already had 単複 → not re-fetched; the rest are.
    assert client.calls == [3, 4, 5, 6, 7, 8]


async def test_no_odds_writes_sentinel_and_skips_on_rerun(engine) -> None:
    empty = {"status": "result", "data": {"official_datetime": None, "odds": {}}}
    client = FakeClient({t: empty for t in (1, 3, 4, 5, 6, 7, 8)})
    status = await _ingest_race(client, engine, RID)
    assert status == "no_odds"
    assert client.calls == [1]  # stops after type=1 comes back empty
    with odds_session_scope(engine) as s:
        assert "__none__" in fetched_bet_types(s, RID)

    client2 = FakeClient({t: empty for t in (1, 3, 4, 5, 6, 7, 8)})
    assert await _ingest_race(client2, engine, RID) == "resumed_skip"
    assert client2.calls == []
