"""Tests for db.odds_db: schema, (de)compression, upsert/resume/load."""

from __future__ import annotations

import pytest

from db.odds_db import (
    compress_odds,
    decompress_odds,
    fetched_bet_types,
    init_odds_db,
    load_race_odds,
    make_odds_engine,
    odds_session_scope,
    upsert_race_odds,
)


@pytest.fixture
def engine(tmp_path):
    eng = make_odds_engine(tmp_path / "odds.db")
    init_odds_db(eng)
    return eng


def test_compress_roundtrip() -> None:
    combos = {"1-2": [22.7, 0.0, 8], "1-3": [73.5, 0.0, 21]}
    assert decompress_odds(compress_odds(combos)) == combos


def test_upsert_and_load(engine) -> None:
    with odds_session_scope(engine) as s:
        upsert_race_odds(s, "RID1", "馬連", "2025-12-14 11:23:52", {"1-2": [22.7, 0.0, 8]})
        upsert_race_odds(s, "RID1", "ワイド", "2025-12-14 11:23:52", {"1-2": [7.2, 8.6, 8]})

    with odds_session_scope(engine) as s:
        loaded = load_race_odds(s, "RID1")
    assert loaded == {"馬連": {"1-2": [22.7, 0.0, 8]}, "ワイド": {"1-2": [7.2, 8.6, 8]}}


def test_upsert_replaces_existing(engine) -> None:
    with odds_session_scope(engine) as s:
        upsert_race_odds(s, "RID1", "馬連", None, {"1-2": [10.0, 0.0, 1]})
    with odds_session_scope(engine) as s:
        upsert_race_odds(s, "RID1", "馬連", "dt", {"1-2": [22.7, 0.0, 8], "1-3": [5.0, 0.0, 2]})
    with odds_session_scope(engine) as s:
        loaded = load_race_odds(s, "RID1")
    assert loaded["馬連"] == {"1-2": [22.7, 0.0, 8], "1-3": [5.0, 0.0, 2]}


def test_fetched_bet_types_for_resume(engine) -> None:
    with odds_session_scope(engine) as s:
        upsert_race_odds(s, "RID1", "馬連", None, {"1-2": [1.0, 0.0, 1]})
        upsert_race_odds(s, "RID1", "三連単", None, {"1→2→3": [1.0, 0.0, 1]})
        upsert_race_odds(s, "RID2", "馬連", None, {"1-2": [1.0, 0.0, 1]})
    with odds_session_scope(engine) as s:
        assert fetched_bet_types(s, "RID1") == {"馬連", "三連単"}
        assert fetched_bet_types(s, "RID2") == {"馬連"}
        assert fetched_bet_types(s, "RID_MISSING") == set()


def test_load_missing_race_empty(engine) -> None:
    with odds_session_scope(engine) as s:
        assert load_race_odds(s, "NOPE") == {}
