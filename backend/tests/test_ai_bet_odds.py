"""Tests for ai/bet_odds.py — baseline odds computation and compute_race_odds."""

from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import keiba_ai.db.models  # noqa: F401
from keiba_ai.ai.bet_odds import (
    _FALLBACK_AMOUNTS,
    compute_baseline_odds,
    compute_baseline_odds_by_class,
    compute_past_race_odds,
    compute_race_odds,
)
from keiba_ai.db.base import Base
from keiba_ai.db.models.entry import Entry
from keiba_ai.db.models.horse import Horse
from keiba_ai.db.models.live_odds import LiveOdds
from keiba_ai.db.models.payout import Payout
from keiba_ai.db.models.race import Race


@pytest.fixture()
def empty_session():
    """In-memory session with no payouts data."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture()
def seeded_session():
    """In-memory session with a few payout rows for testing."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Race(
            race_id="R001",
            date="2025-01-01",
            course="東京",
            surface="芝",
            distance=2000,
            race_class="G1",
            n_runners=10,
        ))
        session.add(Race(
            race_id="R002",
            date="2025-02-01",
            course="中山",
            surface="ダ",
            distance=1600,
            race_class="条件戦",
            n_runners=12,
        ))
        session.flush()

        # 単勝: two races, amounts 1200 and 800 → avg = 1000 → odds = 10.0
        session.add(Payout(race_id="R001", bet_type="単勝", combo="3", amount=1200, popularity=1))
        session.add(Payout(race_id="R002", bet_type="単勝", combo="5", amount=800, popularity=2))

        # 馬連: one race, amount 5500 → odds = 55.0
        session.add(Payout(race_id="R001", bet_type="馬連", combo="3-7", amount=5500, popularity=3))

        session.commit()
        yield session


def test_compute_baseline_odds_empty_payouts_uses_fallback(empty_session):
    """When payouts table is empty, every bet_type should return its fallback value."""
    odds = compute_baseline_odds(empty_session)

    for bet_type, fallback_amount in _FALLBACK_AMOUNTS.items():
        expected_odds = fallback_amount / 100.0
        assert odds[bet_type] == pytest.approx(expected_odds), (
            f"{bet_type}: expected fallback {expected_odds}, got {odds[bet_type]}"
        )


def test_compute_baseline_odds_uses_db_average(seeded_session):
    """When payouts table has data, returns average from DB for those bet_types."""
    odds = compute_baseline_odds(seeded_session)

    # 単勝: (1200 + 800) / 2 / 100 = 10.0
    assert odds["単勝"] == pytest.approx(10.0, rel=1e-4)

    # 馬連: 5500 / 100 = 55.0
    assert odds["馬連"] == pytest.approx(55.0, rel=1e-4)


def test_compute_baseline_odds_missing_types_fall_back(seeded_session):
    """Bet types not present in DB still get the hardcoded fallback."""
    odds = compute_baseline_odds(seeded_session)

    # ワイド has no rows in seeded_session
    expected_wide = _FALLBACK_AMOUNTS["ワイド"] / 100.0
    assert odds["ワイド"] == pytest.approx(expected_wide)

    # 三連単 has no rows
    expected_sanrentan = _FALLBACK_AMOUNTS["三連単"] / 100.0
    assert odds["三連単"] == pytest.approx(expected_sanrentan)


def test_compute_baseline_odds_returns_all_bet_types(seeded_session):
    """Result dict should cover all standard bet types."""
    odds = compute_baseline_odds(seeded_session)
    for bt in ["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "三連複", "三連単"]:
        assert bt in odds, f"Missing bet_type: {bt}"


def test_compute_baseline_odds_by_class_with_match(seeded_session):
    """When enough rows match the filter, returns filtered average."""
    # Only 単勝 from G1 races (R001): amount=1200 → odds=12.0
    # min_samples=1 to ensure we use the filtered value
    odds = compute_baseline_odds_by_class(
        seeded_session,
        race_class="G1",
        min_samples=1,
    )
    assert odds["単勝"] == pytest.approx(12.0, rel=1e-4)


def test_compute_baseline_odds_by_class_falls_back_on_low_samples(seeded_session):
    """When filtered rows < min_samples, falls back to overall average."""
    # G1 has only 1 単勝 row, min_samples=5 forces fallback to overall (10.0)
    odds = compute_baseline_odds_by_class(
        seeded_session,
        race_class="G1",
        min_samples=5,
    )
    # Overall average: (1200 + 800) / 2 / 100 = 10.0
    assert odds["単勝"] == pytest.approx(10.0, rel=1e-4)


def test_compute_baseline_odds_by_class_surface_filter(seeded_session):
    """Filter by surface isolates races on that track type."""
    # ダ surface only has R002: 単勝 amount=800 → odds=8.0
    odds = compute_baseline_odds_by_class(
        seeded_session,
        surface="ダ",
        min_samples=1,
    )
    assert odds["単勝"] == pytest.approx(8.0, rel=1e-4)


def test_compute_baseline_odds_by_class_no_filter_matches_overall(seeded_session):
    """With no filter conditions, by_class returns same as compute_baseline_odds."""
    overall = compute_baseline_odds(seeded_session)
    by_class = compute_baseline_odds_by_class(seeded_session, min_samples=0)
    for bt in overall:
        assert overall[bt] == pytest.approx(by_class[bt], rel=1e-4), f"Mismatch for {bt}"


# ---------------------------------------------------------------------------
# compute_race_odds tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def live_odds_session():
    """In-memory session seeded with live_odds rows."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Race(
            race_id="R_LIVE",
            date="2025-01-01",
            course="東京",
            surface="芝",
            distance=2000,
            n_runners=10,
        ))
        session.flush()

        session.add(LiveOdds(
            race_id="R_LIVE",
            bet_type="馬連",
            combo="3-7",
            odds=25.4,
            odds_max=None,
            popularity=1,
            fetched_at="2025-01-01T10:00:00+00:00",
        ))
        session.add(LiveOdds(
            race_id="R_LIVE",
            bet_type="馬連",
            combo="3-9",
            odds=18.2,
            odds_max=None,
            popularity=2,
            fetched_at="2025-01-01T10:00:00+00:00",
        ))
        session.add(LiveOdds(
            race_id="R_LIVE",
            bet_type="単勝",
            combo="3",
            odds=5.0,
            odds_max=None,
            popularity=1,
            fetched_at="2025-01-01T10:00:00+00:00",
        ))
        # odds=None (未確定) の行は compute_race_odds で除外される
        session.add(LiveOdds(
            race_id="R_LIVE",
            bet_type="単勝",
            combo="5",
            odds=None,
            odds_max=None,
            popularity=None,
            fetched_at="2025-01-01T10:00:00+00:00",
        ))
        session.commit()
        yield session


def test_compute_race_odds_returns_nested_dict(live_odds_session):
    """compute_race_odds returns {bet_type: {combo: odds}} dict."""
    result = compute_race_odds(live_odds_session, "R_LIVE")
    assert isinstance(result, dict)
    assert "馬連" in result
    assert isinstance(result["馬連"], dict)


def test_compute_race_odds_values(live_odds_session):
    """Correct odds values are returned for each combo."""
    result = compute_race_odds(live_odds_session, "R_LIVE")
    assert result["馬連"]["3-7"] == pytest.approx(25.4)
    assert result["馬連"]["3-9"] == pytest.approx(18.2)
    assert result["単勝"]["3"] == pytest.approx(5.0)


def test_compute_race_odds_excludes_none_odds(live_odds_session):
    """Combos with odds=None are excluded from the result."""
    result = compute_race_odds(live_odds_session, "R_LIVE")
    # combo "5" has odds=None → should not appear
    assert "5" not in result.get("単勝", {})


def test_compute_race_odds_empty_for_unknown_race(live_odds_session):
    """Returns empty dict when race_id has no live_odds."""
    result = compute_race_odds(live_odds_session, "NONEXISTENT")
    assert result == {}


def test_compute_race_odds_empty_when_no_live_odds_table_populated(empty_session):
    """Returns empty dict when live_odds table is empty."""
    result = compute_race_odds(empty_session, "R001")
    assert result == {}


# ---------------------------------------------------------------------------
# compute_past_race_odds tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def past_race_session():
    """In-memory session seeded with a completed past race including entries and payouts."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        # payout_place JSON: {finish_position_str: amount_yen}
        payout_place = json.dumps({"1": 120, "2": 170, "3": 140})
        session.add(Race(
            race_id="PAST_R001",
            date="2024-01-01",
            course="東京",
            surface="芝",
            distance=2000,
            n_runners=5,
            payout_win=1500,
            payout_place=payout_place,
        ))
        session.flush()

        # Add horses
        for i in range(1, 6):
            session.add(Horse(horse_id=f"H{i}", name=f"horse{i}"))
        session.flush()

        # Entries: post_position 1-5, finish_position set for 1st, 2nd, 3rd
        # post 3 finished 1st, post 1 finished 2nd, post 5 finished 3rd
        horse_data = [
            # (post_position, odds_win, finish_position)
            (1, 5.0, 2),
            (2, 12.0, 4),
            (3, 3.0, 1),
            (4, 20.0, 5),
            (5, 8.0, 3),
        ]
        for pp, odds_win, finish_pos in horse_data:
            session.add(Entry(
                race_id="PAST_R001",
                horse_id=f"H{pp}",
                post_position=pp,
                odds_win=odds_win,
                finish_position=finish_pos,
            ))
        session.flush()

        # Payouts for winning combinations (連系)
        session.add(Payout(race_id="PAST_R001", bet_type="馬連", combo="1-3", amount=3500))
        session.add(Payout(race_id="PAST_R001", bet_type="ワイド", combo="1-3", amount=800))
        session.add(Payout(race_id="PAST_R001", bet_type="ワイド", combo="3-5", amount=600))
        session.add(Payout(race_id="PAST_R001", bet_type="馬単", combo="3→1", amount=6000))
        session.add(Payout(race_id="PAST_R001", bet_type="三連複", combo="1-3-5", amount=8000))
        session.add(Payout(race_id="PAST_R001", bet_type="三連単", combo="3→1→5", amount=30000))
        session.commit()
        yield session


def test_compute_past_race_odds_tan_all_horses(past_race_session):
    """単勝: entries.odds_win から全馬のオッズを返す。"""
    result = compute_past_race_odds(past_race_session, "PAST_R001")
    assert "単勝" in result
    tan = result["単勝"]
    # 5 頭全員分
    assert len(tan) == 5
    assert tan["1"] == pytest.approx(5.0)
    assert tan["2"] == pytest.approx(12.0)
    assert tan["3"] == pytest.approx(3.0)
    assert tan["4"] == pytest.approx(20.0)
    assert tan["5"] == pytest.approx(8.0)


def test_compute_past_race_odds_fuku_top3_only(past_race_session):
    """複勝: 1〜3 着馬のみ確定オッズを返す（4着以下は含まない）。"""
    result = compute_past_race_odds(past_race_session, "PAST_R001")
    assert "複勝" in result
    fuku = result["複勝"]
    # 1着=post3, 2着=post1, 3着=post5 → payout 120/100=1.2, 170/100=1.7, 140/100=1.4
    assert "3" in fuku
    assert "1" in fuku
    assert "5" in fuku
    # 4 着・5 着馬（post 2, post 4）は含まない
    assert "2" not in fuku
    assert "4" not in fuku
    assert fuku["3"] == pytest.approx(1.2)
    assert fuku["1"] == pytest.approx(1.7)
    assert fuku["5"] == pytest.approx(1.4)


def test_compute_past_race_odds_renki_winning_combos(past_race_session):
    """連系: payouts テーブルの的中 combo のみ返す。"""
    result = compute_past_race_odds(past_race_session, "PAST_R001")
    # 馬連
    assert "馬連" in result
    assert result["馬連"]["1-3"] == pytest.approx(35.0)
    # ワイド
    assert "ワイド" in result
    assert result["ワイド"]["1-3"] == pytest.approx(8.0)
    assert result["ワイド"]["3-5"] == pytest.approx(6.0)
    # 馬単
    assert "馬単" in result
    assert result["馬単"]["3→1"] == pytest.approx(60.0)
    # 三連複
    assert "三連複" in result
    assert result["三連複"]["1-3-5"] == pytest.approx(80.0)
    # 三連単
    assert "三連単" in result
    assert result["三連単"]["3→1→5"] == pytest.approx(300.0)


def test_compute_past_race_odds_empty_for_unknown_race(past_race_session):
    """存在しない race_id は空 dict を返す。"""
    result = compute_past_race_odds(past_race_session, "NONEXISTENT")
    assert result == {}


def test_compute_past_race_odds_empty_when_no_data(empty_session):
    """テーブルが空の場合は空 dict を返す。"""
    result = compute_past_race_odds(empty_session, "R001")
    assert result == {}
