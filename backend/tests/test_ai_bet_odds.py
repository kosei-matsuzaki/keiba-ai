"""Tests for ai/bet_odds.py — baseline odds computation and compute_race_odds."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import keiba_ai.db.models  # noqa: F401
from keiba_ai.ai.bet_odds import (
    _FALLBACK_AMOUNTS,
    compute_baseline_odds,
    compute_baseline_odds_by_class,
    compute_race_odds,
)
from keiba_ai.db.base import Base
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
