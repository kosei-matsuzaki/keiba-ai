"""Tests for features/horse_history.py.

Critical: verifies that compute_horse_history never uses data from
races on or after before_date (leakage prevention).
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import keiba_ai.db.models  # noqa: F401
from keiba_ai.db.base import Base
from keiba_ai.db.models.entry import Entry
from keiba_ai.db.models.horse import Horse
from keiba_ai.db.models.race import Race
from keiba_ai.features.horse_history import compute_horse_history


@pytest.fixture()
def leakage_engine():
    """Create a DB with one horse that has 3 races: past, today, future."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)

    today = date(2024, 6, 15)
    past = today - timedelta(days=10)
    future = today + timedelta(days=10)

    with Session(engine) as session:
        session.add(Horse(horse_id="H001", name=None))
        for rid, _d, _pos in [
            ("R001", past, 1),
            ("R002", today, 2),
            ("R003", future, 3),
        ]:
            session.add(
                Race(
                    race_id=rid,
                    date=_d.isoformat(),
                    course="東京",
                    surface="芝",
                    distance=1600,
                    n_runners=8,
                )
            )
        session.flush()
        for rid, _d, pos in [
            ("R001", past, 1),
            ("R002", today, 2),
            ("R003", future, 3),
        ]:
            session.add(
                Entry(
                    race_id=rid,
                    horse_id="H001",
                    post_position=1,
                    finish_position=pos,
                )
            )
        session.commit()

    yield engine
    engine.dispose()


def test_leakage_before_date_only(leakage_engine):
    """Only the past race (finish=1) should contribute to the aggregate."""
    with Session(leakage_engine) as session:
        result = compute_horse_history(
            session,
            horse_id="H001",
            before_date=date(2024, 6, 15),  # today's race is NOT included
        )

    # Only R001 (past, pos=1) should be included
    assert result["recent_n_starts"] == 1
    assert result["recent_avg_finish"] == pytest.approx(1.0)


def test_leakage_future_not_included(leakage_engine):
    """With before_date = past, only zero races qualify."""
    with Session(leakage_engine) as session:
        result = compute_horse_history(
            session,
            horse_id="H001",
            before_date=date(2024, 6, 1),  # before all races
        )

    assert result["recent_n_starts"] == 0
    assert math.isnan(result["recent_avg_finish"])


def test_no_history_returns_nan(leakage_engine):
    with Session(leakage_engine) as session:
        result = compute_horse_history(
            session,
            horse_id="UNKNOWN",
            before_date=date(2024, 6, 15),
        )

    assert result["recent_n_starts"] == 0
    assert math.isnan(result["recent_avg_finish"])


def test_same_distance_filter(leakage_engine):
    with Session(leakage_engine) as session:
        result = compute_horse_history(
            session,
            horse_id="H001",
            before_date=date(2024, 6, 15),
            distance=1600,
        )
    # R001 has distance=1600 and is before before_date
    assert result["starts_same_distance"] == 1


def test_same_course_filter(leakage_engine):
    with Session(leakage_engine) as session:
        result = compute_horse_history(
            session,
            horse_id="H001",
            before_date=date(2024, 6, 15),
            course="東京",
        )
    assert result["starts_same_course"] == 1


# ── PR-C: new field tests ─────────────────────────────────────────────────────


@pytest.fixture()
def rich_engine():
    """DB with one horse, 4 past races with varied data for PR-C feature tests."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)

    base = date(2024, 6, 15)

    with Session(engine) as session:
        session.add(Horse(horse_id="H001", name=None))
        # Races at different dates (all before 2024-06-15)
        # R1: 30 days ago, finish=1, agari_3f=34.5, 東京1600
        # R2: 20 days ago, finish=2, agari_3f=35.0, 東京1600
        # R3: 10 days ago, finish=3, agari_3f=None, 中山1800
        # R4:  5 days ago, finish=1, agari_3f=33.8, 東京1600
        races = [
            ("R001", base - timedelta(days=30), 1,   34.5, "東京", 1600),
            ("R002", base - timedelta(days=20), 2,   35.0, "東京", 1600),
            ("R003", base - timedelta(days=10), 3,   None, "中山", 1800),
            ("R004", base - timedelta(days=5),  1,   33.8, "東京", 1600),
        ]
        for rid, d, _finish, _agari, course, dist in races:
            session.add(
                Race(
                    race_id=rid,
                    date=d.isoformat(),
                    course=course,
                    surface="芝",
                    distance=dist,
                    n_runners=8,
                )
            )
        session.flush()
        for rid, _d, finish, agari, _course, _dist in races:
            session.add(
                Entry(
                    race_id=rid,
                    horse_id="H001",
                    post_position=1,
                    finish_position=finish,
                    agari_3f=agari,
                )
            )
        session.commit()

    yield engine
    engine.dispose()


def test_compute_horse_history_includes_avg_agari_3f(rich_engine):
    """recent_avg_agari_3f averages only non-None agari_3f values in last 5 races."""
    with Session(rich_engine) as session:
        result = compute_horse_history(
            session,
            horse_id="H001",
            before_date=date(2024, 6, 15),
        )

    # Valid agari values in last 5 races: 34.5, 35.0, 33.8 (R3 is None)
    expected = (34.5 + 35.0 + 33.8) / 3
    assert result["recent_avg_agari_3f"] == pytest.approx(expected)


def test_compute_horse_history_includes_days_since_last_race(rich_engine):
    """days_since_last_race is the gap from the most recent race to before_date."""
    with Session(rich_engine) as session:
        result = compute_horse_history(
            session,
            horse_id="H001",
            before_date=date(2024, 6, 15),
        )

    # Most recent race is R4, 5 days before before_date
    assert result["days_since_last_race"] == pytest.approx(5.0)


def test_compute_horse_history_includes_wins_same_course(rich_engine):
    """wins_same_course counts finish_position==1 on the given course."""
    with Session(rich_engine) as session:
        result = compute_horse_history(
            session,
            horse_id="H001",
            before_date=date(2024, 6, 15),
            course="東京",
        )

    # R001 (finish=1) and R004 (finish=1) are both at 東京
    assert result["wins_same_course"] == 2


def test_compute_horse_history_includes_recent_finish_n(rich_engine):
    """recent_finish_1/2/3 are the last 3 finish positions in reverse-date order."""
    with Session(rich_engine) as session:
        result = compute_horse_history(
            session,
            horse_id="H001",
            before_date=date(2024, 6, 15),
        )

    # Sorted desc by date: R4(finish=1), R3(finish=3), R2(finish=2), R1(finish=1)
    assert result["recent_finish_1"] == pytest.approx(1.0)
    assert result["recent_finish_2"] == pytest.approx(3.0)
    assert result["recent_finish_3"] == pytest.approx(2.0)


def test_compute_horse_history_includes_course_place_rate(rich_engine):
    """horse_course_place_rate = (finishes <= 3 at course) / starts_same_course."""
    with Session(rich_engine) as session:
        result = compute_horse_history(
            session,
            horse_id="H001",
            before_date=date(2024, 6, 15),
            course="東京",
        )
    # At 東京: R1(finish=1), R2(finish=2), R4(finish=1) → 3 places / 3 starts
    assert result["starts_same_course"] == 3
    assert result["horse_course_place_rate"] == pytest.approx(1.0)


def test_horse_course_place_rate_nan_without_course(rich_engine):
    """horse_course_place_rate is NaN when course filter is not provided."""
    with Session(rich_engine) as session:
        result = compute_horse_history(
            session,
            horse_id="H001",
            before_date=date(2024, 6, 15),
        )
    assert math.isnan(result["horse_course_place_rate"])


def test_horse_history_excludes_races_after_before_date(rich_engine):
    """Leakage prevention: races on or after before_date must not appear.

    The rich_engine fixture has 4 races:
      R1: 30d ago (2024-05-16), R2: 20d ago (2024-05-26),
      R3: 10d ago (2024-06-05), R4:  5d ago (2024-06-10).

    Using cutoff = 2024-05-28 (17 days before base 2024-06-15) means only
    R1 (2024-05-16) and R2 (2024-05-26) qualify (both < 2024-05-28).
    R3 and R4 are on or after the cutoff and must be excluded.
    """
    base = date(2024, 6, 15)
    # cutoff = 2024-05-28; R1(5/16) and R2(5/26) are before it
    before = base - timedelta(days=18)  # = 2024-05-28

    with Session(rich_engine) as session:
        result = compute_horse_history(
            session,
            horse_id="H001",
            before_date=before,
        )

    assert result["recent_n_starts"] == 2
    # Most recent qualifying race is R2 = 2024-05-26
    r2_date = base - timedelta(days=20)  # 2024-05-26
    expected_days = float((before - r2_date).days)
    assert result["days_since_last_race"] == pytest.approx(expected_days)
