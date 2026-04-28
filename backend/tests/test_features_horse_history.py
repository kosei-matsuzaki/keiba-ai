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
