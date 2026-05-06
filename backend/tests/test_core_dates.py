"""Tests for keiba_ai.core.dates.this_weekend_dates."""

from __future__ import annotations

from datetime import date

import pytest

from keiba_ai.core.dates import this_weekend_dates


@pytest.mark.parametrize(
    "today_str, expected_sat_str, expected_sun_str",
    [
        # Monday → next Sat/Sun
        ("2026-05-04", "2026-05-09", "2026-05-10"),
        # Tuesday
        ("2026-05-05", "2026-05-09", "2026-05-10"),
        # Wednesday
        ("2026-05-06", "2026-05-09", "2026-05-10"),
        # Thursday
        ("2026-05-07", "2026-05-09", "2026-05-10"),
        # Friday
        ("2026-05-08", "2026-05-09", "2026-05-10"),
        # Saturday → current day and next day
        ("2026-05-09", "2026-05-09", "2026-05-10"),
        # Sunday → previous day and current day
        ("2026-05-10", "2026-05-09", "2026-05-10"),
    ],
)
def test_this_weekend_dates(
    today_str: str, expected_sat_str: str, expected_sun_str: str
) -> None:
    today = date.fromisoformat(today_str)
    sat, sun = this_weekend_dates(today=today)
    assert sat.isoformat() == expected_sat_str
    assert sun.isoformat() == expected_sun_str
    assert sun == sat + __import__("datetime").timedelta(days=1)


def test_sat_is_always_before_sun() -> None:
    """Invariant: sat < sun for any input day."""
    from datetime import timedelta

    # Walk through a full week starting from a known Monday
    base = date(2026, 5, 4)  # Monday
    for offset in range(7):
        sat, sun = this_weekend_dates(today=base + timedelta(days=offset))
        assert sat < sun
        assert sat.weekday() == 5  # Saturday
        assert sun.weekday() == 6  # Sunday
