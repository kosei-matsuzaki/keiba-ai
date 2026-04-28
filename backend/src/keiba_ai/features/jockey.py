"""Jockey performance features.

All aggregations are strictly before before_date to prevent target leakage.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from keiba_ai.db.models.entry import Entry
from keiba_ai.db.models.race import Race


def compute_jockey_stats(
    session: Session,
    jockey_id: str,
    before_date: date,
    course: str | None = None,
    days: int = 30,
) -> dict[str, float]:
    """Compute jockey win/place rates over the last `days` days before before_date.

    If course is provided, also computes course-specific place rate.
    Returns NaN for jockeys with no qualifying history.
    """
    before_str = before_date.isoformat()
    since_str = (before_date - timedelta(days=days)).isoformat()

    # Recent-window stats
    recent_stmt = (
        select(Entry, Race)
        .join(Race, Entry.race_id == Race.race_id)
        .where(Entry.jockey_id == jockey_id)
        .where(Race.date < before_str)
        .where(Race.date >= since_str)
    )
    recent_rows = session.execute(recent_stmt).all()

    if recent_rows:
        wins = sum(1 for r in recent_rows if r.Entry.finish_position == 1)
        places = sum(
            1 for r in recent_rows if r.Entry.finish_position is not None and r.Entry.finish_position <= 3
        )
        n = len(recent_rows)
        recent_win_rate = wins / n
        recent_place_rate = places / n
    else:
        recent_win_rate = math.nan
        recent_place_rate = math.nan

    # Course-specific place rate (all history before before_date)
    if course is not None:
        course_stmt = (
            select(Entry, Race)
            .join(Race, Entry.race_id == Race.race_id)
            .where(Entry.jockey_id == jockey_id)
            .where(Race.date < before_str)
            .where(Race.course == course)
        )
        course_rows = session.execute(course_stmt).all()
        if course_rows:
            places_c = sum(
                1
                for r in course_rows
                if r.Entry.finish_position is not None and r.Entry.finish_position <= 3
            )
            course_place_rate = places_c / len(course_rows)
        else:
            course_place_rate = math.nan
    else:
        course_place_rate = math.nan

    return {
        "jockey_recent_win_rate": recent_win_rate,
        "jockey_recent_place_rate": recent_place_rate,
        "jockey_course_place_rate": course_place_rate,
    }
