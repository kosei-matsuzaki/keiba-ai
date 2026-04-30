"""Horse historical performance features.

All aggregations are strictly before before_date to prevent target leakage.
"""

from __future__ import annotations

import math
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from keiba_ai.db.models.entry import Entry
from keiba_ai.db.models.race import Race


def compute_horse_history(
    session: Session,
    horse_id: str,
    before_date: date,
    distance: int | None = None,
    course: str | None = None,
    n_recent: int = 5,
) -> dict[str, float | int | None]:
    """Compute horse performance aggregates using only races strictly before before_date.

    Returns a dict with NaN values for horses with no prior history, letting
    LightGBM handle missing data natively.

    New fields (PR-C):
        recent_avg_agari_3f: average agari_3f over last n_recent races (None excluded)
        days_since_last_race: calendar days since most recent race, or NaN if none
        wins_same_course: number of wins at the given course before before_date
        recent_finish_1/2/3: finish positions of the last 1/2/3 races (NaN if missing)
    """
    before_str = before_date.isoformat()

    stmt = (
        select(Entry, Race)
        .join(Race, Entry.race_id == Race.race_id)
        .where(Entry.horse_id == horse_id)
        .where(Race.date < before_str)
        .order_by(Race.date.desc())
    )
    rows = session.execute(stmt).all()

    nan = math.nan

    if not rows:
        return {
            "recent_avg_finish": nan,
            "recent_n_starts": 0,
            "starts_same_distance": 0,
            "starts_same_course": 0,
            "recent_avg_agari_3f": nan,
            "days_since_last_race": nan,
            "wins_same_course": 0,
            "recent_finish_1": nan,
            "recent_finish_2": nan,
            "recent_finish_3": nan,
        }

    recent_rows = rows[:n_recent]
    finish_positions = [
        r.Entry.finish_position
        for r in recent_rows
        if r.Entry.finish_position is not None
    ]

    recent_avg_finish = (
        sum(finish_positions) / len(finish_positions) if finish_positions else nan
    )

    starts_same_distance = (
        sum(1 for r in rows if distance is not None and r.Race.distance == distance)
        if distance is not None
        else 0
    )
    starts_same_course = (
        sum(1 for r in rows if course is not None and r.Race.course == course)
        if course is not None
        else 0
    )

    # PR-C: average agari_3f over last n_recent starts (skip None values)
    agari_values = [
        r.Entry.agari_3f
        for r in recent_rows
        if r.Entry.agari_3f is not None
    ]
    recent_avg_agari_3f = sum(agari_values) / len(agari_values) if agari_values else nan

    # PR-C: days between the most recent race and before_date
    last_race_date_str = rows[0].Race.date
    try:
        last_race_date = date.fromisoformat(last_race_date_str)
        days_since_last_race = float((before_date - last_race_date).days)
    except (ValueError, TypeError):
        days_since_last_race = nan

    # PR-C: win count at the given course (finish_position == 1, before before_date)
    wins_same_course = (
        sum(
            1
            for r in rows
            if course is not None
            and r.Race.course == course
            and r.Entry.finish_position == 1
        )
        if course is not None
        else 0
    )

    # PR-C: individual finish positions for the last 3 races
    def _nth_finish(n: int) -> float:
        """Return finish_position of the n-th most recent race (1-indexed), or NaN."""
        if len(rows) >= n:
            pos = rows[n - 1].Entry.finish_position
            return float(pos) if pos is not None else nan
        return nan

    return {
        "recent_avg_finish": recent_avg_finish,
        "recent_n_starts": len(rows),
        "starts_same_distance": starts_same_distance,
        "starts_same_course": starts_same_course,
        "recent_avg_agari_3f": recent_avg_agari_3f,
        "days_since_last_race": days_since_last_race,
        "wins_same_course": wins_same_course,
        "recent_finish_1": _nth_finish(1),
        "recent_finish_2": _nth_finish(2),
        "recent_finish_3": _nth_finish(3),
    }
