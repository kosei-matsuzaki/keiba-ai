"""Trainer performance features.

All aggregations are strictly before before_date to prevent target leakage.
"""

from __future__ import annotations

import math
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from keiba_ai.db.models.entry import Entry
from keiba_ai.db.models.race import Race


def compute_trainer_stats(
    session: Session,
    trainer_id: str,
    before_date: date,
    course: str | None = None,
) -> dict[str, float]:
    """Compute trainer place rate on a given course using history before before_date."""
    before_str = before_date.isoformat()

    stmt = (
        select(Entry, Race)
        .join(Race, Entry.race_id == Race.race_id)
        .where(Entry.trainer_id == trainer_id)
        .where(Race.date < before_str)
    )
    if course is not None:
        stmt = stmt.where(Race.course == course)

    rows = session.execute(stmt).all()

    if not rows:
        return {"trainer_course_place_rate": math.nan}

    places = sum(
        1 for r in rows if r.Entry.finish_position is not None and r.Entry.finish_position <= 3
    )
    return {"trainer_course_place_rate": places / len(rows)}
