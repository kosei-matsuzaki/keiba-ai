"""Trainer performance features.

All aggregations are strictly before before_date to prevent target leakage.

Two implementations (mirror of horse_history.py / jockey.py):
  compute_trainer_stats               — per-trainer SQL (1 query / call)
  build_trainer_history_cache +
    compute_trainer_stats_from_cache  — preload all once, in-memory pandas slice
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import pandas as pd
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


# ---------------------------------------------------------------------------
# Bulk-preload variant
# ---------------------------------------------------------------------------


@dataclass
class TrainerHistoryCache:
    """Pre-loaded trainer race history for fast feature lookup."""

    df: pd.DataFrame  # cols: trainer_id, date(str), course, finish_position
    by_trainer: dict[str, pd.DataFrame]


def build_trainer_history_cache(session: Session) -> TrainerHistoryCache:
    """Load all trainer × race history in one SQL query.

    NULL trainer_id rows are skipped (no stats computable).
    """
    query = (
        select(
            Entry.trainer_id,
            Race.date,
            Race.course,
            Entry.finish_position,
        )
        .join(Race, Entry.race_id == Race.race_id)
        .where(Entry.trainer_id.is_not(None))
    )
    rows = session.execute(query).all()
    df = pd.DataFrame(
        rows,
        columns=["trainer_id", "date", "course", "finish_position"],
    )
    df = df.sort_values(["trainer_id", "date"], ascending=[True, True], kind="stable")
    by_trainer = {tid: g for tid, g in df.groupby("trainer_id", sort=False)}
    return TrainerHistoryCache(df=df, by_trainer=by_trainer)


def compute_trainer_stats_from_cache(
    cache: TrainerHistoryCache,
    trainer_id: str,
    before_date: date,
    course: str | None = None,
) -> dict[str, float]:
    """Cached counterpart of compute_trainer_stats; bit-for-bit identical output."""
    before_str = before_date.isoformat()

    trainer_df = cache.by_trainer.get(trainer_id)
    if trainer_df is None:
        return {"trainer_course_place_rate": math.nan}

    mask = trainer_df["date"] < before_str
    if course is not None:
        mask &= trainer_df["course"] == course

    rows = trainer_df[mask]
    if rows.empty:
        return {"trainer_course_place_rate": math.nan}

    places = int(
        ((rows["finish_position"] >= 1) & (rows["finish_position"] <= 3)).sum()
    )
    return {"trainer_course_place_rate": places / len(rows)}
