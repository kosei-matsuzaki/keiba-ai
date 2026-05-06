"""Jockey performance features.

All aggregations are strictly before before_date to prevent target leakage.

Two implementations (mirror of horse_history.py):
  compute_jockey_stats               — per-jockey SQL (1-2 queries / call)
  build_jockey_history_cache +
    compute_jockey_stats_from_cache  — preload all once, in-memory pandas slice
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
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


# ---------------------------------------------------------------------------
# Bulk-preload variant (mirrors horse_history.HorseHistoryCache).
# ---------------------------------------------------------------------------


@dataclass
class JockeyHistoryCache:
    """Pre-loaded jockey race history for fast feature lookup."""

    df: pd.DataFrame  # cols: jockey_id, date(str), course, finish_position
    by_jockey: dict[str, pd.DataFrame]  # ascending by date within each group


def build_jockey_history_cache(session: Session) -> JockeyHistoryCache:
    """Load all jockey × race history with one SQL query.

    Excludes rows with NULL jockey_id (cannot compute stats anyway).
    Within each jockey group rows are sorted by date ascending so date-range
    filters can use binary search (np.searchsorted) cheaply.
    """
    query = (
        select(
            Entry.jockey_id,
            Race.date,
            Race.course,
            Entry.finish_position,
        )
        .join(Race, Entry.race_id == Race.race_id)
        .where(Entry.jockey_id.is_not(None))
    )
    rows = session.execute(query).all()
    df = pd.DataFrame(
        rows,
        columns=["jockey_id", "date", "course", "finish_position"],
    )
    df = df.sort_values(["jockey_id", "date"], ascending=[True, True], kind="stable")
    by_jockey = {jid: g for jid, g in df.groupby("jockey_id", sort=False)}
    return JockeyHistoryCache(df=df, by_jockey=by_jockey)


def compute_jockey_stats_from_cache(
    cache: JockeyHistoryCache,
    jockey_id: str,
    before_date: date,
    course: str | None = None,
    days: int = 30,
) -> dict[str, float]:
    """Cached counterpart of compute_jockey_stats; bit-for-bit identical output."""
    before_str = before_date.isoformat()
    since_str = (before_date - timedelta(days=days)).isoformat()

    jockey_df = cache.by_jockey.get(jockey_id)
    if jockey_df is None:
        return {
            "jockey_recent_win_rate": math.nan,
            "jockey_recent_place_rate": math.nan,
            "jockey_course_place_rate": math.nan,
        }

    # Recent window
    recent_mask = (jockey_df["date"] >= since_str) & (jockey_df["date"] < before_str)
    recent = jockey_df[recent_mask]

    if not recent.empty:
        wins = int((recent["finish_position"] == 1).sum())
        places = int(
            ((recent["finish_position"] >= 1) & (recent["finish_position"] <= 3)).sum()
        )
        n = len(recent)
        recent_win_rate = wins / n
        recent_place_rate = places / n
    else:
        recent_win_rate = math.nan
        recent_place_rate = math.nan

    # Course-specific (all history before before_date)
    if course is not None:
        before_mask = jockey_df["date"] < before_str
        course_mask = jockey_df["course"] == course
        course_rows = jockey_df[before_mask & course_mask]
        if not course_rows.empty:
            places_c = int(
                ((course_rows["finish_position"] >= 1) & (course_rows["finish_position"] <= 3)).sum()
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
