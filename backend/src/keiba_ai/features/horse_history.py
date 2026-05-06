"""Horse historical performance features.

All aggregations are strictly before before_date to prevent target leakage.

Two implementations:

  compute_horse_history          — per-horse SQL query (1 query per call). Used
                                   for one-off inference (build_inference_frame).

  HorseHistoryCache + compute_horse_history_from_cache
                                 — preload-all-once then per-call pandas slice.
                                   Used by build_training_frame to eliminate the
                                   N+1 SQL pattern (the bulk pre-load is one
                                   query for the entire training run).
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

    Returned keys:
        recent_avg_finish, recent_n_starts, starts_same_distance,
        starts_same_course, wins_same_course, horse_course_place_rate,
        recent_avg_agari_3f, days_since_last_race,
        recent_finish_1, recent_finish_2, recent_finish_3.

    horse_course_place_rate is the share of finishes <= 3 at the given course
    (NaN when course is None or starts_same_course == 0). It is consumed by
    builder.py to compute `course_place_rate_vs_field` and is not itself part
    of FEATURE_COLUMNS.
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
            "horse_course_place_rate": nan,
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

    # Win / place counts at the given course (before before_date).
    # place rate uses starts_same_course as denominator; NaN when 0 starts.
    wins_same_course = 0
    places_same_course = 0
    if course is not None:
        for r in rows:
            if r.Race.course != course:
                continue
            pos = r.Entry.finish_position
            if pos == 1:
                wins_same_course += 1
            if pos is not None and pos <= 3:
                places_same_course += 1
    horse_course_place_rate = (
        places_same_course / starts_same_course
        if starts_same_course > 0
        else nan
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
        "horse_course_place_rate": horse_course_place_rate,
        "recent_finish_1": _nth_finish(1),
        "recent_finish_2": _nth_finish(2),
        "recent_finish_3": _nth_finish(3),
    }


# ---------------------------------------------------------------------------
# Bulk-preload variant — eliminates N+1 SQL by loading all horse history once.
# ---------------------------------------------------------------------------


@dataclass
class HorseHistoryCache:
    """Pre-loaded horse race history for fast feature lookup.

    Build once at the start of build_training_frame; pass to
    compute_horse_history_from_cache for each (horse, target_date) pair.

    Memory: ~24 bytes per past entry × ~100k entries (2 years) ≈ 2 MB.
    """

    # DataFrame with columns: horse_id, date (str), distance (int), course (str),
    # finish_position (int|None), agari_3f (float|None).
    # Rows already sorted by (horse_id, date desc) for fast head() of recent.
    df: pd.DataFrame
    # Per-horse subset cache so each lookup avoids a full-frame filter.
    # Sorted by date desc within each group.
    by_horse: dict[str, pd.DataFrame]


def build_horse_history_cache(session: Session) -> HorseHistoryCache:
    """Load all horse race history in one SQL pass.

    Returns a HorseHistoryCache pre-grouped by horse_id with each group
    sorted descending by date (for fast recent-N head() calls).
    """
    query = (
        select(
            Entry.horse_id,
            Race.date,
            Race.distance,
            Race.course,
            Entry.finish_position,
            Entry.agari_3f,
        )
        .join(Race, Entry.race_id == Race.race_id)
    )
    rows = session.execute(query).all()
    df = pd.DataFrame(
        rows,
        columns=["horse_id", "date", "distance", "course", "finish_position", "agari_3f"],
    )

    # Sort by date desc so head(n_recent) gives the n most recent races.
    # Stable sort keeps insertion order across ties (rare but harmless).
    df = df.sort_values(["horse_id", "date"], ascending=[True, False], kind="stable")

    by_horse = {hid: g for hid, g in df.groupby("horse_id", sort=False)}
    return HorseHistoryCache(df=df, by_horse=by_horse)


def compute_horse_history_from_cache(
    cache: HorseHistoryCache,
    horse_id: str,
    before_date: date,
    distance: int | None = None,
    course: str | None = None,
    n_recent: int = 5,
) -> dict[str, float | int | None]:
    """Cached version of compute_horse_history.

    Output is bit-for-bit identical to compute_horse_history; only the data
    source changes (pandas slice instead of SQL query). The function exists to
    enable batch feature building without per-pair SQL round-trips.
    """
    nan = math.nan
    horse_df = cache.by_horse.get(horse_id)
    if horse_df is None:
        return {
            "recent_avg_finish": nan,
            "recent_n_starts": 0,
            "starts_same_distance": 0,
            "starts_same_course": 0,
            "recent_avg_agari_3f": nan,
            "days_since_last_race": nan,
            "wins_same_course": 0,
            "horse_course_place_rate": nan,
            "recent_finish_1": nan,
            "recent_finish_2": nan,
            "recent_finish_3": nan,
        }

    before_str = before_date.isoformat()
    h = horse_df[horse_df["date"] < before_str]

    if h.empty:
        return {
            "recent_avg_finish": nan,
            "recent_n_starts": 0,
            "starts_same_distance": 0,
            "starts_same_course": 0,
            "recent_avg_agari_3f": nan,
            "days_since_last_race": nan,
            "wins_same_course": 0,
            "horse_course_place_rate": nan,
            "recent_finish_1": nan,
            "recent_finish_2": nan,
            "recent_finish_3": nan,
        }

    # h is already sorted by date desc within the group — head(n_recent) gives
    # the n most recent rows.
    recent = h.head(n_recent)

    finish_positions_recent = recent["finish_position"].dropna()
    recent_avg_finish = (
        float(finish_positions_recent.mean()) if not finish_positions_recent.empty else nan
    )

    starts_same_distance = (
        int((h["distance"] == distance).sum()) if distance is not None else 0
    )

    if course is not None:
        same_course_mask = h["course"] == course
        starts_same_course = int(same_course_mask.sum())
    else:
        same_course_mask = None
        starts_same_course = 0

    agari_recent = recent["agari_3f"].dropna()
    recent_avg_agari_3f = float(agari_recent.mean()) if not agari_recent.empty else nan

    # Most recent date is the first row (sorted desc).
    last_race_date_str = str(h["date"].iloc[0])
    try:
        last_race_date = date.fromisoformat(last_race_date_str)
        days_since_last_race = float((before_date - last_race_date).days)
    except (ValueError, TypeError):
        days_since_last_race = nan

    wins_same_course = 0
    places_same_course = 0
    if same_course_mask is not None and starts_same_course > 0:
        same_course = h[same_course_mask]
        # finish_position can be None; use sum() with bool conditions excluding NaN.
        wins_same_course = int((same_course["finish_position"] == 1).sum())
        # Place: finish in [1, 3]; pandas treats NaN as False in comparisons.
        places_same_course = int(
            ((same_course["finish_position"] >= 1) & (same_course["finish_position"] <= 3)).sum()
        )

    horse_course_place_rate = (
        places_same_course / starts_same_course
        if starts_same_course > 0
        else nan
    )

    def _nth_finish(n: int) -> float:
        if len(h) >= n:
            pos = h["finish_position"].iloc[n - 1]
            return float(pos) if pd.notna(pos) else nan
        return nan

    return {
        "recent_avg_finish": recent_avg_finish,
        "recent_n_starts": int(len(h)),
        "starts_same_distance": starts_same_distance,
        "starts_same_course": starts_same_course,
        "recent_avg_agari_3f": recent_avg_agari_3f,
        "days_since_last_race": days_since_last_race,
        "wins_same_course": wins_same_course,
        "horse_course_place_rate": horse_course_place_rate,
        "recent_finish_1": _nth_finish(1),
        "recent_finish_2": _nth_finish(2),
        "recent_finish_3": _nth_finish(3),
    }
