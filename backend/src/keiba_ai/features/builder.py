"""Feature frame construction for training and inference.

build_training_frame and build_inference_frame both delegate to _build_entry_row,
which strictly uses only data before each race's date to prevent leakage.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from keiba_ai.db.models.entry import Entry
from keiba_ai.db.models.race import Race
from keiba_ai.features.course import extract_race_features
from keiba_ai.features.horse_history import compute_horse_history
from keiba_ai.features.jockey import compute_jockey_stats
from keiba_ai.features.odds import extract_odds_features
from keiba_ai.features.trainer import compute_trainer_stats

# Fixed column order — must stay stable across training and inference.
FEATURE_COLUMNS: list[str] = [
    # Race / course
    "distance",
    "n_runners",
    "post_position",
    "post_position_ratio",
    # Entry basics
    "age",
    "horse_weight",
    "horse_weight_diff",
    # Odds / market
    "odds_win",
    "popularity",
    "log_odds_win",
    # Horse history
    "recent_avg_finish",
    "recent_n_starts",
    "starts_same_distance",
    "starts_same_course",
    # Jockey
    "jockey_recent_win_rate",
    "jockey_recent_place_rate",
    "jockey_course_place_rate",
    # Trainer
    "trainer_course_place_rate",
    # Categorical (listed last; referenced by name in CATEGORICAL_FEATURES)
    "surface",
    "course",
    "weather",
    "track_condition",
    "race_class",
    "sex",
]

CATEGORICAL_FEATURES: list[str] = [
    "surface",
    "course",
    "weather",
    "track_condition",
    "race_class",
    "sex",
]


def _parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def _build_entry_row(
    session: Session,
    race: Race,
    entry: Entry,
    n_runners: int,
    race_date: date,
) -> dict[str, object]:
    """Build a single feature row for one entry in one race.

    All historical lookups use race_date (strictly before) to prevent leakage.
    """
    horse_feats = compute_horse_history(
        session,
        entry.horse_id,
        before_date=race_date,
        distance=race.distance,
        course=race.course,
    )
    jockey_feats = compute_jockey_stats(
        session,
        entry.jockey_id or "",
        before_date=race_date,
        course=race.course,
        days=30,
    ) if entry.jockey_id else {
        "jockey_recent_win_rate": float("nan"),
        "jockey_recent_place_rate": float("nan"),
        "jockey_course_place_rate": float("nan"),
    }
    trainer_feats = compute_trainer_stats(
        session,
        entry.trainer_id or "",
        before_date=race_date,
        course=race.course,
    ) if entry.trainer_id else {"trainer_course_place_rate": float("nan")}

    race_feats = extract_race_features(race, entry, n_runners)
    odds_feats = extract_odds_features(entry)

    row: dict[str, object] = {
        "race_id": race.race_id,
        "horse_id": entry.horse_id,
        "date": race.date,
        "finish_position": entry.finish_position,
        "payout_place": race.payout_place,
    }
    row.update(race_feats)
    row.update(odds_feats)
    row.update(horse_feats)
    row.update(jockey_feats)
    row.update(trainer_feats)
    return row


def _load_races_in_range(
    session: Session,
    start_date: str | None,
    end_date: str | None,
) -> list[Race]:
    stmt = select(Race).order_by(Race.date)
    if start_date:
        stmt = stmt.where(Race.date >= start_date)
    if end_date:
        stmt = stmt.where(Race.date <= end_date)
    return list(session.scalars(stmt).all())


def build_training_frame(
    session: Session,
    train_start: str | None = None,
    train_end: str | None = None,
) -> pd.DataFrame:
    """Build a feature DataFrame for all races in [train_start, train_end].

    Includes finish_position for label assignment.
    Leakage prevention: each entry's features are computed using only records
    strictly before that race's date.
    """
    races = _load_races_in_range(session, train_start, train_end)
    if not races:
        return pd.DataFrame(columns=["race_id", "horse_id", "date", "finish_position"] + FEATURE_COLUMNS)

    rows: list[dict[str, object]] = []
    for race in races:
        entry_stmt = select(Entry).where(Entry.race_id == race.race_id)
        entries = list(session.scalars(entry_stmt).all())
        if not entries:
            continue
        n_runners = race.n_runners or len(entries)
        race_date = _parse_date(race.date)
        for entry in entries:
            rows.append(_build_entry_row(session, race, entry, n_runners, race_date))

    df = pd.DataFrame(rows)
    # Ensure all feature columns exist (fill with NaN if missing)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    return df


def build_inference_frame(session: Session, race_id: str) -> pd.DataFrame:
    """Build a feature DataFrame for a single race (no finish_position).

    Usable at entry-form stage — finish_position is excluded.
    Uses today's date as the cutoff for historical lookups.
    """
    race = session.get(Race, race_id)
    if race is None:
        raise ValueError(f"Race {race_id!r} not found")

    entry_stmt = select(Entry).where(Entry.race_id == race_id)
    entries = list(session.scalars(entry_stmt).all())
    n_runners = race.n_runners or len(entries)
    race_date = _parse_date(race.date)

    rows: list[dict[str, object]] = []
    for entry in entries:
        row = _build_entry_row(session, race, entry, n_runners, race_date)
        row.pop("finish_position", None)
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    return df
