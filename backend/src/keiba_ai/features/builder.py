"""Feature frame construction for training and inference.

build_training_frame and build_inference_frame both delegate to _build_entry_row,
which strictly uses only data before each race's date to prevent leakage.

Within-race relative features (compute_within_race_features) require all
entries for a race to be available simultaneously. To populate
`jockey_recent_win_rate_vs_field` and `course_place_rate_vs_field` without
duplicating DB queries, we build raw entry rows first (which compute jockey
recent win rate and horse same-course place rate), then derive the per-race
relative dict from those values and merge it back.
"""

from __future__ import annotations

import math
from datetime import date, datetime

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from keiba_ai.db.models.entry import Entry
from keiba_ai.db.models.horse import Horse
from keiba_ai.db.models.race import Race
from keiba_ai.features.course import extract_race_features
from keiba_ai.features.horse_history import compute_horse_history
from keiba_ai.features.jockey import compute_jockey_stats
from keiba_ai.features.odds import extract_odds_features
from keiba_ai.features.pedigree import compute_pedigree_features
from keiba_ai.features.relative_features import compute_within_race_features
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
    # Horse history (original)
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
    # Horse history extensions (PR-C)
    "recent_avg_agari_3f",
    "days_since_last_race",
    "wins_same_course",
    "recent_finish_1",
    "recent_finish_2",
    "recent_finish_3",
    # Within-race relative features (PR-C)
    "horse_weight_pct",
    "odds_win_rank",
    "weight_carried_pct",
    "jockey_recent_win_rate_vs_field",
    "course_place_rate_vs_field",
    "odds_win_diff_from_favorite",
    # Pedigree (PR-C)
    "sire_progeny_win_rate",
    "dam_progeny_win_rate",
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
    Relative features are absent here — _build_race_rows merges them in a
    second pass once the full field is known.
    """
    horse_feats = compute_horse_history(
        session,
        entry.horse_id,
        before_date=race_date,
        distance=race.distance,
        course=race.course,
    )
    jockey_feats = (
        compute_jockey_stats(
            session,
            entry.jockey_id,
            before_date=race_date,
            course=race.course,
            days=30,
        )
        if entry.jockey_id
        else {
            "jockey_recent_win_rate": math.nan,
            "jockey_recent_place_rate": math.nan,
            "jockey_course_place_rate": math.nan,
        }
    )
    trainer_feats = (
        compute_trainer_stats(
            session,
            entry.trainer_id,
            before_date=race_date,
            course=race.course,
        )
        if entry.trainer_id
        else {"trainer_course_place_rate": math.nan}
    )

    race_feats = extract_race_features(race, entry, n_runners)
    odds_feats = extract_odds_features(entry)

    # Pedigree features (PR-C); None → NaN so LightGBM gets float columns
    horse = session.get(Horse, entry.horse_id)
    sire = horse.sire if horse else None
    dam = horse.dam if horse else None
    _ped = compute_pedigree_features(session, sire, dam, race_date)
    pedigree_feats = {
        k: (v if v is not None else math.nan) for k, v in _ped.items()
    }

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
    row.update(pedigree_feats)
    return row


def _build_race_rows(
    session: Session,
    race: Race,
    entries: list[Entry],
) -> list[dict[str, object]]:
    """Build all entry rows for a single race, including within-race relative features.

    Two-phase: build raw rows first so jockey_recent_win_rate and
    horse_course_place_rate become available, then derive the relative dict
    and merge it back. This avoids re-querying the DB for those stats.
    """
    n_runners = race.n_runners or len(entries)
    race_date = _parse_date(race.date)

    rows = [_build_entry_row(session, race, entry, n_runners, race_date) for entry in entries]

    nan = math.nan
    jockey_recent_win_rates = {
        row["horse_id"]: row.get("jockey_recent_win_rate", nan) for row in rows
    }
    horse_course_place_rates = {
        row["horse_id"]: row.get("horse_course_place_rate", nan) for row in rows
    }

    relative_dict = compute_within_race_features(
        entries,
        jockey_recent_win_rates=jockey_recent_win_rates,
        horse_course_place_rates=horse_course_place_rates,
    )
    for row in rows:
        row.update(relative_dict.get(row["horse_id"], {}))
    return rows


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
        return pd.DataFrame(
            columns=["race_id", "horse_id", "date", "finish_position"] + FEATURE_COLUMNS
        )

    rows: list[dict[str, object]] = []
    for race in races:
        entry_stmt = select(Entry).where(Entry.race_id == race.race_id)
        entries = list(session.scalars(entry_stmt).all())
        if not entries:
            continue
        rows.extend(_build_race_rows(session, race, entries))

    df = pd.DataFrame(rows)
    # Ensure all feature columns exist (fill with NaN if missing)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    return df


def build_inference_frame(session: Session, race_id: str) -> pd.DataFrame:
    """Build a feature DataFrame for a single race (no finish_position).

    Usable at entry-form stage — finish_position is excluded.
    Uses the race's own date as the cutoff for historical lookups.
    """
    race = session.get(Race, race_id)
    if race is None:
        raise ValueError(f"Race {race_id!r} not found")

    entry_stmt = select(Entry).where(Entry.race_id == race_id)
    entries = list(session.scalars(entry_stmt).all())

    rows = _build_race_rows(session, race, entries)
    for row in rows:
        row.pop("finish_position", None)

    df = pd.DataFrame(rows)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    return df
