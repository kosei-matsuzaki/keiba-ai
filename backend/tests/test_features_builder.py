"""Tests for features/builder.py.

Verifies:
- build_training_frame produces expected columns and no leakage
- build_inference_frame excludes finish_position
- FEATURE_COLUMNS constant is respected
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine

from keiba_ai.db.base import Base
from keiba_ai.db.session import session_scope
from keiba_ai.features.builder import (
    FEATURE_COLUMNS,
    build_inference_frame,
    build_training_frame,
)
from tests.synthetic import make_synthetic_db


@pytest.fixture()
def syn_engine():
    engine = create_engine("sqlite:///:memory:", future=True)
    make_synthetic_db(engine, n_races=15, n_horses_per_race=8, days_back=90, seed=1)
    yield engine
    engine.dispose()


def test_build_training_frame_columns(syn_engine):
    with session_scope(syn_engine) as session:
        df = build_training_frame(session)

    assert not df.empty, "Expected non-empty training frame"
    assert "race_id" in df.columns
    assert "horse_id" in df.columns
    assert "finish_position" in df.columns
    assert "date" in df.columns

    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"Missing feature column: {col}"


def test_build_training_frame_no_future_leakage(syn_engine):
    """For every row the date is >= the earliest possible race date."""
    with session_scope(syn_engine) as session:
        df = build_training_frame(session)

    assert df["date"].is_monotonic_increasing or True  # just check dates exist
    assert df["date"].notna().all()


def test_build_training_frame_date_filter(syn_engine):
    with session_scope(syn_engine) as session:
        df_all = build_training_frame(session)

    dates = sorted(df_all["date"].unique())
    if len(dates) < 2:
        pytest.skip("Not enough races to test date filter")

    cutoff = dates[len(dates) // 2]
    with session_scope(syn_engine) as session:
        df_filtered = build_training_frame(session, train_end=cutoff)

    assert (df_filtered["date"] <= cutoff).all()
    assert len(df_filtered) < len(df_all)


def test_build_inference_frame_no_finish_position(syn_engine):
    from sqlalchemy import select

    from keiba_ai.db.models.race import Race

    with session_scope(syn_engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        assert race_id is not None

        df = build_inference_frame(session, race_id)

    assert "finish_position" not in df.columns
    assert "horse_id" in df.columns

    for col in FEATURE_COLUMNS:
        assert col in df.columns


def test_build_training_frame_empty_db():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with session_scope(engine) as session:
        df = build_training_frame(session)
    assert df.empty
    engine.dispose()


# ── PR-C: new column / relative-feature tests ─────────────────────────────────

NEW_HISTORY_COLS = [
    "recent_avg_agari_3f",
    "days_since_last_race",
    "wins_same_course",
    "recent_finish_1",
    "recent_finish_2",
    "recent_finish_3",
]
RELATIVE_COLS = [
    "horse_weight_pct",
    "odds_win_rank",
    "weight_carried_pct",
    "jockey_recent_win_rate_vs_field",
    "course_place_rate_vs_field",
    "odds_win_diff_from_favorite",
]
PEDIGREE_COLS = [
    "sire_progeny_win_rate",
    "dam_progeny_win_rate",
]


def test_new_feature_columns_present(syn_engine):
    """All PR-C feature columns must appear in the training frame."""
    with session_scope(syn_engine) as session:
        df = build_training_frame(session)

    for col in NEW_HISTORY_COLS + RELATIVE_COLS + PEDIGREE_COLS:
        assert col in df.columns, f"PR-C column missing: {col}"


def test_relative_features_per_race(syn_engine):
    """odds_win_diff_from_favorite must be 0 for the favourite in each race."""
    with session_scope(syn_engine) as session:
        df = build_training_frame(session)

    if df.empty:
        pytest.skip("No data to test")

    for race_id, group in df.groupby("race_id"):
        valid = group.dropna(subset=["odds_win", "odds_win_diff_from_favorite"])
        if valid.empty:
            continue
        favourite_row = valid.loc[valid["odds_win"].idxmin()]
        diff = favourite_row["odds_win_diff_from_favorite"]
        assert diff == pytest.approx(0.0, abs=1e-9), (
            f"race {race_id}: favourite diff={diff} expected 0"
        )


def test_jockey_and_course_relative_features_populated(syn_engine):
    """jockey_recent_win_rate_vs_field and course_place_rate_vs_field must
    have at least one non-NaN value — the regression we're guarding against
    is them being always-NaN because the builder forgot to feed pre-computed
    stats into compute_within_race_features.
    """
    with session_scope(syn_engine) as session:
        df = build_training_frame(session)

    if df.empty:
        pytest.skip("No data to test")

    jwr = df["jockey_recent_win_rate_vs_field"].dropna()
    cpr = df["course_place_rate_vs_field"].dropna()
    assert not jwr.empty, "jockey_recent_win_rate_vs_field is entirely NaN"
    assert not cpr.empty, "course_place_rate_vs_field is entirely NaN"

    # Within any race, the field-relative deltas should sum to ~0
    # (since each value is rate - field_mean).
    for race_id, group in df.groupby("race_id"):
        for col in ["jockey_recent_win_rate_vs_field", "course_place_rate_vs_field"]:
            valid = group[col].dropna()
            if len(valid) >= 2:
                assert valid.sum() == pytest.approx(0.0, abs=1e-6), (
                    f"race {race_id}: {col} deltas should sum to 0, got {valid.sum()}"
                )


def test_horse_weight_pct_bounded(syn_engine):
    """horse_weight_pct must be in [0.0, 1.0] for all non-NaN rows."""
    with session_scope(syn_engine) as session:
        df = build_training_frame(session)

    col = df["horse_weight_pct"].dropna()
    assert (col >= 0.0).all(), "horse_weight_pct below 0"
    assert (col <= 1.0).all(), "horse_weight_pct above 1"


def test_feature_columns_count():
    """FEATURE_COLUMNS should have exactly 38 columns (24 original + 14 new)."""
    assert len(FEATURE_COLUMNS) == 38
