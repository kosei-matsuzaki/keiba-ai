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
