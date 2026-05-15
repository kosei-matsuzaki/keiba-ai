"""Tests for features/relative_features.py.

All tests use plain Entry-like objects (or actual Entry ORM instances backed by
an in-memory DB) — no external API calls.
"""

from __future__ import annotations

import math

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import db.models  # noqa: F401
from db.base import Base
from db.models.entry import Entry
from db.models.horse import Horse
from db.models.race import Race
from features.relative_features import compute_within_race_features


@pytest.fixture()
def rel_engine():
    """In-memory DB with one race and 4 entries with varied horse_weight/odds."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        for hid in ["H001", "H002", "H003", "H004"]:
            session.add(Horse(horse_id=hid, name=None))
        session.add(
            Race(
                race_id="R001",
                date="2024-06-15",
                course="東京",
                surface="芝",
                distance=1600,
                n_runners=4,
            )
        )
        session.flush()
        entries = [
            # horse_id, post, horse_weight, odds_win, weight_carried
            ("H001", 1, 450, 2.0,  54.0),
            ("H002", 2, 480, 5.0,  55.0),
            ("H003", 3, 510, 10.0, 53.0),
            ("H004", 4, 420, 3.0,  56.0),
        ]
        for hid, post, hw, odds, wc in entries:
            session.add(
                Entry(
                    race_id="R001",
                    horse_id=hid,
                    post_position=post,
                    horse_weight=hw,
                    odds_win=odds,
                    weight_carried=wc,
                    finish_position=post,
                )
            )
        session.commit()

    yield engine
    engine.dispose()


def _get_entries(engine) -> list[Entry]:
    with Session(engine) as session:
        entries = list(session.scalars(
            __import__("sqlalchemy", fromlist=["select"]).select(Entry)
            .where(Entry.race_id == "R001")
        ).all())
        # Expunge so they can be used outside the session
        session.expunge_all()
        return entries


def test_horse_weight_pct_range_0_to_1(rel_engine):
    """horse_weight_pct must be in [0.0, 1.0] for all horses."""
    with Session(rel_engine) as session:
        entries = list(session.scalars(
            __import__("sqlalchemy", fromlist=["select"]).select(Entry)
            .where(Entry.race_id == "R001")
        ).all())
        result = compute_within_race_features(entries)

    for hid, feats in result.items():
        pct = feats["horse_weight_pct"]
        assert 0.0 <= pct <= 1.0, f"{hid}: horse_weight_pct={pct} out of range"


def test_odds_win_rank_starts_at_1(rel_engine):
    """The favourite (lowest odds) must receive odds_win_rank == 1."""
    with Session(rel_engine) as session:
        entries = list(session.scalars(
            __import__("sqlalchemy", fromlist=["select"]).select(Entry)
            .where(Entry.race_id == "R001")
        ).all())
        result = compute_within_race_features(entries)

    # H001 has odds_win=2.0 (lowest) → rank 1
    assert result["H001"]["odds_win_rank"] == pytest.approx(1.0)
    # All ranks are >= 1
    for hid, feats in result.items():
        assert feats["odds_win_rank"] >= 1.0, f"{hid}: rank < 1"


def test_odds_win_diff_from_favorite_zero_for_favorite(rel_engine):
    """The favourite's odds_win_diff_from_favorite must be 0.0."""
    with Session(rel_engine) as session:
        entries = list(session.scalars(
            __import__("sqlalchemy", fromlist=["select"]).select(Entry)
            .where(Entry.race_id == "R001")
        ).all())
        result = compute_within_race_features(entries)

    # H001 has the lowest odds (2.0) → diff = 0
    assert result["H001"]["odds_win_diff_from_favorite"] == pytest.approx(0.0)
    # All others should have positive diff
    for hid in ["H002", "H003", "H004"]:
        assert result[hid]["odds_win_diff_from_favorite"] > 0.0


def test_handles_missing_horse_weight_gracefully(rel_engine):
    """Entries with None horse_weight must produce NaN for horse_weight_pct."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        for hid in ["HA", "HB", "HC"]:
            session.add(Horse(horse_id=hid, name=None))
        session.add(
            Race(
                race_id="R999",
                date="2024-06-15",
                course="東京",
                surface="芝",
                distance=1600,
                n_runners=3,
            )
        )
        session.flush()
        session.add(Entry(race_id="R999", horse_id="HA", post_position=1, horse_weight=None, odds_win=2.0))
        session.add(Entry(race_id="R999", horse_id="HB", post_position=2, horse_weight=480, odds_win=3.0))
        session.add(Entry(race_id="R999", horse_id="HC", post_position=3, horse_weight=None, odds_win=5.0))
        session.commit()

        entries = list(session.scalars(
            __import__("sqlalchemy", fromlist=["select"]).select(Entry)
            .where(Entry.race_id == "R999")
        ).all())
        result = compute_within_race_features(entries)

    assert math.isnan(result["HA"]["horse_weight_pct"])
    assert math.isnan(result["HC"]["horse_weight_pct"])
    # HB is the only valid horse; percentile = 0.5 (single element)
    assert result["HB"]["horse_weight_pct"] == pytest.approx(0.5)

    engine.dispose()


def test_jockey_recent_win_rate_vs_field(rel_engine):
    """jockey_recent_win_rate_vs_field == rate - field_average."""
    with Session(rel_engine) as session:
        entries = list(session.scalars(
            __import__("sqlalchemy", fromlist=["select"]).select(Entry)
            .where(Entry.race_id == "R001")
        ).all())
        rates = {"H001": 0.20, "H002": 0.10, "H003": 0.30, "H004": 0.40}
        result = compute_within_race_features(entries, jockey_recent_win_rates=rates)

    field_avg = sum(rates.values()) / len(rates)  # 0.25
    for hid, rate in rates.items():
        expected = rate - field_avg
        assert result[hid]["jockey_recent_win_rate_vs_field"] == pytest.approx(expected)


def test_empty_entries_returns_empty_dict():
    result = compute_within_race_features([])
    assert result == {}
