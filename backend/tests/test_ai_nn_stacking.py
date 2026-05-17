"""Tests for ai.nn.stacking.augment_frame_with_gbdt."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine

from ai.gbm.train import train as gbdt_train
from ai.nn.stacking import GBDT_FEATURE_COLUMNS, augment_frame_with_gbdt
from ai.registry import load_model_full
from db.session import make_engine, session_scope
from features.builder import build_training_frame
from tests.synthetic import make_synthetic_db


@pytest.fixture()
def gbdt_bundle(tmp_path, monkeypatch):
    """Train a tiny GBDT on a synthetic DB and return the loaded bundle."""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    db_file = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_file}", future=True)
    make_synthetic_db(engine, n_races=30, n_horses_per_race=8, days_back=180, seed=42)
    engine.dispose()

    result = gbdt_train(db=db_file, train_end=None, valid_months=2, test_months=1)
    bundle = load_model_full(Path(result["model_dir"]))
    return bundle, db_file


def _load_frame(db_file: Path) -> pd.DataFrame:
    engine = make_engine(db_file)
    with session_scope(engine) as session:
        frame = build_training_frame(session)
    return frame


def test_augment_adds_three_columns(gbdt_bundle):
    bundle, db_file = gbdt_bundle
    frame = _load_frame(db_file)
    out = augment_frame_with_gbdt(frame, bundle)
    for col in GBDT_FEATURE_COLUMNS:
        assert col in out.columns


def test_augment_does_not_drop_rows(gbdt_bundle):
    bundle, db_file = gbdt_bundle
    frame = _load_frame(db_file)
    out = augment_frame_with_gbdt(frame, bundle)
    assert len(out) == len(frame)


def test_augment_preserves_horse_id_order(gbdt_bundle):
    bundle, db_file = gbdt_bundle
    frame = _load_frame(db_file)
    out = augment_frame_with_gbdt(frame, bundle)
    pd.testing.assert_series_equal(
        out["horse_id"].reset_index(drop=True),
        frame["horse_id"].reset_index(drop=True),
        check_names=False,
    )


def test_augment_fills_most_rows_with_finite_predictions(gbdt_bundle):
    bundle, db_file = gbdt_bundle
    frame = _load_frame(db_file)
    out = augment_frame_with_gbdt(frame, bundle)
    non_null = out["gbdt_win_prob"].notna().sum()
    # at least 80% should be filled (only edge-case races with <2 horses skipped)
    assert non_null / len(out) >= 0.8


def test_augment_empty_frame_returns_empty_with_columns(gbdt_bundle):
    bundle, _ = gbdt_bundle
    empty = pd.DataFrame(columns=["race_id", "horse_id"])
    out = augment_frame_with_gbdt(empty, bundle)
    assert out.empty
    for col in GBDT_FEATURE_COLUMNS:
        assert col in out.columns


def test_augment_rejects_nn_bundle(gbdt_bundle, tmp_path):
    """Passing an NN bundle should raise ValueError, not silently misbehave."""
    import torch
    from ai.nn.model import RaceTransformerModel
    from ai.registry import ModelBundle

    nn_bundle = ModelBundle(
        model_type="nn",
        model_dir=tmp_path,
        meta={},
        feature_columns=[],
        nn_model=RaceTransformerModel(horse_feat_dim=4, race_feat_dim=2, embed_dim=8, hidden_dim=16, n_heads=2),
    )
    with pytest.raises(ValueError, match="expects a GBDT bundle"):
        augment_frame_with_gbdt(pd.DataFrame({"race_id": ["r1"], "horse_id": ["h1"]}), nn_bundle)
