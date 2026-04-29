"""Tests for predict_race_with_shap in ai/predict.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

import keiba_ai.db.models  # noqa: F401
from keiba_ai.ai.predict import predict_race_with_shap
from keiba_ai.ai.registry import load_model
from keiba_ai.ai.train import train
from keiba_ai.db.models.race import Race
from keiba_ai.features.builder import FEATURE_COLUMNS, build_inference_frame
from tests.synthetic import make_synthetic_db


@pytest.fixture()
def trained_scenario(tmp_path):
    """Train a small model and return (engine, db_file, model_dir)."""
    import os
    db_file = tmp_path / "shap_test.db"
    engine = create_engine(f"sqlite:///{db_file}", future=True)
    make_synthetic_db(engine, n_races=30, n_horses_per_race=10, days_back=180, seed=42)

    os.environ["KEIBA_DATA_DIR"] = str(tmp_path / "data")
    result = train(db=db_file, train_end=None, valid_months=2, test_months=1)
    model_dir = Path(result["model_dir"])
    return engine, db_file, model_dir


def test_predict_race_with_shap_has_top_features_column(trained_scenario):
    engine, db_file, model_dir = trained_scenario
    model = load_model(model_dir)

    with Session(engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        frame = build_inference_frame(session, race_id)

    result = predict_race_with_shap(model, frame)

    assert "top_features" in result.columns


def test_top_features_are_valid_feature_names(trained_scenario):
    engine, db_file, model_dir = trained_scenario
    model = load_model(model_dir)

    with Session(engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        frame = build_inference_frame(session, race_id)

    result = predict_race_with_shap(model, frame, top_n=3)

    for _, row in result.iterrows():
        features = row["top_features"]
        assert isinstance(features, list), f"Expected list, got {type(features)}"
        for f in features:
            assert isinstance(f, str), f"Feature name must be str, got {type(f)}"
            assert f in FEATURE_COLUMNS, f"Unknown feature: {f!r}"


def test_top_features_length_matches_top_n(trained_scenario):
    engine, db_file, model_dir = trained_scenario
    model = load_model(model_dir)

    with Session(engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        frame = build_inference_frame(session, race_id)

    for top_n in (1, 3, 5):
        result = predict_race_with_shap(model, frame, top_n=top_n)
        for _, row in result.iterrows():
            assert len(row["top_features"]) == top_n, (
                f"Expected {top_n} features, got {len(row['top_features'])}"
            )


def test_predict_race_with_shap_preserves_score_order(trained_scenario):
    engine, db_file, model_dir = trained_scenario
    model = load_model(model_dir)

    with Session(engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        frame = build_inference_frame(session, race_id)

    result = predict_race_with_shap(model, frame)

    scores = result["score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_predict_race_with_shap_empty_frame(trained_scenario):
    """Empty frame should return empty DataFrame with top_features column."""
    import pandas as pd
    _, db_file, model_dir = trained_scenario
    model = load_model(model_dir)

    empty_frame = pd.DataFrame(columns=["horse_id"] + FEATURE_COLUMNS)
    result = predict_race_with_shap(model, empty_frame)

    assert result.empty
    assert "top_features" in result.columns
