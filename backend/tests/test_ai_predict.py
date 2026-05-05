"""Tests for ai/predict.py — single-race inference."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

import keiba_ai.db.models  # noqa: F401
from keiba_ai.ai.predict import predict_race
from keiba_ai.ai.registry import load_model
from keiba_ai.ai.train import train
from keiba_ai.db.models.race import Race
from keiba_ai.features.builder import build_inference_frame
from tests.synthetic import make_synthetic_db


@pytest.fixture()
def trained_model(tmp_path):
    """Train a small model and return (engine, db_file, model_dir)."""
    db_file = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_file}", future=True)
    make_synthetic_db(engine, n_races=30, n_horses_per_race=10, days_back=180, seed=7)

    import os
    os.environ["KEIBA_DATA_DIR"] = str(tmp_path / "data")

    result = train(db=db_file, train_end=None, valid_months=2, test_months=1)
    model_dir = Path(result["model_dir"])
    return engine, db_file, model_dir


def test_win_prob_sums_to_one(trained_model, tmp_path):
    engine, db_file, model_dir = trained_model
    model = load_model(model_dir)

    with Session(engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        frame = build_inference_frame(session, race_id)

    result = predict_race(model, frame)

    assert not result.empty
    assert "win_prob" in result.columns
    assert result["win_prob"].sum() == pytest.approx(1.0, abs=1e-5)


def test_predict_race_columns(trained_model):
    engine, db_file, model_dir = trained_model
    model = load_model(model_dir)

    with Session(engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        frame = build_inference_frame(session, race_id)

    result = predict_race(model, frame)

    for col in ("horse_id", "score", "win_prob", "place_prob"):
        assert col in result.columns


def test_predict_sorted_by_score(trained_model):
    engine, db_file, model_dir = trained_model
    model = load_model(model_dir)

    with Session(engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        frame = build_inference_frame(session, race_id)

    result = predict_race(model, frame)

    scores = result["score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_predict_place_prob_in_range(trained_model):
    engine, db_file, model_dir = trained_model
    model = load_model(model_dir)

    with Session(engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        frame = build_inference_frame(session, race_id)

    result = predict_race(model, frame)
    assert (result["place_prob"] >= 0).all()
    assert (result["place_prob"] <= 1.0 + 1e-6).all()


def test_predict_race_performance(trained_model, monkeypatch):
    """predict_race with plackett_luce must complete within 50 ms per race."""
    engine, db_file, model_dir = trained_model
    model = load_model(model_dir)

    monkeypatch.setenv("KEIBA_PLACE_PROB_METHOD", "plackett_luce")

    with Session(engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        frame = build_inference_frame(session, race_id)

    # Warm up LightGBM predict (first call may load BLAS)
    predict_race(model, frame)

    start = time.perf_counter()
    predict_race(model, frame)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < 50, f"predict_race took {elapsed_ms:.1f} ms, expected < 50 ms"
