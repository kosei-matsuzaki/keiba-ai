"""Tests for ai/predict.py — single-race inference."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

import keiba_ai.db.models  # noqa: F401
from keiba_ai.ai.calibrate import ConditionalIsotonicCalibrator
from keiba_ai.ai.predict import predict_race
from keiba_ai.ai.registry import load_model, load_model_full
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


def test_predict_race_pl_model_win_prob_sums_to_one(tmp_path):
    """Plackett-Luce model predict_race: win_prob sums to 1 via softmax path."""
    db_file = tmp_path / "test_pl.db"
    engine = create_engine(f"sqlite:///{db_file}", future=True)
    make_synthetic_db(engine, n_races=25, n_horses_per_race=8, days_back=150, seed=55)

    import os
    os.environ["KEIBA_DATA_DIR"] = str(tmp_path / "data")

    result = train(db=db_file, train_end=None, valid_months=2, test_months=1,
                   loss="plackett_luce")
    model_dir = Path(result["model_dir"])

    model = load_model(model_dir)
    with Session(engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        from keiba_ai.features.builder import build_inference_frame
        frame = build_inference_frame(session, race_id)

    # No binary_model or calibrator; loss_type drives softmax path
    pred = predict_race(model, frame, loss_type="plackett_luce")

    assert not pred.empty
    assert pred["win_prob"].sum() == pytest.approx(1.0, abs=1e-5)
    # Confirm backward compat: lambdarank path with same model also works
    pred_lr = predict_race(model, frame, loss_type="lambdarank")
    assert pred_lr["win_prob"].sum() == pytest.approx(1.0, abs=1e-5)


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


def test_predict_race_with_conditional_calibrator(trained_model):
    """predict_race with ConditionalIsotonicCalibrator produces valid results.

    We fit the ConditionalIsotonicCalibrator on a batch of raw predictions that
    cover the actual score range the binary model produces, ensuring that the
    calibrated win_probs are non-zero.
    """
    engine, db_file, model_dir = trained_model
    bundle = load_model_full(model_dir)

    if bundle.binary is None:
        pytest.skip("binary model not available in this build")

    from keiba_ai.features.builder import CATEGORICAL_FEATURES, build_training_frame
    from sqlalchemy.orm import Session as _Session

    # Build a small training frame to get actual binary model raw scores.
    with _Session(engine) as session:
        full_frame = build_training_frame(session)

    from keiba_ai.ai.labels import assign_is_winner

    feature_cols_binary = list(bundle.binary.feature_name())
    X = full_frame[feature_cols_binary].copy()
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")
    raw_fit = bundle.binary.predict(X)
    target_fit = full_frame["finish_position"].map(assign_is_winner).values.astype(float)

    n_runners_col = (
        full_frame["n_runners"]
        if "n_runners" in full_frame.columns
        else full_frame.groupby("race_id")["horse_id"].transform("count")
    )
    cond_fit = pd.DataFrame({
        "surface": full_frame["surface"].values if "surface" in full_frame.columns else ["芝"] * len(full_frame),
        "n_runners": n_runners_col.values,
    })

    # min_samples_per_bin=1 so all strata get a calibrator even on small data.
    cond_cal = ConditionalIsotonicCalibrator(min_samples_per_bin=1)
    cond_cal.fit(raw_fit, target_fit, cond_fit)

    with Session(engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        frame = build_inference_frame(session, race_id)

    result = predict_race(
        bundle.lambdarank,
        frame,
        binary_model=bundle.binary,
        calibrator=cond_cal,
    )

    assert not result.empty
    assert "win_prob" in result.columns
    # win_probs must be valid probabilities
    assert (result["win_prob"] >= 0).all()
    assert (result["win_prob"] <= 1.0 + 1e-6).all()
    # After normalise=True, they should sum to ~1 (only fails when all probs are 0,
    # which should not happen since we fit on the same model's output distribution).
    assert result["win_prob"].sum() == pytest.approx(1.0, abs=1e-5)
