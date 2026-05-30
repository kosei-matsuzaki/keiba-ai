"""Tests for ai/nn/fit_calibrator.py — post-hoc NN calibrator fitting.

Uses a tiny synthetic NN training run so the end-to-end fit_and_save flow
can be exercised without external data dependencies. torch is required;
test is auto-skipped when unavailable.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from sqlalchemy import create_engine  # noqa: E402

import db.models  # noqa: F401, E402
from tests.synthetic import make_synthetic_db  # noqa: E402


def _train_minimal_nn(tmp_path: Path):
    """Train a tiny NN model on synthetic data and return (db, model_dir)."""
    from ai.nn.train_nn import train_nn  # noqa: PLC0415

    db_file = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_file}", future=True)
    make_synthetic_db(engine, n_races=80, n_horses_per_race=10, days_back=300, seed=7)

    os.environ["KEIBA_DATA_DIR"] = str(tmp_path / "data")
    result = train_nn(
        db=db_file,
        train_end=None,
        valid_months=2,
        test_months=1,
        max_epochs=2,  # cheap
        batch_size=8,
        hidden_dim=16,
        embed_dim=16,
        n_heads=2,
        n_transformer_layers=1,
    )
    model_dir = Path(result["model_dir"])
    return db_file, model_dir


def test_fit_and_save_writes_pickle(tmp_path):
    """`fit_and_save` writes nn_calibrator.pkl into the model directory."""
    from ai.nn.fit_calibrator import fit_and_save

    db_file, model_dir = _train_minimal_nn(tmp_path)

    result = fit_and_save(model_path=model_dir, db=db_file)
    target = model_dir / "nn_calibrator.pkl"
    assert target.exists(), "nn_calibrator.pkl was not created"
    assert result["saved_to"] == str(target)
    assert "before" in result and "after" in result
    assert result["n_entries"] > 0


def test_fit_and_save_refuses_overwrite_without_force(tmp_path):
    from ai.nn.fit_calibrator import fit_and_save

    db_file, model_dir = _train_minimal_nn(tmp_path)
    fit_and_save(model_path=model_dir, db=db_file)

    with pytest.raises(FileExistsError):
        fit_and_save(model_path=model_dir, db=db_file)


def test_fit_and_save_force_overwrites(tmp_path):
    from ai.nn.fit_calibrator import fit_and_save

    db_file, model_dir = _train_minimal_nn(tmp_path)
    fit_and_save(model_path=model_dir, db=db_file)
    # Should not raise with --force
    result = fit_and_save(model_path=model_dir, db=db_file, force=True)
    assert (model_dir / "nn_calibrator.pkl").exists()
    assert result["saved_to"].endswith("nn_calibrator.pkl")


def test_calibrator_loaded_into_bundle(tmp_path):
    """After fit_and_save, load_model_full picks up the calibrator."""
    from ai.nn.fit_calibrator import fit_and_save
    from ai.registry import load_model_full

    db_file, model_dir = _train_minimal_nn(tmp_path)
    fit_and_save(model_path=model_dir, db=db_file)

    bundle = load_model_full(model_dir)
    assert bundle.model_type == "nn"
    assert bundle.nn_calibrator is not None
    assert bundle.nn_calibrator.fitted


def test_place_calibrator_loaded_into_bundle(tmp_path):
    """fit_and_save も place_calibrator.pkl を作り、load_model_full が拾う。"""
    from ai.nn.fit_calibrator import fit_and_save
    from ai.registry import load_model_full

    db_file, model_dir = _train_minimal_nn(tmp_path)
    result = fit_and_save(model_path=model_dir, db=db_file)

    assert (model_dir / "place_calibrator.pkl").exists()
    assert "place" in result and "before" in result["place"] and "after" in result["place"]

    bundle = load_model_full(model_dir)
    assert bundle.place_calibrator is not None
    assert bundle.place_calibrator.fitted


def test_predict_race_applies_calibrator(tmp_path):
    """predict_race output sums to ~1 per race after calibration."""
    import numpy as np

    from ai.nn.fit_calibrator import fit_and_save
    from ai.predict import predict_race
    from ai.registry import load_model_full
    from db.session import make_engine, session_scope
    from features.builder import build_training_frame

    db_file, model_dir = _train_minimal_nn(tmp_path)
    fit_and_save(model_path=model_dir, db=db_file)

    engine = make_engine(db_file)
    with session_scope(engine) as session:
        frame = build_training_frame(session)

    bundle = load_model_full(model_dir)
    # Take any race and confirm win_prob is a valid distribution
    rid = frame["race_id"].iloc[0]
    race_frame = frame[frame["race_id"] == rid]
    preds = predict_race(bundle, race_frame)

    assert "win_prob" in preds.columns
    assert (preds["win_prob"] >= 0).all()
    assert (preds["win_prob"] <= 1).all()
    # Re-normalised in _predict_race_nn → sum is 1 within float tolerance.
    np.testing.assert_allclose(preds["win_prob"].sum(), 1.0, atol=1e-6)


def test_fit_and_save_gbdt_fits_place_only(tmp_path):
    """GBDT モデルは place head のみ fit する (win head は binary head + calibrator.pkl
    で既に較正済みなので二重較正を避ける)。"""
    from ai.gbm.train import train
    from ai.nn.fit_calibrator import fit_and_save

    db_file = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_file}", future=True)
    make_synthetic_db(engine, n_races=30, n_horses_per_race=10, days_back=180, seed=11)
    os.environ["KEIBA_DATA_DIR"] = str(tmp_path / "data")
    result = train(db=db_file, train_end=None, valid_months=2, test_months=1)
    gbdt_dir = Path(result["model_dir"])

    out = fit_and_save(model_path=gbdt_dir, db=db_file)
    assert out["model_type"] == "gbdt"
    assert out["win"] is None  # GBDT skips win calibrator
    assert out["place"] is not None
    assert (gbdt_dir / "place_calibrator.pkl").exists()


def test_module_importable():
    """Smoke: module imports without side effects."""
    mod = importlib.import_module("ai.nn.fit_calibrator")
    assert hasattr(mod, "fit_and_save")
    assert hasattr(mod, "_cli")
