"""Integration tests for ai.nn.train_nn.

Uses a synthetic DB so the real keiba.db is never touched.
Runs with max_epochs=2 (or 1) to keep CI fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from ai.nn.train_nn import train_nn
from tests.synthetic import make_synthetic_db


@pytest.fixture()
def syn_engine_small(tmp_path):
    """SQLite DB with 20 races, 8 horses per race, 180-day window."""
    db_file = tmp_path / "test_nn.db"
    engine = create_engine(f"sqlite:///{db_file}", future=True)
    make_synthetic_db(engine, n_races=20, n_horses_per_race=8, days_back=180, seed=7)
    yield engine, db_file
    engine.dispose()


# ---------------------------------------------------------------------------
# Main pipeline test: plackett_luce, 2 epochs
# ---------------------------------------------------------------------------


def test_train_nn_creates_artifacts(syn_engine_small, tmp_path, monkeypatch):
    """train_nn() creates model_dir with model.pt and meta.json."""
    engine, db_file = syn_engine_small
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    result = train_nn(
        db=db_file,
        train_end=None,
        valid_months=2,
        test_months=1,
        loss="plackett_luce",
        hidden_dim=16,
        embed_dim=8,
        n_heads=2,
        batch_size=4,
        max_epochs=2,
        learning_rate=1e-3,
        device="cpu",
    )

    model_dir = Path(result["model_dir"])
    assert model_dir.exists(), "model_dir does not exist"
    assert (model_dir / "model.pt").exists(), "model.pt not found"
    assert (model_dir / "meta.json").exists(), "meta.json not found"


def test_train_nn_meta_json_structure(syn_engine_small, tmp_path, monkeypatch):
    """meta.json has the expected keys and model_type == 'nn'."""
    engine, db_file = syn_engine_small
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    result = train_nn(
        db=db_file,
        train_end=None,
        valid_months=2,
        test_months=1,
        loss="plackett_luce",
        hidden_dim=16,
        embed_dim=8,
        n_heads=2,
        batch_size=4,
        max_epochs=2,
        device="cpu",
    )

    meta = json.loads((Path(result["model_dir"]) / "meta.json").read_text())

    assert meta["model_type"] == "nn"
    assert "params" in meta
    assert "metrics" in meta
    assert "feature_columns" in meta
    assert "horse_feature_cols" in meta
    assert "race_feature_cols" in meta
    assert "loss_type" in meta
    assert meta["loss_type"] == "plackett_luce"

    # metrics must contain NDCG keys
    metrics = meta["metrics"]
    assert "ndcg1" in metrics
    assert "ndcg3" in metrics


def test_train_nn_return_dict_keys(syn_engine_small, tmp_path, monkeypatch):
    """Return dict from train_nn() has the expected keys."""
    engine, db_file = syn_engine_small
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    result = train_nn(
        db=db_file,
        train_end=None,
        valid_months=2,
        test_months=1,
        loss="plackett_luce",
        hidden_dim=16,
        embed_dim=8,
        n_heads=2,
        batch_size=4,
        max_epochs=2,
        device="cpu",
    )

    for key in ("model_dir", "valid_loss", "test_loss", "ndcg1", "ndcg3"):
        assert key in result, f"Missing key: {key}"

    assert Path(result["model_dir"]).exists()


# ---------------------------------------------------------------------------
# Alternate loss functions: listmle, time_margin
# ---------------------------------------------------------------------------


def test_train_nn_listmle(syn_engine_small, tmp_path, monkeypatch):
    """train_nn completes with loss=listmle."""
    engine, db_file = syn_engine_small
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    result = train_nn(
        db=db_file,
        train_end=None,
        valid_months=2,
        test_months=1,
        loss="listmle",
        hidden_dim=16,
        embed_dim=8,
        n_heads=2,
        batch_size=4,
        max_epochs=1,
        device="cpu",
    )

    assert Path(result["model_dir"]).exists()
    meta = json.loads((Path(result["model_dir"]) / "meta.json").read_text())
    assert meta["loss_type"] == "listmle"


def test_train_nn_time_margin(syn_engine_small, tmp_path, monkeypatch):
    """train_nn completes with loss=time_margin."""
    engine, db_file = syn_engine_small
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    result = train_nn(
        db=db_file,
        train_end=None,
        valid_months=2,
        test_months=1,
        loss="time_margin",
        hidden_dim=16,
        embed_dim=8,
        n_heads=2,
        batch_size=4,
        max_epochs=1,
        device="cpu",
    )

    assert Path(result["model_dir"]).exists()
    meta = json.loads((Path(result["model_dir"]) / "meta.json").read_text())
    assert meta["loss_type"] == "time_margin"


# ---------------------------------------------------------------------------
# Small batch size
# ---------------------------------------------------------------------------


def test_train_nn_small_batch(syn_engine_small, tmp_path, monkeypatch):
    """train_nn runs with batch_size=2 (edge case)."""
    engine, db_file = syn_engine_small
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    result = train_nn(
        db=db_file,
        train_end=None,
        valid_months=2,
        test_months=1,
        loss="plackett_luce",
        hidden_dim=16,
        embed_dim=8,
        n_heads=2,
        batch_size=2,
        max_epochs=2,
        device="cpu",
    )

    assert Path(result["model_dir"]).exists()
    assert (Path(result["model_dir"]) / "model.pt").exists()


# ---------------------------------------------------------------------------
# Temperature scaler integration tests
# ---------------------------------------------------------------------------


def test_train_nn_fits_temperature_scaler(syn_engine_small, tmp_path, monkeypatch):
    """train_nn() with default fit_temperature=True saves temperature_scaler.pkl
    and sets has_temperature_scaler=True in meta.json when valid set is non-empty.
    """
    engine, db_file = syn_engine_small
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    result = train_nn(
        db=db_file,
        train_end=None,
        valid_months=2,
        test_months=1,
        loss="plackett_luce",
        hidden_dim=16,
        embed_dim=8,
        n_heads=2,
        batch_size=4,
        max_epochs=2,
        device="cpu",
        fit_temperature=True,
    )

    model_dir = Path(result["model_dir"])
    assert (model_dir / "temperature_scaler.pkl").exists(), (
        "temperature_scaler.pkl was not saved"
    )

    meta = json.loads((model_dir / "meta.json").read_text())
    assert meta.get("has_temperature_scaler") is True, (
        f"has_temperature_scaler should be True, got: {meta.get('has_temperature_scaler')}"
    )

    # Verify the pickle loads and has expected attributes
    import pickle
    with (model_dir / "temperature_scaler.pkl").open("rb") as f:
        scaler = pickle.load(f)
    assert hasattr(scaler, "T_win"), "TemperatureScaler missing T_win"
    assert hasattr(scaler, "T_place"), "TemperatureScaler missing T_place"
    assert isinstance(scaler.T_win, float)
    assert isinstance(scaler.T_place, float)


def test_train_nn_no_fit_temperature(syn_engine_small, tmp_path, monkeypatch):
    """train_nn() with fit_temperature=False skips temperature scaler fitting."""
    engine, db_file = syn_engine_small
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    result = train_nn(
        db=db_file,
        train_end=None,
        valid_months=2,
        test_months=1,
        loss="plackett_luce",
        hidden_dim=16,
        embed_dim=8,
        n_heads=2,
        batch_size=4,
        max_epochs=2,
        device="cpu",
        fit_temperature=False,
    )

    model_dir = Path(result["model_dir"])
    assert not (model_dir / "temperature_scaler.pkl").exists(), (
        "temperature_scaler.pkl should not exist when fit_temperature=False"
    )

    meta = json.loads((model_dir / "meta.json").read_text())
    assert meta.get("has_temperature_scaler") is False, (
        f"has_temperature_scaler should be False, got: {meta.get('has_temperature_scaler')}"
    )
