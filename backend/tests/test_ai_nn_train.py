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
    """train_nn() creates model_dir with model.pt, meta.json, preprocessor.pkl."""
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
        # Combo calibrators need >=100 samples per bet type, which the tiny
        # synthetic DB cannot provide.  Disable to keep the test fast.
    )

    model_dir = Path(result["model_dir"])
    assert model_dir.exists(), "model_dir does not exist"
    assert (model_dir / "model.pt").exists(), "model.pt not found"
    assert (model_dir / "meta.json").exists(), "meta.json not found"
    assert (model_dir / "preprocessor.pkl").exists(), "preprocessor.pkl not found"

    meta = json.loads((model_dir / "meta.json").read_text())
    assert meta.get("has_preprocessor") is True


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

    # arch v2: cat metadata + new optimizer params present in meta
    assert meta.get("arch_version") == 2
    assert "cat_metadata" in meta
    assert "n_transformer_layers" in meta["params"]
    assert "weight_decay" in meta["params"]

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


# ---------------------------------------------------------------------------
# Real-odds ROI monitor + log_growth betting loss (decision-focused)
# ---------------------------------------------------------------------------


def test_train_nn_roi_metrics_present(syn_engine_small, tmp_path, monkeypatch):
    """metrics dict exposes real-odds 単勝/複勝 ROI and the chosen monitor."""
    engine, db_file = syn_engine_small
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    result = train_nn(
        db=db_file, train_end=None, valid_months=2, test_months=1,
        loss="plackett_luce", hidden_dim=16, embed_dim=8, n_heads=2,
        batch_size=4, max_epochs=2, device="cpu",
        monitor="valid_tansho_roi",
    )

    for key in ("valid_tansho_roi", "valid_fukusho_roi", "test_tansho_roi"):
        assert key in result, f"{key} missing from metrics"

    meta = json.loads((Path(result["model_dir"]) / "meta.json").read_text())
    assert meta.get("monitor") == "valid_tansho_roi"


def test_train_nn_log_growth_loss(syn_engine_small, tmp_path, monkeypatch):
    """train_nn() trains end-to-end with the log_growth betting loss."""
    engine, db_file = syn_engine_small
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    result = train_nn(
        db=db_file, train_end=None, valid_months=2, test_months=1,
        loss="log_growth", hidden_dim=16, embed_dim=8, n_heads=2,
        batch_size=4, max_epochs=2, device="cpu",
        monitor="valid_tansho_roi",
    )

    model_dir = Path(result["model_dir"])
    assert (model_dir / "model.pt").exists()
    meta = json.loads((model_dir / "meta.json").read_text())
    assert meta.get("loss_type") == "log_growth"


def test_train_nn_invalid_monitor_raises(syn_engine_small, tmp_path, monkeypatch):
    """An unknown monitor name is rejected early."""
    engine, db_file = syn_engine_small
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    with pytest.raises(ValueError, match="Unknown monitor"):
        train_nn(
            db=db_file, train_end=None, valid_months=2, test_months=1,
            loss="plackett_luce", hidden_dim=16, embed_dim=8, n_heads=2,
            batch_size=4, max_epochs=1, device="cpu",
            monitor="valid_bogus",
        )


def test_train_nn_init_from_warm_start(syn_engine_small, tmp_path, monkeypatch):
    """Two-stage: PL pretrain → log_growth fine-tune via init_from warm-start."""
    engine, db_file = syn_engine_small
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    base = train_nn(
        db=db_file, train_end=None, valid_months=2, test_months=1,
        loss="plackett_luce", hidden_dim=16, embed_dim=8, n_heads=2,
        batch_size=4, max_epochs=2, device="cpu",
    )

    tuned = train_nn(
        db=db_file, train_end=None, valid_months=2, test_months=1,
        loss="log_growth", hidden_dim=16, embed_dim=8, n_heads=2,
        batch_size=4, max_epochs=2, device="cpu",
        monitor="valid_tansho_roi",
        init_from=Path(base["model_dir"]),
    )

    meta = json.loads((Path(tuned["model_dir"]) / "meta.json").read_text())
    assert meta["params"]["init_from"] == base["model_dir"]
    assert meta["loss_type"] == "log_growth"


def test_train_nn_combo_nll_loss(syn_engine_small, tmp_path, monkeypatch):
    """combo_nll trains end-to-end (calibration objective; no payouts needed)."""
    engine, db_file = syn_engine_small
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))
    result = train_nn(
        db=db_file, train_end=None, valid_months=2, test_months=1,
        loss="combo_nll", combo_bet_type="all",
        hidden_dim=16, embed_dim=8, n_heads=2, batch_size=4, max_epochs=2,
        device="cpu", monitor="valid_ndcg3",
    )
    meta = json.loads((Path(result["model_dir"]) / "meta.json").read_text())
    assert meta["loss_type"] == "combo_nll"


def test_train_nn_multi_objective_loss(syn_engine_small, tmp_path, monkeypatch):
    """multi (log_growth + combo_weight·combo_nll) trains end-to-end."""
    engine, db_file = syn_engine_small
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))
    result = train_nn(
        db=db_file, train_end=None, valid_months=2, test_months=1,
        loss="multi", combo_weight=0.01,
        hidden_dim=16, embed_dim=8, n_heads=2, batch_size=4, max_epochs=2,
        device="cpu", monitor="valid_tansho_roi",
    )
    meta = json.loads((Path(result["model_dir"]) / "meta.json").read_text())
    assert meta["loss_type"] == "multi"
    assert meta["combo_weight"] == 0.01
