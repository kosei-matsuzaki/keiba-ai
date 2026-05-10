"""Tests for registry.py NN model support.

Covers save_nn_model → load_model_full round-trip and ModelBundle field
population for NN models.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from keiba_ai.ai.nn.model import RaceModel
from keiba_ai.ai.registry import ModelBundle, load_model_full, save_nn_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_small_race_model(horse_feat_dim: int = 4, race_feat_dim: int = 2) -> RaceModel:
    return RaceModel(
        horse_feat_dim=horse_feat_dim,
        race_feat_dim=race_feat_dim,
        embed_dim=8,
        hidden_dim=16,
        n_heads=2,
    )


def _save_nn_artifacts(tmp_path: Path, model: RaceModel, horse_cols: list[str], race_cols: list[str]) -> Path:
    """Save model.pt + call save_nn_model; returns model_dir."""
    model_dir = tmp_path / "test-nn"
    model_dir.mkdir()

    pt_path = model_dir / "model.pt"
    torch.save(model.state_dict(), pt_path)

    all_cols = horse_cols + race_cols
    meta_dict = {
        "model_type": "nn",
        "loss_type": "plackett_luce",
        "params": {
            "hidden_dim": 16,
            "embed_dim": 8,
            "n_heads": 2,
        },
        "metrics": {"ndcg1": 0.5, "ndcg3": 0.6},
        "feature_columns": all_cols,
        "horse_feature_cols": horse_cols,
        "race_feature_cols": race_cols,
        "train_range": None,
        "valid_range": None,
    }

    save_nn_model(pt_path, meta_dict)
    return model_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_save_nn_model_writes_meta_json(tmp_path):
    """save_nn_model writes meta.json next to model.pt."""
    horse_cols = ["feat_a", "feat_b", "feat_c", "feat_d"]
    race_cols = ["course", "distance"]
    model = _make_small_race_model(len(horse_cols), len(race_cols))
    model_dir = _save_nn_artifacts(tmp_path, model, horse_cols, race_cols)

    assert (model_dir / "meta.json").exists()
    meta = json.loads((model_dir / "meta.json").read_text())
    assert meta["model_type"] == "nn"
    assert meta["horse_feature_cols"] == horse_cols
    assert meta["race_feature_cols"] == race_cols


def test_load_model_full_nn_returns_bundle(tmp_path):
    """load_model_full on an NN dir returns ModelBundle with model_type='nn'."""
    horse_cols = ["feat_a", "feat_b", "feat_c", "feat_d"]
    race_cols = ["course", "distance"]
    model = _make_small_race_model(len(horse_cols), len(race_cols))
    model_dir = _save_nn_artifacts(tmp_path, model, horse_cols, race_cols)

    bundle = load_model_full(model_dir)

    assert isinstance(bundle, ModelBundle)
    assert bundle.model_type == "nn"


def test_load_model_full_nn_has_nn_model(tmp_path):
    """Loaded NN bundle has a non-None nn_model (RaceModel instance)."""
    horse_cols = ["feat_a", "feat_b", "feat_c", "feat_d"]
    race_cols = ["course", "distance"]
    model = _make_small_race_model(len(horse_cols), len(race_cols))
    model_dir = _save_nn_artifacts(tmp_path, model, horse_cols, race_cols)

    bundle = load_model_full(model_dir)

    assert bundle.nn_model is not None
    assert isinstance(bundle.nn_model, RaceModel)


def test_load_model_full_nn_feature_columns(tmp_path):
    """Loaded NN bundle has correct feature_columns, horse_feature_cols, race_feature_cols."""
    horse_cols = ["feat_a", "feat_b", "feat_c", "feat_d"]
    race_cols = ["course", "distance"]
    model = _make_small_race_model(len(horse_cols), len(race_cols))
    model_dir = _save_nn_artifacts(tmp_path, model, horse_cols, race_cols)

    bundle = load_model_full(model_dir)

    assert bundle.feature_columns == horse_cols + race_cols
    assert bundle.nn_horse_feature_cols == horse_cols
    assert bundle.nn_race_feature_cols == race_cols


def test_load_model_full_nn_gbdt_fields_none(tmp_path):
    """NN bundle has None for GBDT-specific fields."""
    horse_cols = ["feat_a", "feat_b", "feat_c", "feat_d"]
    race_cols = ["course", "distance"]
    model = _make_small_race_model(len(horse_cols), len(race_cols))
    model_dir = _save_nn_artifacts(tmp_path, model, horse_cols, race_cols)

    bundle = load_model_full(model_dir)

    assert bundle.lambdarank is None
    assert bundle.binary is None
    assert bundle.calibrator is None
    assert bundle.combo_calibrators is None


def test_load_model_full_nn_model_eval_mode(tmp_path):
    """Loaded NN model is in eval mode (no dropout active by default)."""
    horse_cols = ["feat_a", "feat_b", "feat_c", "feat_d"]
    race_cols = ["course", "distance"]
    model = _make_small_race_model(len(horse_cols), len(race_cols))
    model_dir = _save_nn_artifacts(tmp_path, model, horse_cols, race_cols)

    bundle = load_model_full(model_dir)

    assert not bundle.nn_model.training, "nn_model should be in eval mode"


def test_load_model_full_nn_weights_preserved(tmp_path):
    """State dict is correctly restored after save → load."""
    horse_cols = ["feat_a", "feat_b", "feat_c", "feat_d"]
    race_cols = ["course", "distance"]
    original = _make_small_race_model(len(horse_cols), len(race_cols))
    model_dir = _save_nn_artifacts(tmp_path, original, horse_cols, race_cols)

    bundle = load_model_full(model_dir)
    loaded = bundle.nn_model

    for (n1, p1), (n2, p2) in zip(
        original.state_dict().items(), loaded.state_dict().items()
    ):
        assert n1 == n2
        assert torch.allclose(p1, p2), f"Weight mismatch for {n1}"


def test_load_model_full_gbdt_not_affected(tmp_path, monkeypatch):
    """load_model_full on a GBDT dir still returns GBDT ModelBundle."""
    import os
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    from sqlalchemy import create_engine
    from keiba_ai.ai.train import train
    from tests.synthetic import make_synthetic_db

    db_file = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_file}", future=True)
    make_synthetic_db(engine, n_races=30, n_horses_per_race=8, days_back=180, seed=99)
    engine.dispose()

    result = train(db=db_file, train_end=None, valid_months=2, test_months=1)
    model_dir = Path(result["model_dir"])

    bundle = load_model_full(model_dir)

    assert bundle.model_type == "gbdt"
    assert bundle.lambdarank is not None
