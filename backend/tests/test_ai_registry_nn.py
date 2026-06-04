"""Tests for registry.py NN model support.

Covers save_nn_model → load_model_full round-trip and ModelBundle field
population for NN models.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ai.nn.model import RaceModel
from ai.registry import ModelBundle, load_model_full, save_nn_model

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


def test_load_model_full_nn_optional_fields_none(tmp_path):
    """NN bundle has None for optional calibrator/scaler fields when absent."""
    horse_cols = ["feat_a", "feat_b", "feat_c", "feat_d"]
    race_cols = ["course", "distance"]
    model = _make_small_race_model(len(horse_cols), len(race_cols))
    model_dir = _save_nn_artifacts(tmp_path, model, horse_cols, race_cols)

    bundle = load_model_full(model_dir)

    assert bundle.combo_calibrators is None
    assert bundle.nn_calibrator is None
    assert bundle.place_calibrator is None
    assert bundle.temperature_scaler is None


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
        original.state_dict().items(), loaded.state_dict().items(), strict=True,
    ):
        assert n1 == n2
        assert torch.allclose(p1, p2), f"Weight mismatch for {n1}"


def test_load_model_full_nn_loads_preprocessor_when_present(tmp_path):
    """If preprocessor.pkl exists in the model dir, ModelBundle.nn_preprocessor is populated."""
    import pandas as pd

    from ai.nn.preprocess import NNPreprocessor

    horse_cols = ["feat_a", "feat_b", "feat_c", "feat_d"]
    race_cols = ["course", "distance"]
    model = _make_small_race_model(len(horse_cols), len(race_cols))
    model_dir = _save_nn_artifacts(tmp_path, model, horse_cols, race_cols)

    train_df = pd.DataFrame(
        {
            "feat_a": [0.0, 1.0, 2.0],
            "feat_b": [1.0, 2.0, 3.0],
            "feat_c": [0.5, 0.5, 0.5],
            "feat_d": [-1.0, 0.0, 1.0],
            "course": ["東京", "中山", "東京"],
            "distance": [1600.0, 2000.0, 1800.0],
        }
    )
    pp = NNPreprocessor.fit(train_df, horse_cols, race_cols)
    pp.save(model_dir / "preprocessor.pkl")

    bundle = load_model_full(model_dir)

    assert bundle.nn_preprocessor is not None
    assert isinstance(bundle.nn_preprocessor, NNPreprocessor)
    assert bundle.nn_preprocessor.categorical_maps == pp.categorical_maps


def test_load_model_full_nn_preprocessor_none_when_absent(tmp_path):
    """Legacy NN models without preprocessor.pkl get nn_preprocessor=None."""
    horse_cols = ["feat_a", "feat_b", "feat_c", "feat_d"]
    race_cols = ["course", "distance"]
    model = _make_small_race_model(len(horse_cols), len(race_cols))
    model_dir = _save_nn_artifacts(tmp_path, model, horse_cols, race_cols)
    # no preprocessor.pkl written

    bundle = load_model_full(model_dir)

    assert bundle.nn_preprocessor is None


def test_load_model_full_nn_arch_v2(tmp_path):
    """arch_version=2 → registry instantiates RaceTransformerModel with cat metadata."""
    import pandas as pd

    from ai.nn.model import RaceTransformerModel
    from ai.nn.preprocess import NNPreprocessor

    horse_cols = ["feat_a", "feat_b", "feat_c", "feat_d"]
    race_cols = ["course", "distance"]

    horse_cat_positions: list[int] = []
    horse_cat_cardinalities: list[int] = []
    race_cat_positions = [0]
    race_cat_cardinalities = [3]

    model = RaceTransformerModel(
        horse_feat_dim=len(horse_cols),
        race_feat_dim=len(race_cols),
        embed_dim=8,
        hidden_dim=16,
        n_heads=2,
        horse_cat_positions=horse_cat_positions,
        horse_cat_cardinalities=horse_cat_cardinalities,
        race_cat_positions=race_cat_positions,
        race_cat_cardinalities=race_cat_cardinalities,
        cat_embed_dim=4,
        n_transformer_layers=2,
    )

    model_dir = tmp_path / "test-nn-v2"
    model_dir.mkdir()
    torch.save(model.state_dict(), model_dir / "model.pt")

    meta = {
        "model_type": "nn",
        "arch_version": 2,
        "loss_type": "plackett_luce",
        "params": {
            "hidden_dim": 16,
            "embed_dim": 8,
            "n_heads": 2,
            "n_transformer_layers": 2,
            "cat_embed_dim": 4,
        },
        "metrics": {"ndcg1": 0.5},
        "feature_columns": horse_cols + race_cols,
        "horse_feature_cols": horse_cols,
        "race_feature_cols": race_cols,
        "cat_metadata": {
            "horse_cat_positions": horse_cat_positions,
            "horse_cat_cardinalities": horse_cat_cardinalities,
            "race_cat_positions": race_cat_positions,
            "race_cat_cardinalities": race_cat_cardinalities,
        },
    }
    save_nn_model(model_dir / "model.pt", meta)

    train_df = pd.DataFrame(
        {
            "feat_a": [0.0, 1.0, 2.0],
            "feat_b": [1.0, 2.0, 3.0],
            "feat_c": [0.5, 0.5, 0.5],
            "feat_d": [-1.0, 0.0, 1.0],
            "course": ["東京", "中山", "京都"],
            "distance": [1600.0, 2000.0, 1800.0],
        }
    )
    NNPreprocessor.fit(train_df, horse_cols, race_cols).save(model_dir / "preprocessor.pkl")

    bundle = load_model_full(model_dir)

    assert bundle.model_type == "nn"
    assert isinstance(bundle.nn_model, RaceTransformerModel)
