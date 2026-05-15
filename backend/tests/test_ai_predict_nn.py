"""Tests for NN inference via predict_race and predict_race_with_combinations.

Uses small synthetic RaceModel instances to avoid long training times.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ai.nn.model import RaceModel
from ai.predict import predict_race, predict_race_with_combinations
from ai.registry import ModelBundle, load_model_full, save_nn_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HORSE_COLS = ["feat_a", "feat_b", "feat_c", "feat_d"]
_RACE_COLS = ["course", "distance"]


def _make_bundle(tmp_path: Path, n_horses_in_model: int = 4) -> ModelBundle:
    """Build a small RaceModel, save it, and load it as a ModelBundle."""
    model = RaceModel(
        horse_feat_dim=len(_HORSE_COLS),
        race_feat_dim=len(_RACE_COLS),
        embed_dim=8,
        hidden_dim=16,
        n_heads=2,
    )
    model_dir = tmp_path / "nn-model"
    model_dir.mkdir()
    pt_path = model_dir / "model.pt"
    torch.save(model.state_dict(), pt_path)

    meta_dict = {
        "model_type": "nn",
        "loss_type": "plackett_luce",
        "params": {"hidden_dim": 16, "embed_dim": 8, "n_heads": 2},
        "metrics": {"ndcg1": 0.5},
        "feature_columns": _HORSE_COLS + _RACE_COLS,
        "horse_feature_cols": _HORSE_COLS,
        "race_feature_cols": _RACE_COLS,
        "train_range": None,
        "valid_range": None,
    }
    save_nn_model(pt_path, meta_dict)
    return load_model_full(model_dir)


def _make_race_frame(n_horses: int = 6) -> pd.DataFrame:
    """Build a minimal inference DataFrame for one race."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_horses):
        row = {
            "horse_id": f"horse_{i:02d}",
            "post_position": i + 1,
        }
        for col in _HORSE_COLS:
            row[col] = float(rng.standard_normal())
        # Race-level features (constant within the race)
        row["course"] = 0.0  # already numeric after _encode_categoricals
        row["distance"] = 1600.0
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# predict_race — NN path
# ---------------------------------------------------------------------------


def test_predict_race_bundle_nn_returns_dataframe(tmp_path):
    bundle = _make_bundle(tmp_path)
    frame = _make_race_frame(6)
    result = predict_race(bundle, frame)
    assert isinstance(result, pd.DataFrame)


def test_predict_race_bundle_nn_columns(tmp_path):
    bundle = _make_bundle(tmp_path)
    frame = _make_race_frame(6)
    result = predict_race(bundle, frame)
    assert set(result.columns) >= {"horse_id", "score", "win_prob", "place_prob"}


def test_predict_race_bundle_nn_win_prob_sums_to_one(tmp_path):
    bundle = _make_bundle(tmp_path)
    frame = _make_race_frame(6)
    result = predict_race(bundle, frame)
    assert abs(result["win_prob"].sum() - 1.0) < 1e-4, (
        f"win_prob sum = {result['win_prob'].sum()}"
    )


def test_predict_race_bundle_nn_score_descending(tmp_path):
    bundle = _make_bundle(tmp_path)
    frame = _make_race_frame(6)
    result = predict_race(bundle, frame)
    scores = result["score"].values
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), (
        "DataFrame is not sorted by score descending"
    )


def test_predict_race_bundle_nn_row_count(tmp_path):
    n_horses = 8
    bundle = _make_bundle(tmp_path)
    frame = _make_race_frame(n_horses)
    result = predict_race(bundle, frame)
    assert len(result) == n_horses


def test_predict_race_bundle_nn_place_prob_in_range(tmp_path):
    bundle = _make_bundle(tmp_path)
    frame = _make_race_frame(6)
    result = predict_race(bundle, frame)
    assert (result["place_prob"] >= 0).all()
    assert (result["place_prob"] <= 1).all()


def test_predict_race_bundle_nn_empty_frame(tmp_path):
    bundle = _make_bundle(tmp_path)
    empty = pd.DataFrame(columns=["horse_id", "post_position"] + _HORSE_COLS + _RACE_COLS)
    result = predict_race(bundle, empty)
    assert result.empty
    assert set(result.columns) >= {"horse_id", "score", "win_prob", "place_prob"}


# ---------------------------------------------------------------------------
# predict_race_with_combinations — NN path
# ---------------------------------------------------------------------------


def test_predict_race_with_combinations_bundle_nn_returns_dict(tmp_path):
    bundle = _make_bundle(tmp_path)
    frame = _make_race_frame(6)
    result = predict_race_with_combinations(bundle, frame)
    assert isinstance(result, dict)


def test_predict_race_with_combinations_bundle_nn_bet_types(tmp_path):
    bundle = _make_bundle(tmp_path)
    frame = _make_race_frame(6)
    result = predict_race_with_combinations(bundle, frame)
    expected_keys = {"単勝", "複勝", "馬連", "ワイド", "馬単", "三連複", "三連単"}
    assert set(result.keys()) == expected_keys


def test_predict_race_with_combinations_bundle_nn_tansho_count(tmp_path):
    n_horses = 6
    bundle = _make_bundle(tmp_path)
    frame = _make_race_frame(n_horses)
    result = predict_race_with_combinations(bundle, frame)
    assert len(result["単勝"]) == n_horses


def test_predict_race_with_combinations_bundle_nn_combination_prediction_fields(tmp_path):
    from ai.types import CombinationPrediction
    bundle = _make_bundle(tmp_path)
    frame = _make_race_frame(4)
    result = predict_race_with_combinations(bundle, frame)
    cp = result["単勝"][0]
    assert isinstance(cp, CombinationPrediction)
    assert hasattr(cp, "combo")
    assert hasattr(cp, "prob")
    assert hasattr(cp, "ev")


# ---------------------------------------------------------------------------
# Bundle dispatch: GBDT path still works after registry refactor
# ---------------------------------------------------------------------------


def test_predict_race_bundle_gbdt_path(tmp_path, monkeypatch):
    """predict_race correctly routes GBDT models to the GBDT path."""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    from sqlalchemy import create_engine

    from ai.gbm.train import train
    from tests.synthetic import make_synthetic_db

    db_file = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_file}", future=True)
    make_synthetic_db(engine, n_races=30, n_horses_per_race=8, days_back=180, seed=11)
    engine.dispose()

    result = train(db=db_file, train_end=None, valid_months=2, test_months=1)
    model_dir = Path(result["model_dir"])
    bundle = load_model_full(model_dir)

    assert bundle.model_type == "gbdt"
    assert bundle.lambdarank is not None

    import lightgbm as lgb

    from db.session import make_engine, session_scope
    from features.builder import FEATURE_COLUMNS, build_training_frame

    engine2 = make_engine(db_file)
    with session_scope(engine2) as session:
        frame = build_training_frame(session)

    # Take the first race
    first_race_id = frame["race_id"].iloc[0]
    race_frame = frame[frame["race_id"] == first_race_id].copy()

    out = predict_race(bundle, race_frame)
    assert set(out.columns) >= {"horse_id", "score", "win_prob", "place_prob"}
    assert abs(out["win_prob"].sum() - 1.0) < 1e-4
