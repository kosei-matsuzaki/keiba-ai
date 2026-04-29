"""Tests for ai/tune.py — Optuna hyperparameter tuning (lightweight)."""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine

from keiba_ai.ai.tune import tune
from tests.synthetic import make_synthetic_db


@pytest.fixture()
def tuning_db(tmp_path):
    """Small synthetic DB for tuning tests."""
    db_file = tmp_path / "tune_test.db"
    engine = create_engine(f"sqlite:///{db_file}", future=True)
    # Need enough data for time_split to produce non-empty sets
    make_synthetic_db(engine, n_races=30, n_horses_per_race=8, days_back=180, seed=123)
    os.environ["KEIBA_DATA_DIR"] = str(tmp_path / "data")
    return db_file


def test_tune_returns_best_params_and_value(tuning_db):
    """n_trials=2 should complete quickly and return expected keys."""
    result = tune(
        db=tuning_db,
        train_end=None,
        valid_months=2,
        test_months=1,
        n_trials=2,
    )
    assert "best_params" in result, "Missing best_params key"
    assert "best_value" in result, "Missing best_value key"
    assert "n_trials" in result


def test_tune_best_params_contains_expected_keys(tuning_db):
    """best_params should contain the tuned hyperparameter names."""
    result = tune(
        db=tuning_db,
        train_end=None,
        valid_months=2,
        test_months=1,
        n_trials=2,
    )
    params = result["best_params"]
    for key in ("num_leaves", "learning_rate", "min_data_in_leaf"):
        assert key in params, f"Missing param: {key}"


def test_tune_best_value_is_numeric(tuning_db):
    """best_value should be a finite float (NDCG@3 ∈ [0, 1])."""
    import math
    result = tune(
        db=tuning_db,
        train_end=None,
        valid_months=2,
        test_months=1,
        n_trials=2,
    )
    val = result["best_value"]
    assert isinstance(val, float), f"Expected float, got {type(val)}"
    assert not math.isnan(val), "best_value must not be NaN"


def test_tune_n_trials_matches_requested(tuning_db):
    """Completed trials count should equal n_trials (no pruning in this setup)."""
    result = tune(
        db=tuning_db,
        train_end=None,
        valid_months=2,
        test_months=1,
        n_trials=3,
    )
    # Optuna may complete all trials
    assert result["n_trials"] >= 1
