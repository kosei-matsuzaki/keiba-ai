"""Integration test: train pipeline on synthetic data.

Verifies:
- model.txt and meta.json are created
- model_runs table gets a new row
- Training completes without errors on small synthetic data
- Recency-weighted training (recency_lambda > 0) works for both heads
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

import keiba_ai.db.models  # noqa: F401
from keiba_ai.ai.train import _compute_recency_weights, train
from keiba_ai.db.models.model_run import ModelRun
from tests.synthetic import make_synthetic_db


@pytest.fixture()
def syn_engine(tmp_path):
    """In-file SQLite DB in tmp_path (LightGBM needs real files for some ops)."""
    db_file = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_file}", future=True)
    # Need n_races large enough to get non-empty train/valid/test splits
    make_synthetic_db(engine, n_races=30, n_horses_per_race=10, days_back=180, seed=42)
    yield engine, db_file
    engine.dispose()


def test_train_creates_model_files(syn_engine, tmp_path, monkeypatch):
    engine, db_file = syn_engine

    # Override data_dir so models go into tmp_path
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    result = train(
        db=db_file,
        train_end=None,
        valid_months=2,  # small to ensure all splits have data
        test_months=1,
    )

    model_dir = Path(result["model_dir"])
    assert (model_dir / "model.txt").exists(), "model.txt not found"
    assert (model_dir / "meta.json").exists(), "meta.json not found"

    meta = json.loads((model_dir / "meta.json").read_text())
    assert "params" in meta
    assert "metrics" in meta
    assert "feature_columns" in meta


def test_train_inserts_model_run(syn_engine, tmp_path, monkeypatch):
    engine, db_file = syn_engine
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    with Session(engine) as session:
        count_before = session.scalar(select(func.count()).select_from(ModelRun))

    train(db=db_file, train_end=None, valid_months=2, test_months=1)

    with Session(engine) as session:
        count_after = session.scalar(select(func.count()).select_from(ModelRun))

    assert count_after == count_before + 1


def test_train_metrics_keys_present(syn_engine, tmp_path, monkeypatch):
    engine, db_file = syn_engine
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    result = train(db=db_file, train_end=None, valid_months=2, test_months=1)

    assert "model_dir" in result
    # Metrics keys may be nan for small data but must be present
    for key in ("valid_ndcg1", "valid_ndcg3", "test_ndcg1", "test_ndcg3"):
        assert key in result


class TestComputeRecencyWeights:
    """Unit tests for _compute_recency_weights."""

    def _make_df(self, dates: list[str]) -> pd.DataFrame:
        return pd.DataFrame({"date": dates})

    def test_lambda_zero_returns_none(self):
        df = self._make_df(["2024-01-01", "2024-06-01", "2025-01-01"])
        assert _compute_recency_weights(df, 0.0) is None

    def test_lambda_negative_returns_none(self):
        df = self._make_df(["2024-01-01", "2024-06-01"])
        assert _compute_recency_weights(df, -1.0) is None

    def test_most_recent_row_has_weight_one(self):
        """The row with the latest date should have weight = exp(0) = 1.0."""
        df = self._make_df(["2023-01-01", "2024-01-01", "2025-01-01"])
        weights = _compute_recency_weights(df, recency_lambda=1.0)
        assert weights is not None
        assert math.isclose(float(weights[2]), 1.0, rel_tol=1e-5)

    def test_older_rows_have_lower_weight(self):
        """Weights must decrease monotonically from newest to oldest."""
        df = self._make_df(["2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"])
        weights = _compute_recency_weights(df, recency_lambda=0.5)
        assert weights is not None
        # Sorted newest (idx=3) → oldest (idx=0); weight must be strictly decreasing
        assert weights[3] > weights[2] > weights[1] > weights[0]

    def test_formula_correctness(self):
        """Verify exp(-λ × age_years) numerically for a known pair."""
        # Two rows exactly 365.25 days apart
        dates = ["2024-01-01", "2025-01-01"]
        df = pd.DataFrame({"date": pd.to_datetime(dates)})
        # Manually compute age_years for the older row
        delta_days = (pd.Timestamp("2025-01-01") - pd.Timestamp("2024-01-01")).days
        expected_age = delta_days / 365.25
        lam = 2.0
        expected_weight = math.exp(-lam * expected_age)

        weights = _compute_recency_weights(df, recency_lambda=lam)
        assert weights is not None
        assert math.isclose(float(weights[0]), expected_weight, rel_tol=1e-5)

    def test_all_same_date_returns_ones(self):
        """When all rows share the same date, age_years = 0 and weight = 1 for all."""
        df = self._make_df(["2025-03-01", "2025-03-01", "2025-03-01"])
        weights = _compute_recency_weights(df, recency_lambda=1.0)
        assert weights is not None
        np.testing.assert_allclose(weights, 1.0, rtol=1e-5)


def test_train_with_zero_valid_does_not_leak_test(syn_engine, tmp_path, monkeypatch):
    """valid_months=0 で valid が空でも、train_df に test 行が混ざってはいけない。

    回帰防止: 旧実装では `train_df.empty or valid_df.empty` の fallback が
    train_df = frame.copy() で全行を train に化けさせ、test を完全リーク
    して NDCG=1.0 を出していた。修正後は valid 空のときに train_df を
    保持し、test は別物のままであることを test_ndcg < 1.0 で確認する。

    Note: synthetic データは特徴量とラベルの相関が緩いため、リークさえ
    なければ NDCG@1 は <0.99 に収まる。1.0 ぴったりは過学習＋リークの
    シグネチャ。
    """
    engine, db_file = syn_engine
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    # Use a recent train_end so test_df has enough races to be meaningful
    # (synthetic data spans 180 days back from "today")
    result = train(
        db=db_file,
        train_end=None,
        valid_months=0,   # ← intentionally empty valid
        test_months=2,    # ← test must remain separate
    )

    # If the leak fallback fired, test_ndcg1 would round to 1.0.
    # With the fix, test must be evaluated on rows the model never saw.
    import math
    # Guard: if test_ndcg1 is NaN the test_df was empty and the leak check is
    # vacuous — fail loudly so future synthetic changes don't silently weaken
    # this regression.
    assert not math.isnan(result["test_ndcg1"]), (
        "test_ndcg1 is NaN — test split is empty, leak regression cannot be verified"
    )
    assert result["test_ndcg1"] < 0.99, (
        f"Suspicious test_ndcg1={result['test_ndcg1']:.4f} — likely test leak"
    )


def test_train_recency_lambda_completes(syn_engine, tmp_path, monkeypatch):
    """train() with recency_lambda > 0 completes without error on synthetic data."""
    engine, db_file = syn_engine
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    result = train(
        db=db_file,
        train_end=None,
        valid_months=2,
        test_months=1,
        recency_lambda=1.0,
    )

    model_dir = Path(result["model_dir"])
    assert (model_dir / "model.txt").exists()
    assert (model_dir / "meta.json").exists()
    for key in ("valid_ndcg1", "valid_ndcg3", "test_ndcg1", "test_ndcg3"):
        assert key in result


def test_train_recency_lambda_persisted_in_meta(syn_engine, tmp_path, monkeypatch):
    """recency_lambda is stored in meta.json → params so it can be audited later."""
    engine, db_file = syn_engine
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    lam = 0.75
    result = train(
        db=db_file,
        train_end=None,
        valid_months=2,
        test_months=1,
        recency_lambda=lam,
    )

    meta = json.loads((Path(result["model_dir"]) / "meta.json").read_text())
    assert "recency_lambda" in meta["params"], "recency_lambda must be saved in meta.json params"
    assert math.isclose(meta["params"]["recency_lambda"], lam)


def test_train_recency_lambda_zero_identical_to_default(syn_engine, tmp_path, monkeypatch):
    """recency_lambda=0.0 must produce the same model files as omitting the argument.

    We cannot guarantee bit-identical model weights because LightGBM has
    internal non-determinism, but we verify the pipeline reaches completion and
    meta.json records recency_lambda=0.0 (backward-compatible path).
    """
    engine, db_file = syn_engine
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    result = train(
        db=db_file,
        train_end=None,
        valid_months=2,
        test_months=1,
        recency_lambda=0.0,
    )

    meta = json.loads((Path(result["model_dir"]) / "meta.json").read_text())
    assert math.isclose(meta["params"]["recency_lambda"], 0.0)
    assert (Path(result["model_dir"]) / "model.txt").exists()
