"""Unit tests for ai/calibration_diagnosis.py.

Focus on the pure aggregation helpers (_per_rank_bucket / _brier_score /
_expected_calibration_error). The full CLI flow needs a trained model
and DB so it's covered by integration testing instead.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ai.calibration_diagnosis import (
    _brier_score,
    _expected_calibration_error,
    _per_rank_bucket,
)


def _scored(rows: list[dict]) -> pd.DataFrame:
    """Helper: build a scored DataFrame matching _score_all_races output."""
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _per_rank_bucket
# ---------------------------------------------------------------------------


def test_per_rank_bucket_aggregates_by_pred_rank():
    """同一 pred_rank が複数 race にわたっても集計される。"""
    rows = [
        # race 1: pred_rank 1 wins (finish=1), pred_rank 2 doesn't
        {"race_id": "R1", "pred_rank": 1, "win_prob": 0.40, "finish_position": 1.0},
        {"race_id": "R1", "pred_rank": 2, "win_prob": 0.20, "finish_position": 3.0},
        # race 2: pred_rank 1 wins, pred_rank 2 doesn't
        {"race_id": "R2", "pred_rank": 1, "win_prob": 0.30, "finish_position": 1.0},
        {"race_id": "R2", "pred_rank": 2, "win_prob": 0.15, "finish_position": 4.0},
    ]
    buckets = _per_rank_bucket(_scored(rows))
    assert len(buckets) == 2

    rank1 = next(b for b in buckets if b["rank"] == 1)
    assert rank1["n"] == 2
    assert rank1["mean_pred_prob"] == pytest.approx(0.35)
    assert rank1["actual_win_rate"] == pytest.approx(1.0)  # 2/2 wins
    assert rank1["ratio_pred_over_actual"] == pytest.approx(0.35)

    rank2 = next(b for b in buckets if b["rank"] == 2)
    assert rank2["n"] == 2
    assert rank2["actual_win_rate"] == pytest.approx(0.0)
    assert rank2["ratio_pred_over_actual"] is None  # zero actual → None


def test_per_rank_bucket_empty_input():
    assert _per_rank_bucket(pd.DataFrame()) == []


# ---------------------------------------------------------------------------
# _brier_score
# ---------------------------------------------------------------------------


def test_brier_score_perfect_prediction():
    """予測が完璧 (P=1 for actual winner, P=0 for losers) で Brier = 0。"""
    rows = [
        {"win_prob": 1.0, "finish_position": 1.0},
        {"win_prob": 0.0, "finish_position": 2.0},
        {"win_prob": 0.0, "finish_position": 3.0},
    ]
    assert _brier_score(_scored(rows)) == pytest.approx(0.0)


def test_brier_score_uniform_unseen_winner():
    """全部 P=0.5 で 1 件勝者 → Brier = ((0.5-1)^2 + 2*(0.5-0)^2)/3 = 0.25"""
    rows = [
        {"win_prob": 0.5, "finish_position": 1.0},
        {"win_prob": 0.5, "finish_position": 5.0},
        {"win_prob": 0.5, "finish_position": 6.0},
    ]
    assert _brier_score(_scored(rows)) == pytest.approx(0.25)


def test_brier_score_overconfident_wrong_prediction():
    """強気 P=0.9 で外れ (winner is 別の馬) → Brier 大きい"""
    rows = [
        {"win_prob": 0.9, "finish_position": 5.0},  # 強気だが外れ
        {"win_prob": 0.05, "finish_position": 1.0},  # 弱気だが当たり
        {"win_prob": 0.05, "finish_position": 3.0},
    ]
    expected = ((0.9 - 0) ** 2 + (0.05 - 1) ** 2 + (0.05 - 0) ** 2) / 3
    assert _brier_score(_scored(rows)) == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------------------------
# _expected_calibration_error
# ---------------------------------------------------------------------------


def test_ece_perfectly_calibrated_returns_zero():
    """予測 = 実勝率 (各 bin) → ECE = 0"""
    np.random.seed(0)
    n = 1000
    pred = np.random.uniform(0, 1, n)
    # outcomes follow predicted probability exactly (calibrated)
    outcomes = (np.random.uniform(0, 1, n) < pred).astype(int)
    df = pd.DataFrame({
        "win_prob": pred,
        "finish_position": np.where(outcomes == 1, 1.0, 5.0),
    })
    ece = _expected_calibration_error(df, n_bins=10)
    # noise tolerance for n=1000
    assert ece < 0.05, f"ECE should be near zero for calibrated preds, got {ece}"


def test_ece_systematic_overprediction():
    """予測が常に actual の 2 倍 → ECE > 0"""
    np.random.seed(42)
    n = 1000
    actual_rate = np.random.uniform(0, 0.4, n)  # actual prob 0-0.4
    pred = actual_rate * 2  # prediction is 2x actual
    outcomes = (np.random.uniform(0, 1, n) < actual_rate).astype(int)
    df = pd.DataFrame({
        "win_prob": pred,
        "finish_position": np.where(outcomes == 1, 1.0, 5.0),
    })
    ece = _expected_calibration_error(df, n_bins=10)
    assert ece > 0.05, f"ECE should be substantial for miscalibrated preds, got {ece}"


def test_ece_empty_input():
    df = pd.DataFrame({"win_prob": [], "finish_position": []})
    ece = _expected_calibration_error(df)
    assert math.isnan(ece)


# ---------------------------------------------------------------------------
# Integration smoke test (bundle-first, GBDT path)
# ---------------------------------------------------------------------------


def test_diagnose_calibration_runs_via_bundle_on_gbdt(tmp_path):
    """End-to-end: train a tiny GBDT model on synthetic data, then run
    diagnose_calibration through the bundle-aware predict_race path.

    Guards the bundle-first refactor — confirms _score_all_races dispatches
    correctly and the returned dict has the expected shape.
    """
    import os
    from pathlib import Path

    from sqlalchemy import create_engine

    import db.models  # noqa: F401
    from ai.calibration_diagnosis import diagnose_calibration
    from ai.gbm.train import train
    from tests.synthetic import make_synthetic_db

    db_file = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_file}", future=True)
    make_synthetic_db(engine, n_races=30, n_horses_per_race=10, days_back=180, seed=99)

    os.environ["KEIBA_DATA_DIR"] = str(tmp_path / "data")
    result = train(db=db_file, train_end=None, valid_months=2, test_months=1)
    model_dir = Path(result["model_dir"])

    diag = diagnose_calibration(model_path=model_dir, db=db_file)

    assert "n_races" in diag
    assert "n_entries" in diag
    assert "rank_buckets" in diag
    assert "brier_score" in diag
    assert "ece" in diag
    assert diag["n_races"] > 0
    # Brier on synthetic data should be a finite, sane number
    assert 0.0 <= diag["brier_score"] <= 1.0
    assert 0.0 <= diag["ece"] <= 1.0
    # rank_buckets non-empty when entries exist
    assert len(diag["rank_buckets"]) > 0
