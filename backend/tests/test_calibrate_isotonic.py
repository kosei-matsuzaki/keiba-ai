"""Tests for ai/calibrate.py:IsotonicCalibrator."""

from __future__ import annotations

import numpy as np
import pytest

from keiba_ai.ai.calibrate import IsotonicCalibrator


def test_fit_predict_monotonic():
    """isotonic は monotonic non-decreasing なので、入力が増えれば出力も増える。"""
    raw = np.linspace(0.05, 0.95, 100)
    # outcomes follow raw probability (well-calibrated synthetic data)
    rng = np.random.default_rng(0)
    outcomes = (rng.uniform(0, 1, 100) < raw).astype(np.float32)

    cal = IsotonicCalibrator()
    cal.fit(raw, outcomes)
    out = cal.predict(np.array([0.1, 0.3, 0.5, 0.7, 0.9]), normalise=False)
    # 単調非減少
    diffs = np.diff(out)
    assert np.all(diffs >= -1e-9), f"non-monotonic output: {out}"


def test_normalise_true_sums_to_one():
    """normalise=True (デフォルト) の出力はレース内合計 = 1。"""
    rng = np.random.default_rng(42)
    raw = rng.uniform(0, 0.4, 200)
    outcomes = (rng.uniform(0, 1, 200) < raw).astype(np.float32)
    cal = IsotonicCalibrator()
    cal.fit(raw, outcomes)

    race = np.array([0.4, 0.2, 0.1, 0.05])  # one race's worth of horses
    probs = cal.predict(race, normalise=True)
    assert probs.sum() == pytest.approx(1.0, abs=1e-6)


def test_normalise_false_keeps_raw_scale():
    """normalise=False は isotonic そのままの値を返す（合計 ≠ 1）。"""
    rng = np.random.default_rng(7)
    raw = rng.uniform(0, 0.4, 200)
    outcomes = (rng.uniform(0, 1, 200) < raw).astype(np.float32)
    cal = IsotonicCalibrator()
    cal.fit(raw, outcomes)

    race = np.array([0.3, 0.2, 0.1])
    probs = cal.predict(race, normalise=False)
    # 合計は 1 にはならない (各馬の独立な calibrated win prob の合計)
    assert probs.sum() != pytest.approx(1.0)


def test_unfit_predict_raises():
    cal = IsotonicCalibrator()
    with pytest.raises(RuntimeError, match="must be fit"):
        cal.predict(np.array([0.5]))


def test_clipping_outside_train_range():
    """学習範囲外の入力は y_min/y_max にクリップされる。"""
    raw_train = np.linspace(0.1, 0.4, 100)
    rng = np.random.default_rng(11)
    outcomes = (rng.uniform(0, 1, 100) < raw_train).astype(np.float32)
    cal = IsotonicCalibrator()
    cal.fit(raw_train, outcomes)

    # 0.05 (range 下限以下) → y_min = 0.0 にクリップされない場合は学習範囲の最低値
    # 0.95 (range 上限以上) → y_max = 1.0 にクリップされない場合は最大値
    extrapolated = cal.predict(np.array([0.05, 0.95]), normalise=False)
    assert 0.0 <= extrapolated[0] <= 1.0
    assert 0.0 <= extrapolated[1] <= 1.0


def test_corrects_systematic_overprediction():
    """予測 = actual の 2 倍 のデータで学習すれば、calibrator は半分にする方向に補正する。"""
    np.random.seed(123)
    n = 5000
    actual_rate = np.random.uniform(0.05, 0.45, n)
    raw_pred = actual_rate * 2  # systematic 2x over-prediction
    outcomes = (np.random.uniform(0, 1, n) < actual_rate).astype(np.float32)

    cal = IsotonicCalibrator()
    cal.fit(raw_pred, outcomes)

    # raw_pred=0.4 (overshoot) → calibrated should be near 0.2 (actual rate)
    probe = np.array([0.4])
    out = cal.predict(probe, normalise=False)
    assert 0.10 < out[0] < 0.30, (
        f"calibrator should reduce overshoot, got {out[0]:.3f} for raw 0.4"
    )


def test_shape_mismatch_raises():
    cal = IsotonicCalibrator()
    with pytest.raises(ValueError, match="shape"):
        cal.fit(np.array([0.1, 0.2, 0.3]), np.array([0, 1]))
