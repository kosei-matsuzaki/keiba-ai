"""Tests for ConditionalIsotonicCalibrator (surface × n_runners bin)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai.calibrate import (
    ConditionalIsotonicCalibrator,
    _n_runners_bin,
)

# ---------------------------------------------------------------------------
# _n_runners_bin helper
# ---------------------------------------------------------------------------


def test_n_runners_bin_boundaries():
    assert _n_runners_bin(1) == 0
    assert _n_runners_bin(8) == 0
    assert _n_runners_bin(9) == 1
    assert _n_runners_bin(12) == 1
    assert _n_runners_bin(13) == 2
    assert _n_runners_bin(15) == 2
    assert _n_runners_bin(16) == 3
    assert _n_runners_bin(18) == 3


# ---------------------------------------------------------------------------
# ConditionalIsotonicCalibrator — basic fit / predict
# ---------------------------------------------------------------------------


def _make_conditions(surfaces, n_runners_list) -> pd.DataFrame:
    return pd.DataFrame({"surface": surfaces, "n_runners": n_runners_list})


def test_fit_predict_basic():
    """Calibrator should fit and predict without error."""
    rng = np.random.default_rng(0)
    n = 500
    raw = rng.uniform(0, 1, n)
    target = (raw + rng.normal(0, 0.1, n) > 0.5).astype(float)
    surfaces = ["芝" if i % 2 == 0 else "ダ" for i in range(n)]
    n_runners = [10 if i % 3 == 0 else 8 for i in range(n)]
    cond = _make_conditions(surfaces, n_runners)

    cal = ConditionalIsotonicCalibrator(min_samples_per_bin=10)
    cal.fit(raw, target, cond)

    # Predict with same conditions
    result = cal.predict(raw[:10], cond.iloc[:10])
    assert result.shape == (10,)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_global_fallback_when_bin_undersized():
    """Bins with < min_samples_per_bin must fall back to global calibrator."""
    rng = np.random.default_rng(1)
    n = 200
    raw = rng.uniform(0, 1, n)
    target = (raw > 0.5).astype(float)

    # Only 5 samples for (ダ, n_runners=16) — well below min_samples_per_bin=50.
    surfaces = ["芝"] * (n - 5) + ["ダ"] * 5
    n_runners_list = [8] * (n - 5) + [16] * 5
    cond = _make_conditions(surfaces, n_runners_list)

    cal = ConditionalIsotonicCalibrator(min_samples_per_bin=50)
    cal.fit(raw, target, cond)

    # (ダ, 3) bin must NOT be in the per-stratum dict.
    assert ("ダ", 3) not in cal._calibrators

    # Predicting for that bin should use global fallback without error.
    test_cond = _make_conditions(["ダ"], [18])
    result = cal.predict(np.array([0.4]), test_cond)
    assert result.shape == (1,)
    assert 0.0 <= float(result[0]) <= 1.0


def test_per_stratum_calibrator_fitted_when_sufficient_samples():
    """Strata with >= min_samples_per_bin must get their own calibrator."""
    rng = np.random.default_rng(2)
    n = 600
    raw = rng.uniform(0, 1, n)
    target = (raw > 0.5).astype(float)

    # 300 samples each for two strata.
    surfaces = ["芝"] * 300 + ["ダ"] * 300
    n_runners_list = [8] * 300 + [16] * 300
    cond = _make_conditions(surfaces, n_runners_list)

    cal = ConditionalIsotonicCalibrator(min_samples_per_bin=100)
    cal.fit(raw, target, cond)

    assert ("芝", 0) in cal._calibrators
    assert ("ダ", 3) in cal._calibrators


def test_predict_no_error_for_unseen_strata():
    """predict must not raise even if a (surface, bin) combination was not
    seen during fit — global fallback must be used transparently."""
    rng = np.random.default_rng(3)
    n = 300
    raw = rng.uniform(0, 1, n)
    target = (raw > 0.5).astype(float)
    cond = _make_conditions(["芝"] * n, [8] * n)

    cal = ConditionalIsotonicCalibrator(min_samples_per_bin=100)
    cal.fit(raw, target, cond)

    # Predict for combinations never seen during fit.
    unseen_cond = _make_conditions(["ダ", "芝", "ダ"], [12, 16, 18])
    raw_pred = rng.uniform(0, 1, 3)
    result = cal.predict(raw_pred, unseen_cond)
    assert result.shape == (3,)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_normalise_flag():
    """With normalise=True the result should sum to 1."""
    rng = np.random.default_rng(4)
    n = 200
    raw = rng.uniform(0, 1, n)
    target = (raw > 0.5).astype(float)
    cond = _make_conditions(["芝"] * n, [8] * n)
    cal = ConditionalIsotonicCalibrator(min_samples_per_bin=100)
    cal.fit(raw, target, cond)

    race_raw = rng.uniform(0, 1, 8)
    race_cond = _make_conditions(["芝"] * 8, [8] * 8)
    result = cal.predict(race_raw, race_cond, normalise=True)
    assert result.sum() == pytest.approx(1.0, abs=1e-6)


def test_raises_before_fit():
    """predict must raise RuntimeError when called before fit."""
    cal = ConditionalIsotonicCalibrator()
    with pytest.raises(RuntimeError, match="must be fit"):
        cal.predict(np.array([0.5]), _make_conditions(["芝"], [8]))


def test_fit_raises_mismatched_shapes():
    """fit must raise ValueError when raw and target shapes differ."""
    cal = ConditionalIsotonicCalibrator()
    raw = np.array([0.1, 0.2, 0.3])
    target = np.array([0.0, 1.0])
    cond = _make_conditions(["芝"] * 3, [8] * 3)
    with pytest.raises(ValueError, match="shape"):
        cal.fit(raw, target, cond)


def test_fit_raises_conditions_length_mismatch():
    """fit must raise ValueError when conditions length differs from raw."""
    cal = ConditionalIsotonicCalibrator()
    raw = np.array([0.1, 0.2, 0.3])
    target = np.array([0.0, 1.0, 0.0])
    cond = _make_conditions(["芝"] * 2, [8] * 2)  # length 2 != 3
    with pytest.raises(ValueError, match="length"):
        cal.fit(raw, target, cond)


# ---------------------------------------------------------------------------
# Calibration quality: conditional should outperform global on biased data
# ---------------------------------------------------------------------------


def test_conditional_better_than_global_on_biased_data():
    """Synthetic data where dirt/sprint and turf/long have opposing biases.

    For dirt sprint (n<=8): raw > 0.6 → hit, model over-estimates low prob.
    For turf long (n>=16): raw < 0.4 → hit, model over-estimates high prob.

    ConditionalIsotonicCalibrator should capture these opposite patterns
    and produce lower mean-squared error than a single global calibrator.
    """
    from sklearn.isotonic import IsotonicRegression

    rng = np.random.default_rng(42)
    n_each = 300

    # Dirt sprint: P(win) rises steeply above raw=0.6
    def dirt_prob(x):
        return np.where(x > 0.6, 0.8, 0.1)

    # Turf long: P(win) is high for raw < 0.4, low above
    def turf_prob(x):
        return np.where(x < 0.4, 0.8, 0.1)

    raw_dirt = rng.uniform(0, 1, n_each)
    target_dirt = (rng.uniform(0, 1, n_each) < dirt_prob(raw_dirt)).astype(float)

    raw_turf = rng.uniform(0, 1, n_each)
    target_turf = (rng.uniform(0, 1, n_each) < turf_prob(raw_turf)).astype(float)

    raw_all = np.concatenate([raw_dirt, raw_turf])
    target_all = np.concatenate([target_dirt, target_turf])
    surfaces = ["ダ"] * n_each + ["芝"] * n_each
    n_runners = [8] * n_each + [16] * n_each
    cond = _make_conditions(surfaces, n_runners)

    # Conditional calibrator
    cond_cal = ConditionalIsotonicCalibrator(min_samples_per_bin=50)
    cond_cal.fit(raw_all, target_all, cond)
    cond_pred = cond_cal.predict(raw_all, cond)
    cond_mse = float(np.mean((cond_pred - target_all) ** 2))

    # Global isotonic calibrator
    global_iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0, increasing=True)
    global_iso.fit(raw_all, target_all)
    global_pred = global_iso.predict(raw_all)
    global_mse = float(np.mean((global_pred - target_all) ** 2))

    assert cond_mse < global_mse, (
        f"Expected conditional MSE ({cond_mse:.4f}) < global MSE ({global_mse:.4f})"
    )


# ---------------------------------------------------------------------------
# ComboCalibrators with use_conditional=True
# ---------------------------------------------------------------------------


def test_combo_calibrators_use_conditional_fit_and_predict():
    """ComboCalibrators with use_conditional=True should store
    ConditionalIsotonicCalibrator and accept conditions in predict."""
    from ai.calibrate import ComboCalibrators, ConditionalIsotonicCalibrator

    rng = np.random.default_rng(10)
    n = 300
    raw = rng.uniform(0, 1, n)
    outcomes = (raw > 0.5).astype(float)
    surfaces = ["芝" if i % 2 == 0 else "ダ" for i in range(n)]
    n_runners_list = [10 if i % 3 == 0 else 8 for i in range(n)]
    cond_df = pd.DataFrame({"surface": surfaces, "n_runners": n_runners_list})

    cal = ComboCalibrators(use_conditional=True)
    cal.fit_for("馬連", raw, outcomes, conditions=cond_df)

    assert cal.has("馬連")
    assert isinstance(cal._calibrators["馬連"], ConditionalIsotonicCalibrator)

    pred_cond = pd.DataFrame({"surface": ["芝"], "n_runners": [8]})
    result = cal.predict("馬連", np.array([0.3]), conditions=pred_cond)
    assert result.shape == (1,)
    assert 0.0 <= float(result[0]) <= 1.0


def test_combo_calibrators_default_backward_compat():
    """ComboCalibrators with default use_conditional=False should behave
    identically to the pre-extension version (no conditions needed)."""
    from ai.calibrate import ComboCalibrators

    rng = np.random.default_rng(11)
    n = 300
    raw = rng.uniform(0, 1, n)
    outcomes = (raw > 0.5).astype(float)

    cal = ComboCalibrators()
    cal.fit_for("馬連", raw, outcomes)

    assert cal.has("馬連")
    result = cal.predict("馬連", np.array([0.3]))
    assert result.shape == (1,)


def test_combo_calibrators_use_conditional_raises_without_conditions():
    """predict with conditional calibrator must raise if conditions not passed."""
    from ai.calibrate import ComboCalibrators

    rng = np.random.default_rng(12)
    n = 300
    raw = rng.uniform(0, 1, n)
    outcomes = (raw > 0.5).astype(float)
    cond_df = pd.DataFrame({"surface": ["芝"] * n, "n_runners": [8] * n})

    cal = ComboCalibrators(use_conditional=True)
    cal.fit_for("馬連", raw, outcomes, conditions=cond_df)

    with pytest.raises(ValueError, match="conditions"):
        cal.predict("馬連", np.array([0.3]))
