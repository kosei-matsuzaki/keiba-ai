"""Tests for ai/cv.py — rolling-origin (walk-forward) splits.

All tests use synthetic DataFrames with a 'date' column; no DB access required.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from ai.cv import rolling_origin_splits

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frame(start: date, n_days: int, rows_per_day: int = 3) -> pd.DataFrame:
    """Build a minimal frame with 'date', 'race_id', 'horse_id' columns."""
    rows = []
    for i in range(n_days):
        d = (start + timedelta(days=i)).isoformat()
        for j in range(rows_per_day):
            rows.append({"race_id": f"R{i:04d}", "date": d, "horse_id": f"H{j}"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Basic structural tests
# ---------------------------------------------------------------------------

def test_yields_correct_number_of_folds():
    """n_folds non-empty folds should be produced when data is abundant."""
    frame = _make_frame(date(2020, 1, 1), n_days=730)
    folds = list(rolling_origin_splits(frame, n_folds=3, valid_months=3, test_months=2))
    assert len(folds) == 3


def test_empty_frame_yields_nothing():
    frame = pd.DataFrame(columns=["race_id", "date", "horse_id"])
    folds = list(rolling_origin_splits(frame, n_folds=3, valid_months=3, test_months=2))
    assert folds == []


def test_single_fold_matches_shape():
    """With n_folds=1 we get exactly one (train, valid, test) tuple."""
    frame = _make_frame(date(2021, 1, 1), n_days=365)
    folds = list(rolling_origin_splits(frame, n_folds=1, valid_months=2, test_months=1))
    assert len(folds) == 1
    train_df, valid_df, test_df = folds[0]
    assert not train_df.empty
    # test covers the last 1 month window
    assert not test_df.empty


# ---------------------------------------------------------------------------
# Ordering / no-overlap invariants
# ---------------------------------------------------------------------------

def test_no_date_overlap_within_fold():
    """train, valid, test dates must be disjoint within each fold."""
    frame = _make_frame(date(2020, 1, 1), n_days=600)
    for train_df, valid_df, test_df in rolling_origin_splits(
        frame, n_folds=3, valid_months=3, test_months=2
    ):
        train_dates = set(train_df["date"])
        valid_dates = set(valid_df["date"])
        test_dates = set(test_df["date"])

        assert train_dates.isdisjoint(valid_dates), "train/valid overlap"
        assert train_dates.isdisjoint(test_dates), "train/test overlap"
        assert valid_dates.isdisjoint(test_dates), "valid/test overlap"


def test_chronological_order_within_fold():
    """train < valid < test in date order within every fold."""
    frame = _make_frame(date(2020, 1, 1), n_days=600)
    for train_df, valid_df, test_df in rolling_origin_splits(
        frame, n_folds=3, valid_months=3, test_months=2
    ):
        if not valid_df.empty:
            assert train_df["date"].max() < valid_df["date"].min()
            assert valid_df["date"].max() < test_df["date"].min()
        else:
            assert train_df["date"].max() < test_df["date"].min()


# ---------------------------------------------------------------------------
# Rolling-window shift correctness
# ---------------------------------------------------------------------------

def test_fold1_test_covers_most_recent_data():
    """Fold 1 (first yielded) test window should end at max_date of the frame."""
    frame = _make_frame(date(2021, 1, 1), n_days=365)
    max_date = frame["date"].max()

    folds = list(rolling_origin_splits(frame, n_folds=2, valid_months=2, test_months=1))
    assert len(folds) >= 1
    _, _, test_df = folds[0]
    # The most recent test window must include max_date.
    assert test_df["date"].max() == max_date


def test_fold2_test_ends_before_fold1_test():
    """Fold 2 test window must be older than (not overlap with) fold 1 test window.

    The windows are contiguous: fold 2's max date equals fold 1's min date minus
    one step because the test window for fold N starts at test_start of fold N-1.
    We therefore check that fold 2's test dates are strictly less than fold 1's
    maximum test date (i.e. they don't share the same upper bound).
    """
    frame = _make_frame(date(2020, 1, 1), n_days=730)
    folds = list(rolling_origin_splits(frame, n_folds=3, valid_months=2, test_months=2))
    assert len(folds) >= 2
    _, _, test1 = folds[0]
    _, _, test2 = folds[1]
    # Fold 2's test window max must be strictly less than fold 1's test window max.
    assert test2["date"].max() < test1["date"].max(), (
        "Fold 2 test window must be shifted back relative to fold 1"
    )


def test_later_folds_have_larger_or_equal_train():
    """Older folds (higher yield index) must have equal or larger training sets.

    Fold 1 (index 0, most recent) trains up to valid_start of the latest test window.
    Fold 2 (index 1) shifts everything back by test_months, so its valid_start is
    further in the past and its train set ends earlier — giving it a smaller or equal
    training set than fold 1.  Therefore train_sizes[0] >= train_sizes[1] >= ...
    """
    frame = _make_frame(date(2019, 1, 1), n_days=1000)
    folds = list(rolling_origin_splits(frame, n_folds=4, valid_months=3, test_months=2))
    train_sizes = [len(t) for t, _, _ in folds]
    # Fold 1 has the most recent test, so its train window ends at the latest
    # valid_start, meaning it encompasses the most historical rows.
    for i in range(len(train_sizes) - 1):
        assert train_sizes[i] >= train_sizes[i + 1], (
            f"Fold {i+1} train ({train_sizes[i]}) should be >= fold {i+2} train ({train_sizes[i+1]})"
        )


# ---------------------------------------------------------------------------
# Edge: folds with empty train are skipped
# ---------------------------------------------------------------------------

def test_folds_with_empty_train_are_skipped():
    """When requested n_folds would push train into empty territory, those are dropped."""
    # Only 40 days of data; with test_months=1 and valid_months=1 (≈60 days combined)
    # fold 1 might still have a small train, but fold 3 almost certainly won't.
    frame = _make_frame(date(2023, 1, 1), n_days=40)
    folds = list(rolling_origin_splits(frame, n_folds=5, valid_months=1, test_months=1))
    # At least one fold should have been dropped (not all 5 are returned).
    assert len(folds) < 5
    # Every returned fold must have a non-empty train.
    for train_df, _, _ in folds:
        assert not train_df.empty


# ---------------------------------------------------------------------------
# Regression: n_folds=1 consistent with time_split
# ---------------------------------------------------------------------------

def test_n_folds_1_consistent_with_time_split():
    """With n_folds=1, rolling_origin_splits should produce the same boundaries
    as ai.splits.time_split when train_end is None."""
    from ai.splits import time_split

    frame = _make_frame(date(2021, 1, 1), n_days=500)
    valid_months, test_months = 3, 2

    folds = list(rolling_origin_splits(frame, n_folds=1, valid_months=valid_months, test_months=test_months))
    assert len(folds) == 1
    cv_train, cv_valid, cv_test = folds[0]

    sp_train, sp_valid, sp_test = time_split(frame, train_end=None, valid_months=valid_months, test_months=test_months)

    # Date ranges must match.
    assert set(cv_train["date"]) == set(sp_train["date"]), "train mismatch between CV fold-1 and time_split"
    assert set(cv_valid["date"]) == set(sp_valid["date"]), "valid mismatch"
    assert set(cv_test["date"]) == set(sp_test["date"]), "test mismatch"
