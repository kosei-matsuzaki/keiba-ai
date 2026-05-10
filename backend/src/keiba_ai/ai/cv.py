"""Time-series cross-validation: rolling-origin (walk-forward) splits.

Each fold shifts the test window one step backward in time so that every
fold evaluates on unseen future data from the model's perspective.

Fold layout (n_folds=3, valid_months=V, test_months=T):

    |←── fold 3 train ──→|←V→|←T→|              (oldest)
         |←── fold 2 train ──→|←V→|←T→|
              |←── fold 1 train ──→|←V→|←T→|     (newest)
                                            ^ max_date

Fold 1 covers the most recent data (last T months = test), fold N the
oldest.  Folds with an empty train set are silently dropped so callers
never need to guard for that case.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date

import pandas as pd
from dateutil.relativedelta import relativedelta


def rolling_origin_splits(
    frame: pd.DataFrame,
    n_folds: int,
    valid_months: int,
    test_months: int,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Yield (train_df, valid_df, test_df) for each rolling-origin fold.

    Splits are computed on the 'date' column (string YYYY-MM-DD, lexicographically
    sortable) — no random shuffling, future data never leaks into earlier sets.

    Args:
        frame: Full feature frame with a 'date' column.
        n_folds: Number of folds to generate.  Must be >= 1.
        valid_months: Width of the validation window in months.
        test_months: Width of the test window in months.

    Yields:
        (train_df, valid_df, test_df) tuples in order fold-1 .. fold-N.
        Folds where train_df would be empty are skipped.
    """
    if frame.empty:
        return

    # Sort once; downstream slices inherit the ordering.
    frame = frame.sort_values("date").reset_index(drop=True)

    max_date_str: str = frame["date"].max()
    max_date = date.fromisoformat(max_date_str)

    for fold_idx in range(n_folds):
        # Each fold shifts the test-window end back by fold_idx * test_months.
        # fold 0 (== fold 1 in 1-based UI language) is the most recent.
        shift = relativedelta(months=fold_idx * test_months)

        fold_end = max_date - shift               # last day included in test
        test_start = fold_end - relativedelta(months=test_months)
        valid_start = test_start - relativedelta(months=valid_months)

        fold_end_str = fold_end.isoformat()
        test_start_str = test_start.isoformat()
        valid_start_str = valid_start.isoformat()

        train_df = frame[frame["date"] < valid_start_str].copy()
        valid_df = frame[
            (frame["date"] >= valid_start_str) & (frame["date"] < test_start_str)
        ].copy()
        # test includes both endpoints: [test_start, fold_end]
        test_df = frame[
            (frame["date"] >= test_start_str) & (frame["date"] <= fold_end_str)
        ].copy()

        # Skip folds where there is nothing to learn from.
        if train_df.empty:
            continue

        yield train_df, valid_df, test_df
