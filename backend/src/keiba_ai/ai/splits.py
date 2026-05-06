"""Time-series train/valid/test splits.

Splits are computed on the 'date' column (string YYYY-MM-DD, sortable lexicographically).
No random shuffling — future data must never appear in training or validation.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from dateutil.relativedelta import relativedelta


def time_split(
    frame: pd.DataFrame,
    train_end: str | None,
    valid_months: int = 12,
    test_months: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split frame into (train, valid, test) based on date boundaries.

    Boundary logic:
        test_start  = max_date - test_months
        valid_start = test_start - valid_months
        train       = [min_date, valid_start)
        valid       = [valid_start, test_start)
        test        = [test_start, max_date]

    If train_end is given, the split reference is train_end instead of max_date.
    """
    if frame.empty:
        empty = pd.DataFrame(columns=frame.columns)
        return empty, empty, empty

    max_date_str: str = frame["date"].max()
    ref_date = date.fromisoformat(train_end if train_end else max_date_str)

    test_start = ref_date - relativedelta(months=test_months)
    valid_start = test_start - relativedelta(months=valid_months)

    test_start_str = test_start.isoformat()
    valid_start_str = valid_start.isoformat()

    train = frame[frame["date"] < valid_start_str].copy()
    valid = frame[(frame["date"] >= valid_start_str) & (frame["date"] < test_start_str)].copy()
    test = frame[frame["date"] >= test_start_str].copy()

    return train, valid, test
