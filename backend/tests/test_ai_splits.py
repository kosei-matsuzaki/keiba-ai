"""Tests for ai/splits.py — time-series split correctness."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from keiba_ai.ai.splits import time_split


def _make_frame(start: date, n_days: int, rows_per_day: int = 5) -> pd.DataFrame:
    rows = []
    for i in range(n_days):
        d = (start + timedelta(days=i)).isoformat()
        for j in range(rows_per_day):
            rows.append({"race_id": f"R{i:03d}", "date": d, "horse_id": f"H{j}"})
    return pd.DataFrame(rows)


def test_split_basic():
    start = date(2023, 1, 1)
    frame = _make_frame(start, n_days=600)

    train, valid, test = time_split(frame, train_end=None, valid_months=6, test_months=3)

    assert not train.empty
    assert not valid.empty
    assert not test.empty

    # No overlap between sets
    assert set(train["date"]).isdisjoint(set(valid["date"]))
    assert set(valid["date"]).isdisjoint(set(test["date"]))


def test_split_order():
    start = date(2022, 1, 1)
    frame = _make_frame(start, n_days=730)

    train, valid, test = time_split(frame, train_end=None, valid_months=6, test_months=3)

    assert train["date"].max() < valid["date"].min()
    assert valid["date"].max() < test["date"].min()


def test_split_with_train_end():
    start = date(2022, 1, 1)
    frame = _make_frame(start, n_days=730)

    train_end = "2023-06-01"
    train, valid, test = time_split(frame, train_end=train_end, valid_months=6, test_months=3)

    # test set should start at train_end - test_months
    assert test["date"].min() <= train_end


def test_split_empty_frame():
    empty = pd.DataFrame(columns=["race_id", "date", "horse_id"])
    train, valid, test = time_split(empty, train_end=None)
    assert train.empty
    assert valid.empty
    assert test.empty


def test_split_total_rows_preserved():
    start = date(2022, 1, 1)
    frame = _make_frame(start, n_days=730)
    train, valid, test = time_split(frame, train_end=None, valid_months=6, test_months=3)
    assert len(train) + len(valid) + len(test) == len(frame)
