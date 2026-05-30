"""Tests for ai.nn.dataset (RaceDataset, collate_fn)."""

from __future__ import annotations

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from ai.nn.dataset import RaceDataset, collate_fn

HORSE_FEAT_COLS = ["feat_a", "feat_b", "feat_c"]
RACE_FEAT_COLS = ["race_dist", "race_surface"]


def _make_frame(races: list[tuple[str, int]]) -> pd.DataFrame:
    """Build a synthetic DataFrame.

    Args:
        races: list of (race_id, n_horses)
    """
    rows = []
    horse_counter = 0
    for race_id, n_horses in races:
        for rank in range(1, n_horses + 1):
            rows.append(
                {
                    "race_id": race_id,
                    "horse_id": f"H{horse_counter:04d}",
                    "feat_a": float(rank),
                    "feat_b": float(rank * 2),
                    "feat_c": float(rank * 3),
                    "race_dist": 1600.0,
                    "race_surface": 1.0,
                    "finish_position": float(rank),
                    "finish_time": 90.0 + rank * 0.1,
                }
            )
            horse_counter += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3 races with different horse counts
# ---------------------------------------------------------------------------

RACE_SPEC = [("R001", 5), ("R002", 8), ("R003", 10)]


@pytest.fixture()
def frame():
    return _make_frame(RACE_SPEC)


@pytest.fixture()
def dataset(frame):
    return RaceDataset(frame, HORSE_FEAT_COLS, RACE_FEAT_COLS)


class TestRaceDataset:
    def test_len(self, dataset):
        assert len(dataset) == len(RACE_SPEC)

    def test_horse_features_shape(self, dataset):
        sample = dataset[0]  # 5 horses
        assert sample["horse_features"].shape == (5, len(HORSE_FEAT_COLS))

    def test_race_features_shape(self, dataset):
        sample = dataset[0]
        assert sample["race_features"].shape == (len(RACE_FEAT_COLS),)

    def test_n_horses_correct(self, dataset):
        for i, (_, n_horses) in enumerate(RACE_SPEC):
            assert dataset[i]["n_horses"] == n_horses

    def test_finish_positions_shape(self, dataset):
        sample = dataset[1]  # 8 horses
        assert sample["finish_positions"].shape == (8,)

    def test_finish_times_shape(self, dataset):
        sample = dataset[2]  # 10 horses
        assert sample["finish_times"].shape == (10,)

    def test_missing_race_feat_cols_ignored(self, frame):
        """race_feature_cols that are absent from frame should be silently dropped."""
        ds = RaceDataset(
            frame,
            HORSE_FEAT_COLS,
            race_feature_cols=["race_dist", "NONEXISTENT"],
        )
        sample = ds[0]
        # Only 1 race feat col survives
        assert sample["race_features"].shape == (1,)

    def test_missing_time_col_gives_nan(self, frame):
        frame_no_time = frame.drop(columns=["finish_time"])
        ds = RaceDataset(frame_no_time, HORSE_FEAT_COLS, RACE_FEAT_COLS)
        sample = ds[0]
        assert torch.all(torch.isnan(sample["finish_times"]))


class TestCollateFn:
    def test_batch_horse_features_shape(self, dataset):
        batch = [dataset[i] for i in range(len(RACE_SPEC))]
        result = collate_fn(batch)
        max_n = max(n for _, n in RACE_SPEC)
        assert result["horse_features"].shape == (len(RACE_SPEC), max_n, len(HORSE_FEAT_COLS))

    def test_batch_race_features_shape(self, dataset):
        batch = [dataset[i] for i in range(len(RACE_SPEC))]
        result = collate_fn(batch)
        assert result["race_features"].shape == (len(RACE_SPEC), len(RACE_FEAT_COLS))

    def test_mask_matches_n_horses(self, dataset):
        batch = [dataset[i] for i in range(len(RACE_SPEC))]
        result = collate_fn(batch)
        for i, (_, n_horses) in enumerate(RACE_SPEC):
            assert result["mask"][i, :n_horses].all()
            if n_horses < result["mask"].size(1):
                assert not result["mask"][i, n_horses:].any()

    def test_padded_positions_are_nan(self, dataset):
        batch = [dataset[i] for i in range(len(RACE_SPEC))]
        result = collate_fn(batch)
        # Race 0 has 5 horses; positions 5..9 should be NaN
        padded_pos = result["finish_positions"][0, 5:]
        assert torch.all(torch.isnan(padded_pos))

    def test_padded_times_are_nan(self, dataset):
        batch = [dataset[i] for i in range(len(RACE_SPEC))]
        result = collate_fn(batch)
        padded_times = result["finish_times"][0, 5:]
        assert torch.all(torch.isnan(padded_times))

    def test_valid_positions_not_nan(self, dataset):
        batch = [dataset[i] for i in range(len(RACE_SPEC))]
        result = collate_fn(batch)
        for i, (_, n_horses) in enumerate(RACE_SPEC):
            assert not torch.any(torch.isnan(result["finish_positions"][i, :n_horses]))

    def test_dataloader_integration(self, dataset):
        """DataLoader with collate_fn should produce correctly shaped batches."""
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=False)
        first_batch = next(iter(loader))
        # First batch: races 0 and 1 (5 and 8 horses) → max = 8
        assert first_batch["horse_features"].shape[0] == 2
        assert first_batch["mask"].shape == first_batch["finish_positions"].shape
