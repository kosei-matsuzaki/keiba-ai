"""Tests for ai.model.dataset (RaceDataset, collate_fn)."""

from __future__ import annotations

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from ai.model.dataset import RaceDataset, collate_fn

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


# ---------------------------------------------------------------------------
# History sequence integration (optional history_cache)
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

from features.history_sequence import HistorySequenceCache  # noqa: E402

HF = 4


def _history_cache(frame: pd.DataFrame) -> HistorySequenceCache:
    """Give every horse in R002 a history of length = its finish position; others none."""
    seqs: dict[tuple[str, str], np.ndarray] = {}
    for _, row in frame.iterrows():
        if row["race_id"] == "R002":
            length = int(row["finish_position"])
            seqs[(row["race_id"], row["horse_id"])] = np.arange(
                length * HF, dtype="float32"
            ).reshape(length, HF)
    return HistorySequenceCache(seqs, [f"h{i}" for i in range(HF)], max_len=15)


class TestRaceDatasetHistory:
    def test_history_shapes_and_lengths(self, frame):
        ds = RaceDataset(
            frame, HORSE_FEAT_COLS, RACE_FEAT_COLS, history_cache=_history_cache(frame)
        )
        sample = ds[1]  # R002 (sorted), 8 horses, lengths 1..8
        hs = sample["history_seq"]
        lengths = sample["history_lengths"]
        assert hs.shape == (8, 8, HF)  # [n, L_max=8, Hf]
        assert sorted(lengths.tolist()) == list(range(1, 9))
        # R001 horses have no history → all length 0
        assert ds[0]["history_lengths"].sum().item() == 0

    def test_collate_history_padding(self, frame):
        ds = RaceDataset(
            frame, HORSE_FEAT_COLS, RACE_FEAT_COLS, history_cache=_history_cache(frame)
        )
        batch = collate_fn([ds[0], ds[1], ds[2]])  # n=5,8,10 ; L=1(min),8,1(min)
        assert batch["history_seq"].shape == (3, 10, 8, HF)  # B, max_n, max_L, Hf
        assert batch["history_lengths"].shape == (3, 10)
        # R002 (i=1) keeps its true lengths 1..8 in the first 8 horse slots
        assert sorted(batch["history_lengths"][1, :8].tolist()) == list(range(1, 9))

    def test_no_history_cache_is_backward_compatible(self, dataset):
        assert "history_seq" not in dataset[0]
        batch = collate_fn([dataset[0], dataset[1]])
        assert "history_seq" not in batch
