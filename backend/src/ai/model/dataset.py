"""Race-grouped PyTorch Dataset for NN training.

One sample = one race.  Variable-length races are padded inside collate_fn.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset


class RaceDataset(Dataset):
    """Dataset that groups a DataFrame by race_id.

    Args:
        frame:             DataFrame with at least ``race_id``, ``feature_cols``,
                           and ``label_col`` columns.  Produced by build_training_frame.
        feature_cols:      Per-horse feature column names.
        race_feature_cols: Race-level feature column names.  Columns absent from
                           ``frame`` are silently ignored (defensive).
        label_col:         Column name for finish position.  Defaults to
                           ``"finish_position"``.
        time_col:          Column name for finish time in seconds.  Defaults to
                           ``"finish_time"``.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_cols: list[str],
        race_feature_cols: list[str],
        label_col: str = "finish_position",
        time_col: str = "finish_time",
        odds_col: str = "odds_win_raw",
    ) -> None:
        self.feature_cols = feature_cols
        # Only keep race feature cols that are actually present in the frame
        self.race_feature_cols = [c for c in race_feature_cols if c in frame.columns]
        self.label_col = label_col
        self.time_col = time_col
        # Raw (un-standardised) 単勝 odds for the betting-return losses.  Kept as
        # a separate non-feature column so NNPreprocessor never standardises it.
        self.odds_col = odds_col

        self._races: list[pd.DataFrame] = [
            group for _, group in frame.groupby("race_id", sort=True)
        ]

    def __len__(self) -> int:
        return len(self._races)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        race = self._races[idx]

        horse_features = torch.tensor(
            race[self.feature_cols].values, dtype=torch.float32
        )  # [n_horses, horse_feat_dim]

        if self.race_feature_cols:
            race_features = torch.tensor(
                race[self.race_feature_cols].iloc[0].values, dtype=torch.float32
            )  # [race_feat_dim]
        else:
            race_features = torch.zeros(0, dtype=torch.float32)

        def _col_to_tensor(col: str) -> torch.Tensor:
            if col in race.columns:
                return torch.tensor(race[col].values, dtype=torch.float32)
            return torch.full((len(race),), float("nan"))

        finish_positions = _col_to_tensor(self.label_col)
        finish_times = _col_to_tensor(self.time_col)
        odds_win = _col_to_tensor(self.odds_col)

        return {
            "horse_features": horse_features,
            "race_features": race_features,
            "finish_positions": finish_positions,
            "finish_times": finish_times,
            "odds_win": odds_win,
            "n_horses": len(race),
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Pad variable-length races to a common max_n_horses and stack into batch tensors.

    Args:
        batch: list of dicts returned by RaceDataset.__getitem__

    Returns:
        dict with keys:
            horse_features  [B, max_n_horses, F]
            race_features   [B, R]
            finish_positions [B, max_n_horses]   padded with NaN
            finish_times    [B, max_n_horses]    padded with NaN
            mask            [B, max_n_horses]    True = valid horse
    """
    max_n_horses = max(s["n_horses"] for s in batch)
    B = len(batch)

    horse_feat_dim = batch[0]["horse_features"].size(1)
    race_feat_dim = batch[0]["race_features"].size(0)

    horse_features_out = torch.zeros(B, max_n_horses, horse_feat_dim)
    race_features_out = torch.zeros(B, race_feat_dim)
    finish_positions_out = torch.full((B, max_n_horses), float("nan"))
    finish_times_out = torch.full((B, max_n_horses), float("nan"))
    odds_win_out = torch.full((B, max_n_horses), float("nan"))
    mask_out = torch.zeros(B, max_n_horses, dtype=torch.bool)

    for i, sample in enumerate(batch):
        n = sample["n_horses"]
        horse_features_out[i, :n] = sample["horse_features"]
        race_features_out[i] = sample["race_features"]
        finish_positions_out[i, :n] = sample["finish_positions"]
        finish_times_out[i, :n] = sample["finish_times"]
        odds_win_out[i, :n] = sample["odds_win"]
        mask_out[i, :n] = True

    return {
        "horse_features": horse_features_out,
        "race_features": race_features_out,
        "finish_positions": finish_positions_out,
        "finish_times": finish_times_out,
        "odds_win": odds_win_out,
        "mask": mask_out,
    }
