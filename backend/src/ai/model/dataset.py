"""Race-grouped PyTorch Dataset for NN training.

One sample = one race.  Variable-length races are padded inside collate_fn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from features.history_sequence import HistorySequenceCache


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
        history_cache: HistorySequenceCache | None = None,
        history_norm: tuple[np.ndarray, np.ndarray] | None = None,
        odds_feature_cols: list[str] | None = None,
    ) -> None:
        self.feature_cols = feature_cols
        # Only keep race feature cols that are actually present in the frame
        self.race_feature_cols = [c for c in race_feature_cols if c in frame.columns]
        self.label_col = label_col
        self.time_col = time_col
        # odds-at-scoring head 用の標準化済み odds 列 (任意)。None/空なら現行と
        # 完全に同一の dict を返す (odds_features キーを足さない)。
        self.odds_feature_cols = [c for c in (odds_feature_cols or []) if c in frame.columns]
        # Raw (un-standardised) 単勝 odds for the betting-return losses.  Kept as
        # a separate non-feature column so NNPreprocessor never standardises it.
        self.odds_col = odds_col
        # 履歴エンコーダ用 (任意)。None なら現行 (集約のみ) と完全に同一の dict を返す。
        self.history_cache = history_cache
        self.history_norm = history_norm

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

        out: dict[str, Any] = {
            "horse_features": horse_features,
            "race_features": race_features,
            "finish_positions": finish_positions,
            "finish_times": finish_times,
            "odds_win": odds_win,
            "n_horses": len(race),
        }

        if self.odds_feature_cols:
            out["odds_features"] = torch.tensor(
                race[self.odds_feature_cols].values, dtype=torch.float32
            )  # [n_horses, odds_dim] — 標準化済み (frame は transform 後)

        if self.history_cache is not None:
            out["history_seq"], out["history_lengths"] = self._history_for_race(race)
        return out

    def _history_for_race(self, race: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        """Build [n_horses, L_max, Hf] normalized history + [n_horses] lengths.

        過去走 0 件の馬は length=0 (history_seq は zero 行)。raw トークンを
        train fit の (mean, std) で標準化し NaN→0。
        """
        cache = self.history_cache
        assert cache is not None
        hf = cache.n_features
        race_id = str(race["race_id"].iloc[0])
        horse_ids = [str(h) for h in race["horse_id"].tolist()]
        if self.history_norm is not None:
            mean, std = self.history_norm
        else:
            mean, std = np.zeros(hf, dtype="float32"), np.ones(hf, dtype="float32")

        seqs: list[np.ndarray] = []
        lengths: list[int] = []
        for hid in horse_ids:
            raw = cache.seqs.get((race_id, hid))
            if raw is None or len(raw) == 0:
                seqs.append(np.zeros((0, hf), dtype="float32"))
                lengths.append(0)
            else:
                norm = np.nan_to_num((raw - mean) / std, nan=0.0).astype("float32")
                seqs.append(norm)
                lengths.append(len(norm))

        l_max = max((s.shape[0] for s in seqs), default=0)
        l_max = max(l_max, 1)  # テンソル形状確保のため最低 1
        hist = torch.zeros(len(horse_ids), l_max, hf)
        for j, s in enumerate(seqs):
            if s.shape[0] > 0:
                hist[j, : s.shape[0]] = torch.from_numpy(s)
        return hist, torch.tensor(lengths, dtype=torch.long)


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

    out = {
        "horse_features": horse_features_out,
        "race_features": race_features_out,
        "finish_positions": finish_positions_out,
        "finish_times": finish_times_out,
        "odds_win": odds_win_out,
        "mask": mask_out,
    }

    # 履歴系列 (任意): [B, max_n_horses, L_max, Hf] + lengths [B, max_n_horses]
    if "history_seq" in batch[0]:
        hf = batch[0]["history_seq"].shape[2]
        max_l = max(s["history_seq"].shape[1] for s in batch)
        history_seq_out = torch.zeros(B, max_n_horses, max_l, hf)
        history_lengths_out = torch.zeros(B, max_n_horses, dtype=torch.long)
        for i, sample in enumerate(batch):
            n = sample["n_horses"]
            hs = sample["history_seq"]
            history_seq_out[i, :n, : hs.shape[1]] = hs
            history_lengths_out[i, :n] = sample["history_lengths"]
        out["history_seq"] = history_seq_out
        out["history_lengths"] = history_lengths_out

    # odds 特徴 (任意, head 用): [B, max_n_horses, odds_dim] zero-pad
    if "odds_features" in batch[0]:
        od = batch[0]["odds_features"].shape[1]
        odds_out = torch.zeros(B, max_n_horses, od)
        for i, sample in enumerate(batch):
            n = sample["n_horses"]
            odds_out[i, :n] = sample["odds_features"]
        out["odds_features"] = odds_out

    return out
