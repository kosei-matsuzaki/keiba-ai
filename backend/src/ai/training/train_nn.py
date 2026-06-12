"""CLI: Train a PyTorch NN (Set Transformer) model.

The default objective is ``multi`` — a production **all-markets** objective that
optimises 単複 betting return (``log_growth``) while calibrating the 連系 combo
probabilities inside the NN (``combo_nll``), so no external combo_calibrators are
needed.  Model selection is on validation 単勝 ROI.  ``plackett_luce`` is the
ranking loss used as the first stage of the recommended **two-stage** recipe
(PL pretrain → ``--init-from`` → ``multi`` fine-tune).  See docs/ai-model.md.

Usage:
    python -m ai.training.train_nn \\
        --train-end YYYY-MM-DD --valid-months 12 --test-months 6 \\
        --loss {multi,log_growth,combo_nll,plackett_luce} \\
        --monitor {valid_tansho_roi,valid_fukusho_roi,valid_ndcg3} \\
        --combo-weight 0.01 --combo-bet-type 馬連 \\
        --hidden-dim 64 --embed-dim 32 --n-heads 4 \\
        --batch-size 32 --max-epochs 100 --learning-rate 1e-3 \\
        --device {cpu,cuda} \\
        --db PATH
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import math
from datetime import UTC, datetime
from pathlib import Path

import lightning as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import DataLoader

from ai.core.labels import assign_relevance
from ai.core.splits import time_split
from ai.core.temperature import TemperatureScaler
from ai.model.dataset import RaceDataset, collate_fn
from ai.model.loss import (
    combo_nll_loss,
    log_growth_loss,
    multi_objective_loss,
    plackett_luce_loss,
)
from ai.model.net import RaceTransformerModel
from ai.model.preprocess import NNPreprocessor
from ai.model.registry import save_nn_model
from core.paths import data_dir, db_path
from db.models.model_run import ModelRun
from db.session import make_engine, session_scope
from features.builder import CATEGORICAL_FEATURES, build_training_frame, get_active_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Race-level features: constant within a race (course/distance/surface/etc.)
# These are broadcast to the race vector rather than per-horse vectors.
RACE_FEATURE_COLS: list[str] = [
    "course",
    "distance",
    "surface",
    "weather",
    "track_condition",
    "race_class",
    "n_runners",
]


def _split_feature_cols(all_feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """Split feature columns into (horse_feature_cols, race_feature_cols).

    Race-level features are those constant within a race (RACE_FEATURE_COLS).
    Horse-level features are everything else.

    Returns:
        (horse_feature_cols, race_feature_cols_present)
        Both lists contain only columns that are in all_feature_cols.
    """
    race_set = set(RACE_FEATURE_COLS)
    race_cols = [c for c in all_feature_cols if c in race_set]
    horse_cols = [c for c in all_feature_cols if c not in race_set]
    return horse_cols, race_cols


def _encode_categoricals(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Legacy: label-encode categoricals + numeric coercion, fitting on the frame itself.

    **Bugged when called separately on train/valid/test** — each split produces its
    own mapping (e.g. course=Tokyo could be 5.0 in train and 2.0 in valid).
    Use :class:`NNPreprocessor` instead, which fits on train and applies the same
    mapping to all splits + inference.

    Retained only as a fallback for inference against legacy NN models saved
    before preprocessor.pkl was introduced.
    """
    frame = frame.copy()
    cat_set = set(CATEGORICAL_FEATURES)

    for col in feature_cols:
        if col not in frame.columns:
            frame[col] = 0.0
            continue
        if col in cat_set or frame[col].dtype == object:
            unique_vals = [v for v in frame[col].dropna().unique()]
            mapping = {v: float(i) for i, v in enumerate(sorted(unique_vals, key=str))}
            frame[col] = frame[col].map(mapping).fillna(-1.0)
        else:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    return frame


def _sample_history_kw(sample: dict, device: torch.device) -> dict:
    """単一レース sample の履歴を model.forward kwargs ([1,n,L,Hf]) に。履歴なしは空。"""
    if "history_seq" not in sample:
        return {}
    return {
        "history_seq": sample["history_seq"].unsqueeze(0).to(device),
        "history_lengths": sample["history_lengths"].unsqueeze(0).to(device),
    }


def _batch_history_kw(batch: dict, device: torch.device) -> dict:
    """バッチの履歴を model.forward kwargs に。履歴なしは空 dict。"""
    if "history_seq" not in batch:
        return {}
    return {
        "history_seq": batch["history_seq"].to(device),
        "history_lengths": batch["history_lengths"].to(device),
    }


def _hit_rate(records: list, gate_key: str, hit_fn) -> float:
    """top-1 賭け記録のうち、gate_key が有効な賭けで hit_fn(r) が真の割合 (的中率)。

    tansho: gate=tansho_ret (有効オッズで賭けた), hit=top-1 が 1 着。
    fukusho: gate=place_ret (複勝払戻あり), hit=top-1 が複勝圏 (払戻>0)。
    """
    rs = [r for r in records if r.get(gate_key) is not None]
    if not rs:
        return float("nan")
    return sum(1 for r in rs if hit_fn(r)) / len(rs)


def _compute_ndcg_nn(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    horse_feature_cols: list[str],
    race_feature_cols: list[str],
    at: int,
    device: torch.device,
    history_cache=None,
    history_norm=None,
) -> float:
    """Compute NDCG@at for the NN model across all races in frame."""
    if frame.empty:
        return float("nan")

    from sklearn.metrics import ndcg_score

    frame = frame.copy()
    frame["relevance"] = frame["finish_position"].map(assign_relevance)

    # RaceDataset groups by race_id (sort=True) → same order as groupby below
    dataset = RaceDataset(
        frame, horse_feature_cols, race_feature_cols,
        history_cache=history_cache, history_norm=history_norm,
    )

    model.eval()
    race_scores: list[list[float]] = []

    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            n = sample["n_horses"]
            hf = sample["horse_features"].unsqueeze(0).to(device)  # [1, n, F]
            rf = sample["race_features"].unsqueeze(0).to(device)   # [1, R]
            mask_single = torch.zeros(1, n, dtype=torch.bool, device=device)
            mask_single[0, :n] = True

            scores = model(hf, rf, mask_single, **_sample_history_kw(sample, device))  # [1, n]
            race_scores.append(scores[0, :n].cpu().tolist())

    ndcg_vals: list[float] = []
    for (_, grp), scores in zip(frame.groupby("race_id", sort=True), race_scores, strict=True):
        if len(grp) < 2:
            continue
        true_rel = grp["relevance"].values.reshape(1, -1)
        pred_sc = np.array(scores).reshape(1, -1)
        ndcg_vals.append(float(ndcg_score(true_rel, pred_sc, k=at)))

    return float(np.mean(ndcg_vals)) if ndcg_vals else float("nan")


# Outcome / payoff columns captured *before* NNPreprocessor.transform so that
# real-odds ROI (and temperature payback) use raw odds, not standardised values.
# odds_win is a FEATURE column → transform standardises it (2.0 → -0.64); reading
# it post-transform yields garbage.  finish_position / payout_place are not
# features and are unaffected, but we capture them together for alignment.
_OUTCOME_COLS: list[str] = [
    "race_id",
    "horse_id",
    "finish_position",
    "odds_win",
    "payout_place",
]


def _capture_raw_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    """Subset of outcome/payoff columns, taken before preprocessing.

    Row order is preserved so that ``frame.groupby('race_id', sort=True)`` on the
    returned frame aligns row-for-row (and group-for-group) with a RaceDataset
    built from the corresponding transformed frame.
    """
    if frame.empty:
        return frame
    cols = [c for c in _OUTCOME_COLS if c in frame.columns]
    return frame[cols].copy()


def _parse_payout_place_cell(raw: object) -> dict[int, int] | None:
    """Parse a payout_place JSON cell ({finish_position: payout_yen}) → dict."""
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return None
    try:
        d = json.loads(raw)
        return {int(k): int(v) for k, v in d.items()}
    except (ValueError, TypeError):
        return None


def _compute_winplace_roi_nn(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    raw_df: pd.DataFrame,
    horse_feature_cols: list[str],
    race_feature_cols: list[str],
    device: torch.device,
    history_cache=None,
    history_norm=None,
    collect_records: list | None = None,
) -> tuple[float, float]:
    """Top-1 flat-stake 単勝 / 複勝 ROI on real odds across all races in frame.

    collect_records (任意): 渡すと各レースの top-1 賭け記録
    {odds, won, tansho_ret, place_ret} を append する (bootstrap CI / オッズ分布用)。

    Per race, stake 1 unit on the highest-score horse:
      - 単勝: return = odds_win   if it finished 1st        else 0
      - 複勝: return = payout/100 if it finished in-the-money else 0

    `raw_df` must carry **un-standardised** odds_win / payout_place columns
    (captured via _capture_raw_outcomes before NNPreprocessor.transform), aligned
    row-for-row with `frame`.  Returns (tansho_roi, fukusho_roi); NaN when no
    eligible bets exist.
    """
    if frame.empty or raw_df.empty:
        return float("nan"), float("nan")

    dataset = RaceDataset(
        frame, horse_feature_cols, race_feature_cols,
        history_cache=history_cache, history_norm=history_norm,
    )

    model.eval()
    t_inv = t_gross = 0.0
    f_inv = f_gross = 0.0

    with torch.no_grad():
        for i, (_race_id, grp) in enumerate(raw_df.groupby("race_id", sort=True)):
            if len(grp) < 2:
                continue

            sample = dataset[i]
            n = sample["n_horses"]
            hf = sample["horse_features"].unsqueeze(0).to(device)
            rf = sample["race_features"].unsqueeze(0).to(device)
            mask_single = torch.zeros(1, n, dtype=torch.bool, device=device)
            mask_single[0, :n] = True

            scores = model(hf, rf, mask_single, **_sample_history_kw(sample, device))[0, :n].cpu().numpy()
            top = int(np.argmax(scores))

            positions = grp["finish_position"].values
            pos = float(positions[top]) if top < len(positions) else float("nan")
            won = (not np.isnan(pos)) and int(pos) == 1

            rec_odds: float | None = None
            rec_tansho: float | None = None
            rec_place: float | None = None

            # 単勝
            if "odds_win" in grp.columns:
                o = float(grp["odds_win"].values[top])
                if not (np.isnan(o) or o <= 0.0):
                    t_inv += 1.0
                    rec_odds = o
                    rec_tansho = o if won else 0.0
                    if won:
                        t_gross += o

            # 複勝
            if "payout_place" in grp.columns:
                payout_map = _parse_payout_place_cell(grp["payout_place"].values[top])
                if payout_map:
                    f_inv += 1.0
                    hit_place = (not np.isnan(pos)) and int(pos) in payout_map
                    rec_place = payout_map[int(pos)] / 100.0 if hit_place else 0.0
                    if hit_place:
                        f_gross += payout_map[int(pos)] / 100.0

            if collect_records is not None:
                collect_records.append({
                    "odds": rec_odds, "won": won,
                    "tansho_ret": rec_tansho, "place_ret": rec_place,
                })

    tansho = t_gross / t_inv if t_inv > 0 else float("nan")
    fukusho = f_gross / f_inv if f_inv > 0 else float("nan")
    return tansho, fukusho


def _fit_temperature_scaler_nn(
    model: torch.nn.Module,
    valid_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    horse_feature_cols: list[str],
    race_feature_cols: list[str],
    device: torch.device,
    history_cache=None,
    history_norm=None,
) -> TemperatureScaler:
    """Fit TemperatureScaler on valid_df using payback-maximising grid search.

    Per-race scores are extracted by running the NN in eval mode through
    RaceDataset (same indexing order as groupby("race_id", sort=True)).

    Args:
        model: Trained RaceModel (already in eval mode).
        valid_df: Transformed validation DataFrame (used only for model forward
            via RaceDataset).
        raw_df: Pre-transform outcome frame (raw odds_win / payout_place /
            finish_position), aligned row-for-row with valid_df.  Reading odds
            from valid_df would use standardised values (odds_win is a feature).
        horse_feature_cols: Horse-level feature columns.
        race_feature_cols: Race-level feature columns.
        device: Torch device for inference.

    Returns:
        Fitted TemperatureScaler.
    """
    dataset = RaceDataset(
        valid_df, horse_feature_cols, race_feature_cols,
        history_cache=history_cache, history_norm=history_norm,
    )

    scores_per_race: list[np.ndarray] = []
    finish_positions_per_race: list[np.ndarray] = []
    odds_win_per_race: list[np.ndarray] = []
    payout_place_per_race: list = []

    model.eval()
    # Iterate dataset by integer index — same groupby("race_id", sort=True) order
    # as RaceDataset._races, so grp and dataset[i] are always aligned.  Odds are
    # read from raw_df (un-standardised); valid_df only drives the model forward.
    with torch.no_grad():
        for i, (_race_id, grp) in enumerate(raw_df.groupby("race_id", sort=True)):
            if len(grp) < 2:
                continue

            sample = dataset[i]
            n = sample["n_horses"]
            hf = sample["horse_features"].unsqueeze(0).to(device)
            rf = sample["race_features"].unsqueeze(0).to(device)
            mask_single = torch.zeros(1, n, dtype=torch.bool, device=device)
            mask_single[0, :n] = True

            raw_scores = model(hf, rf, mask_single, **_sample_history_kw(sample, device))  # [1, n]
            scores = raw_scores[0, :n].cpu().numpy()

            finish_pos = grp["finish_position"].values.astype(float)
            odds_win = (
                grp["odds_win"].values.astype(float)
                if "odds_win" in grp.columns
                else np.full(len(grp), float("nan"))
            )

            payout_map: dict[int, int] | None = None
            if "payout_place" in grp.columns:
                raw_val = grp["payout_place"].dropna()
                if not raw_val.empty:
                    import json as _json
                    try:
                        raw_dict = _json.loads(raw_val.iloc[0])
                        payout_map = {int(k): int(v) for k, v in raw_dict.items()}
                    except (ValueError, TypeError):
                        payout_map = None

            scores_per_race.append(scores)
            finish_positions_per_race.append(finish_pos)
            odds_win_per_race.append(odds_win)
            payout_place_per_race.append(payout_map)

    scaler = TemperatureScaler()
    scaler.fit(
        scores_per_race=scores_per_race,
        finish_positions_per_race=finish_positions_per_race,
        odds_win_per_race=odds_win_per_race,
        payout_place_per_race=payout_place_per_race,
    )
    return scaler


def _compute_loss_on_dataset(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    horse_feature_cols: list[str],
    race_feature_cols: list[str],
    loss_fn_name: str,
    device: torch.device,
    combo_bet_type: str = "馬連",
    combo_weight: float = 0.01,
    history_cache=None,
    history_norm=None,
) -> float:
    """Compute mean loss over all races in frame (for test-set reporting)."""
    if frame.empty:
        return float("nan")

    dataset = RaceDataset(
        frame, horse_feature_cols, race_feature_cols,
        history_cache=history_cache, history_norm=history_norm,
    )
    if len(dataset) == 0:
        return float("nan")

    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, shuffle=False)

    loss_fn = _build_loss_fn(
        loss_fn_name, combo_bet_type=combo_bet_type, combo_weight=combo_weight
    )

    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            hf = batch["horse_features"].to(device)
            rf = batch["race_features"].to(device)
            fp = batch["finish_positions"].to(device)
            mask = batch["mask"].to(device)

            scores = model(hf, rf, mask, **_batch_history_kw(batch, device))

            if loss_fn_name == "log_growth":
                loss = loss_fn(scores, fp, batch["odds_win"].to(device), mask)
            elif loss_fn_name == "combo_nll":
                loss = loss_fn(scores, fp, mask)
            elif loss_fn_name == "multi":
                loss = loss_fn(scores, fp, batch["odds_win"].to(device), mask)
            else:
                loss = loss_fn(scores, fp, mask)

            if not torch.isnan(loss):
                total_loss += loss.item()
                n_batches += 1

    return total_loss / n_batches if n_batches > 0 else float("nan")


def _build_loss_fn(
    loss_name: str,
    kelly_fraction: float = 0.25,
    combo_bet_type: str = "馬連",
    combo_weight: float = 0.01,
):
    """Return the loss callable for the given name.

    kelly_fraction affects the betting losses (bankroll fraction staked per race);
    combo_bet_type selects the 連系 type for combo_nll / the multi combo term;
    combo_weight weights the combo-calibration term of the `multi` objective.
    All are ignored by plackett_luce.
    """
    if loss_name == "plackett_luce":
        return plackett_luce_loss
    if loss_name == "log_growth":
        return functools.partial(log_growth_loss, kelly_fraction=kelly_fraction)
    if loss_name == "combo_nll":
        return functools.partial(combo_nll_loss, bet_type=combo_bet_type)
    if loss_name == "multi":
        return functools.partial(
            multi_objective_loss,
            combo_weight=combo_weight,
            kelly_fraction=kelly_fraction,
            combo_bet_type=combo_bet_type,
        )
    raise ValueError(
        f"Unknown loss: {loss_name!r}. "
        "Choose from multi, log_growth, combo_nll, plackett_luce"
    )


class RaceLitModule(pl.LightningModule):
    """Lightning wrapper around RaceModel / RaceTransformerModel.

    Args:
        model: torch.nn.Module exposing (horse_features, race_features, mask) -> scores
        loss_fn_name: one of "multi", "log_growth", "combo_nll", "plackett_luce"
        learning_rate: AdamW initial learning rate (cosine-annealed to 0)
        weight_decay: AdamW weight decay (set 0 to disable)
        max_epochs: total epochs — used as the cosine schedule period
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn_name: str,
        learning_rate: float,
        weight_decay: float = 0.0,
        max_epochs: int = 100,
        combo_bet_type: str = "馬連",
        combo_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn_name = loss_fn_name
        self.loss_fn = _build_loss_fn(
            loss_fn_name, combo_bet_type=combo_bet_type, combo_weight=combo_weight
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        # history_seq/history_lengths は履歴有効時のみ batch に存在 (Lightning が
        # device 転送済み)。無い場合 None → model は現行パス。
        scores = self.model(
            batch["horse_features"],
            batch["race_features"],
            batch["mask"],
            history_seq=batch.get("history_seq"),
            history_lengths=batch.get("history_lengths"),
        )
        if self.loss_fn_name in ("log_growth", "multi"):
            loss = self.loss_fn(
                scores,
                batch["finish_positions"],
                batch["odds_win"],
                batch["mask"],
            )
        else:  # combo_nll / plackett_luce
            loss = self.loss_fn(scores, batch["finish_positions"], batch["mask"])
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self._compute_loss(batch)
        if torch.isnan(loss):
            # Return a dummy loss to avoid crashing when all races are degenerate
            loss = torch.tensor(0.0, requires_grad=True, device=self.device)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss = self._compute_loss(batch)
        if not torch.isnan(loss):
            self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, self.max_epochs)
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class _NDCG3Callback(pl.Callback):
    """Compute NDCG@3 on the validation frame at end of each validation epoch.

    The scalar is logged as ``valid_ndcg3`` so EarlyStopping can monitor a
    ranking metric directly (PL loss and NDCG often peak at different epochs).
    """

    def __init__(
        self,
        valid_df: pd.DataFrame,
        horse_feature_cols: list[str],
        race_feature_cols: list[str],
        device: torch.device,
        history_cache=None,
        history_norm=None,
    ) -> None:
        self.valid_df = valid_df
        self.horse_feature_cols = horse_feature_cols
        self.race_feature_cols = race_feature_cols
        self.device = device
        self.history_cache = history_cache
        self.history_norm = history_norm

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.valid_df.empty:
            return
        ndcg3 = _compute_ndcg_nn(
            pl_module.model,
            self.valid_df,
            self.horse_feature_cols,
            self.race_feature_cols,
            at=3,
            device=self.device,
            history_cache=self.history_cache,
            history_norm=self.history_norm,
        )
        if math.isnan(ndcg3):
            return
        pl_module.log("valid_ndcg3", ndcg3, prog_bar=True)


class _WinPlaceROICallback(pl.Callback):
    """Log real-odds top-1 単勝 / 複勝 ROI on the validation set each epoch.

    Logged as ``valid_tansho_roi`` / ``valid_fukusho_roi`` so EarlyStopping can
    monitor a betting-return metric directly (the deployment objective) instead
    of the NDCG ranking proxy.
    """

    def __init__(
        self,
        valid_df: pd.DataFrame,
        valid_raw_df: pd.DataFrame,
        horse_feature_cols: list[str],
        race_feature_cols: list[str],
        device: torch.device,
        history_cache=None,
        history_norm=None,
    ) -> None:
        self.valid_df = valid_df
        self.valid_raw_df = valid_raw_df
        self.horse_feature_cols = horse_feature_cols
        self.race_feature_cols = race_feature_cols
        self.device = device
        self.history_cache = history_cache
        self.history_norm = history_norm

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.valid_df.empty:
            return
        tansho, fukusho = _compute_winplace_roi_nn(
            pl_module.model,
            self.valid_df,
            self.valid_raw_df,
            self.horse_feature_cols,
            self.race_feature_cols,
            self.device,
            history_cache=self.history_cache,
            history_norm=self.history_norm,
        )
        # Always log both keys (substitute a floor for NaN) so EarlyStopping can
        # monitor either without crashing when a degenerate epoch yields no
        # eligible bets.  mode="max", so -1.0 is never selected as the best.
        pl_module.log("valid_tansho_roi", tansho if not math.isnan(tansho) else -1.0, prog_bar=True)
        pl_module.log("valid_fukusho_roi", fukusho if not math.isnan(fukusho) else -1.0, prog_bar=True)


# EarlyStopping monitors — all maximised (mode="max")
_VALID_MONITORS: frozenset[str] = frozenset(
    {"valid_ndcg3", "valid_tansho_roi", "valid_fukusho_roi"}
)


def train_nn(
    db: Path | None = None,
    train_end: str | None = None,
    valid_months: int = 12,
    test_months: int = 6,
    loss: str = "multi",
    hidden_dim: int = 64,
    embed_dim: int = 32,
    n_heads: int = 4,
    batch_size: int = 32,
    max_epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    fit_temperature: bool = True,
    n_transformer_layers: int = 2,
    cat_embed_dim: int = 4,
    weight_decay: float = 1e-4,
    gradient_clip_val: float = 1.0,
    early_stopping_patience: int = 10,
    monitor: str = "valid_tansho_roi",
    prebuilt_frame: pd.DataFrame | None = None,
    init_from: Path | None = None,
    combo_bet_type: str = "馬連",
    combo_weight: float = 0.01,
    persist: bool = True,
    use_history: bool = False,
    history_seq_len: int = 15,
    prebuilt_history=None,
    return_test_bets: bool = False,
) -> dict:
    """Run the full NN training pipeline. Returns metrics dict.

    monitor: EarlyStopping metric (all maximised).  Default valid_tansho_roi
        (real-odds betting return — aligns model selection with the deployment
        objective); also valid_fukusho_roi or the legacy valid_ndcg3 ranking proxy.
    prebuilt_frame: optional pre-built training frame (output of
        build_training_frame).  When provided, the expensive feature build is
        skipped — useful for sweeping multiple configs over the same data.
    init_from: optional path to a saved NN model dir; its model.pt state_dict is
        loaded into the fresh model before training (two-stage / fine-tuning,
        e.g. PL-pretrain → log_growth fine-tune).  Architecture must match.
    """
    if monitor not in _VALID_MONITORS:
        raise ValueError(
            f"Unknown monitor {monitor!r}. Choose from {sorted(_VALID_MONITORS)}"
        )
    resolved_db = db or db_path()
    engine = make_engine(resolved_db)

    if prebuilt_frame is not None:
        log.info("Using prebuilt feature frame (%d rows)", len(prebuilt_frame))
        frame = prebuilt_frame
    else:
        log.info("Building feature frame from %s", resolved_db)
        with session_scope(engine) as session:
            frame = build_training_frame(session)

    if frame.empty:
        raise RuntimeError("No training data found in the database.")

    log.info("Total rows: %d | Races: %d", len(frame), frame["race_id"].nunique())

    train_df, valid_df, test_df = time_split(frame, train_end, valid_months, test_months)
    log.info(
        "Split → train=%d rows, valid=%d rows, test=%d rows",
        len(train_df),
        len(valid_df),
        len(test_df),
    )

    if train_df.empty:
        log.warning(
            "Train set is empty — using full frame for training (test will leak)."
        )
        train_df = frame.copy()
        valid_df = pd.DataFrame(columns=frame.columns)

    # Feature column split
    all_feature_cols = get_active_features()
    horse_feature_cols, race_feature_cols = _split_feature_cols(all_feature_cols)

    # Only keep cols that are actually present in the frame
    horse_feature_cols = [c for c in horse_feature_cols if c in frame.columns]
    race_feature_cols = [c for c in race_feature_cols if c in frame.columns]

    log.info(
        "Features — horse: %d, race: %d",
        len(horse_feature_cols),
        len(race_feature_cols),
    )

    # Fit preprocessor on train only — categorical maps and numeric mean/std
    # are computed once and applied identically to train/valid/test (and later
    # at inference via bundle.nn_preprocessor).  Calling _encode_categoricals
    # per-split was the previous behavior and produced inconsistent mappings.
    preprocessor = NNPreprocessor.fit(
        train_df, horse_feature_cols, race_feature_cols
    )
    # Capture raw odds / payoff columns BEFORE transform standardises odds_win.
    valid_raw_df = _capture_raw_outcomes(valid_df)
    test_raw_df = _capture_raw_outcomes(test_df)
    # Persist a raw, non-feature copy of 単勝 odds so RaceDataset (and the
    # log_growth betting loss) read un-standardised odds.  Not in feature_cols,
    # so NNPreprocessor.transform leaves it intact.
    for _df in (train_df, valid_df, test_df):
        if not _df.empty and "odds_win" in _df.columns:
            _df["odds_win_raw"] = _df["odds_win"]
    train_df = preprocessor.transform(train_df)
    if not valid_df.empty:
        valid_df = preprocessor.transform(valid_df)
    if not test_df.empty:
        test_df = preprocessor.transform(test_df)

    horse_feat_dim = len(horse_feature_cols)
    race_feat_dim = len(race_feature_cols)

    # 履歴系列エンコーダ (任意)。use_history=True のとき per-(race,horse) 過去走
    # トークンを leak-safe に構築し、train split から正規化を fit する。
    history_cache = None
    history_norm = None
    history_feat_dim = 0
    if use_history:
        from features.history_sequence import (
            build_history_sequences,
            fit_history_normalizer,
        )
        if prebuilt_history is not None:
            history_cache = prebuilt_history
        else:
            log.info("Building per-(race,horse) history sequences…")
            with session_scope(engine) as _session:
                history_cache = build_history_sequences(_session, max_len=history_seq_len)
        history_feat_dim = history_cache.n_features
        train_race_ids = set(train_df["race_id"].unique())
        history_norm = fit_history_normalizer(history_cache, train_race_ids)
        log.info(
            "History encoder ON — %d (race,horse) seqs, %d token features",
            len(history_cache.seqs), history_feat_dim,
        )

    # Build datasets
    train_dataset = RaceDataset(
        train_df, horse_feature_cols, race_feature_cols,
        history_cache=history_cache, history_norm=history_norm,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
    )

    val_loader = None
    if not valid_df.empty:
        val_dataset = RaceDataset(
            valid_df, horse_feature_cols, race_feature_cols,
            history_cache=history_cache, history_norm=history_norm,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
        )

    # Build model
    # n_heads must divide embed_dim evenly; clamp if needed
    effective_n_heads = n_heads
    if embed_dim % n_heads != 0:
        # Find the largest divisor of embed_dim that is <= n_heads
        effective_n_heads = max(h for h in range(1, n_heads + 1) if embed_dim % h == 0)
        log.warning(
            "embed_dim=%d not divisible by n_heads=%d; using n_heads=%d",
            embed_dim,
            n_heads,
            effective_n_heads,
        )

    horse_cat_positions, horse_cat_cardinalities = preprocessor.horse_cat_metadata()
    race_cat_positions, race_cat_cardinalities = preprocessor.race_cat_metadata()
    log.info(
        "Categorical embeddings — horse: %d cols, race: %d cols (cat_embed_dim=%d, n_transformer_layers=%d)",
        len(horse_cat_positions),
        len(race_cat_positions),
        cat_embed_dim,
        n_transformer_layers,
    )

    race_model = RaceTransformerModel(
        horse_feat_dim=horse_feat_dim,
        race_feat_dim=race_feat_dim,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_heads=effective_n_heads,
        horse_cat_positions=horse_cat_positions,
        horse_cat_cardinalities=horse_cat_cardinalities,
        race_cat_positions=race_cat_positions,
        race_cat_cardinalities=race_cat_cardinalities,
        cat_embed_dim=cat_embed_dim,
        n_transformer_layers=n_transformer_layers,
        use_history=use_history,
        history_feat_dim=history_feat_dim,
    )

    # Two-stage / fine-tuning: warm-start weights from a previously saved model.
    if init_from is not None:
        init_pt = Path(init_from) / "model.pt"
        log.info("Warm-starting weights from %s", init_pt)
        state = torch.load(init_pt, map_location="cpu")
        missing, unexpected = race_model.load_state_dict(state, strict=False)
        if missing or unexpected:
            log.warning(
                "init_from partial load — missing=%s unexpected=%s",
                list(missing), list(unexpected),
            )

    lit_module = RaceLitModule(
        model=race_model,
        loss_fn_name=loss,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        combo_bet_type=combo_bet_type,
        combo_weight=combo_weight,
    )

    # Trainer callbacks
    callbacks: list[pl.Callback] = []
    if val_loader is not None and not valid_df.empty:
        # Always log both the NDCG ranking metric and the real-odds betting ROI
        # every validation epoch, so either can be monitored / reported.
        callbacks.append(
            _NDCG3Callback(
                valid_df=valid_df,
                horse_feature_cols=horse_feature_cols,
                race_feature_cols=race_feature_cols,
                device=torch.device(device),
                history_cache=history_cache,
                history_norm=history_norm,
            )
        )
        callbacks.append(
            _WinPlaceROICallback(
                valid_df=valid_df,
                valid_raw_df=valid_raw_df,
                horse_feature_cols=horse_feature_cols,
                race_feature_cols=race_feature_cols,
                device=torch.device(device),
                history_cache=history_cache,
                history_norm=history_norm,
            )
        )
        callbacks.append(
            EarlyStopping(
                monitor=monitor,
                patience=early_stopping_patience,
                mode="max",
            )
        )
        log.info("EarlyStopping monitor: %s (mode=max)", monitor)

    torch_device = torch.device(device)
    accelerator = "gpu" if device.startswith("cuda") else "cpu"

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        gradient_clip_val=gradient_clip_val,
    )

    log.info("Starting NN training (loss=%s, max_epochs=%d)…", loss, max_epochs)
    if val_loader is not None:
        trainer.fit(lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    else:
        trainer.fit(lit_module, train_dataloaders=train_loader)

    # Move model to target device for evaluation
    race_model = race_model.to(torch_device)

    # Evaluate
    log.info("Evaluating…")
    _hist_kw = {"history_cache": history_cache, "history_norm": history_norm}
    valid_loss = _compute_loss_on_dataset(
        race_model, valid_df, horse_feature_cols, race_feature_cols, loss, torch_device,
        combo_bet_type=combo_bet_type, combo_weight=combo_weight, **_hist_kw,
    )
    test_loss = _compute_loss_on_dataset(
        race_model, test_df, horse_feature_cols, race_feature_cols, loss, torch_device,
        combo_bet_type=combo_bet_type, combo_weight=combo_weight, **_hist_kw,
    )

    valid_ndcg1 = _compute_ndcg_nn(
        race_model, valid_df, horse_feature_cols, race_feature_cols, 1, torch_device, **_hist_kw
    ) if not valid_df.empty else float("nan")
    valid_ndcg3 = _compute_ndcg_nn(
        race_model, valid_df, horse_feature_cols, race_feature_cols, 3, torch_device, **_hist_kw
    ) if not valid_df.empty else float("nan")
    test_ndcg1 = _compute_ndcg_nn(
        race_model, test_df, horse_feature_cols, race_feature_cols, 1, torch_device, **_hist_kw
    ) if not test_df.empty else float("nan")
    test_ndcg3 = _compute_ndcg_nn(
        race_model, test_df, horse_feature_cols, race_feature_cols, 3, torch_device, **_hist_kw
    ) if not test_df.empty else float("nan")

    # Real-odds top-1 単勝/複勝 ROI (deployment objective) on valid + test.
    valid_tansho_roi, valid_fukusho_roi = (
        _compute_winplace_roi_nn(
            race_model, valid_df, valid_raw_df,
            horse_feature_cols, race_feature_cols, torch_device, **_hist_kw,
        ) if not valid_df.empty else (float("nan"), float("nan"))
    )
    # test 記録を常時収集して的中率を計算 (ROI を目的にしているので評価は ROI +
    # 的中率。ndcg は最適化対象でないため参考値扱い)。return_test_bets 時のみ返す。
    test_bets: list = []
    test_tansho_roi, test_fukusho_roi = (
        _compute_winplace_roi_nn(
            race_model, test_df, test_raw_df,
            horse_feature_cols, race_feature_cols, torch_device, **_hist_kw,
            collect_records=test_bets,
        ) if not test_df.empty else (float("nan"), float("nan"))
    )
    test_tansho_hit = _hit_rate(test_bets, "tansho_ret", lambda r: bool(r["won"]))
    test_fukusho_hit = _hit_rate(test_bets, "place_ret", lambda r: r["place_ret"] > 0)

    metrics = {
        "valid_loss": valid_loss,
        "test_loss": test_loss,
        "valid_ndcg1": valid_ndcg1,
        "valid_ndcg3": valid_ndcg3,
        "test_ndcg1": test_ndcg1,
        "test_ndcg3": test_ndcg3,
        "ndcg1": test_ndcg1 if not math.isnan(test_ndcg1) else valid_ndcg1,
        "ndcg3": test_ndcg3 if not math.isnan(test_ndcg3) else valid_ndcg3,
        "valid_tansho_roi": valid_tansho_roi,
        "valid_fukusho_roi": valid_fukusho_roi,
        "test_tansho_roi": test_tansho_roi,
        "test_fukusho_roi": test_fukusho_roi,
        "test_tansho_hit": test_tansho_hit,
        "test_fukusho_hit": test_fukusho_hit,
        "monitor": monitor,
    }
    log.info("Metrics: %s", metrics)

    # ── Temperature scaling: fit per-bet-type temperature on valid set ────────
    temperature_scaler: TemperatureScaler | None = None
    if fit_temperature:
        if valid_df.empty:
            log.info("Valid set is empty — skipping temperature scaler fit.")
        else:
            log.info("Fitting TemperatureScaler on valid set (NN)…")
            try:
                temperature_scaler = _fit_temperature_scaler_nn(
                    model=race_model,
                    valid_df=valid_df,
                    raw_df=valid_raw_df,
                    horse_feature_cols=horse_feature_cols,
                    race_feature_cols=race_feature_cols,
                    device=torch_device,
                    history_cache=history_cache,
                    history_norm=history_norm,
                )
                log.info(
                    "TemperatureScaler fitted: T_win=%.3f, T_place=%.3f",
                    temperature_scaler.T_win,
                    temperature_scaler.T_place,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "TemperatureScaler fit failed: %s — proceeding without temperature scaling",
                    exc,
                )
                temperature_scaler = None

    # test_bets は metrics に含めない (meta.json 肥大回避)。return_test_bets 時のみ返す。
    _extra = {"test_bets": test_bets} if return_test_bets else {}

    # persist=False: 実験/スイープ用。モデルファイルも model_runs DB 行も書かず
    # metrics だけ返す (keiba.db を read-only に保ち、models/ と DB を汚さない)。
    if not persist:
        return {"model_dir": None, **metrics, **_extra}

    # Save model
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    model_dir = data_dir() / "models" / f"{timestamp}-nn"
    model_dir.mkdir(parents=True, exist_ok=True)

    pt_path = model_dir / "model.pt"
    torch.save(race_model.state_dict(), pt_path)

    train_range = (
        f"{train_df['date'].min()}/{train_df['date'].max()}"
        if not train_df.empty
        else None
    )
    valid_range = (
        f"{valid_df['date'].min()}/{valid_df['date'].max()}"
        if not valid_df.empty
        else None
    )
    test_range = (
        f"{test_df['date'].min()}/{test_df['date'].max()}"
        if not test_df.empty
        else None
    )

    has_temperature_scaler = temperature_scaler is not None
    if has_temperature_scaler:
        with (model_dir / "temperature_scaler.pkl").open("wb") as f:
            import pickle
            pickle.dump(temperature_scaler, f)
        log.info("temperature_scaler.pkl saved to %s", model_dir)

    preprocessor.save(model_dir / "preprocessor.pkl")
    log.info("preprocessor.pkl saved to %s", model_dir)


    meta_dict = {
        "model_type": "nn",
        "arch_version": 2,
        "loss_type": loss,
        "monitor": monitor,
        "combo_bet_type": (
            combo_bet_type
            if loss in ("combo_nll", "multi")
            else None
        ),
        "combo_weight": combo_weight if loss == "multi" else None,
        "params": {
            "hidden_dim": hidden_dim,
            "embed_dim": embed_dim,
            "n_heads": effective_n_heads,
            "n_transformer_layers": n_transformer_layers,
            "cat_embed_dim": cat_embed_dim,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "gradient_clip_val": gradient_clip_val,
            "device": device,
            "init_from": str(init_from) if init_from is not None else None,
        },
        "metrics": metrics,
        "feature_columns": all_feature_cols,
        "horse_feature_cols": horse_feature_cols,
        "race_feature_cols": race_feature_cols,
        "cat_metadata": {
            "horse_cat_positions": horse_cat_positions,
            "horse_cat_cardinalities": horse_cat_cardinalities,
            "race_cat_positions": race_cat_positions,
            "race_cat_cardinalities": race_cat_cardinalities,
        },
        "train_range": train_range,
        "valid_range": valid_range,
        "test_range": test_range,
        "has_temperature_scaler": has_temperature_scaler,
        "has_preprocessor": True,
    }

    # Delegate to registry (writes meta.json; model.pt is already in model_dir)
    save_nn_model(pt_path, meta_dict)

    log.info("Model saved to %s", model_dir)

    # Record in model_runs
    with session_scope(engine) as session:
        run = ModelRun(
            created_at=datetime.now(UTC).isoformat(),
            model_path=str(model_dir),
            params_json=json.dumps(meta_dict["params"]),
            train_range=train_range,
            valid_range=valid_range,
            metrics_json=json.dumps(metrics),
            notes="NN Set Transformer ranking model",
            is_active=0,
            model_type="nn",
        )
        session.add(run)

    log.info("model_runs row inserted.")

    return {"model_dir": str(model_dir), **metrics, **_extra}


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Train keiba-ai NN (Set Transformer); default objective is "
        "ROI-targeted (log_growth + valid_tansho_roi)."
    )
    parser.add_argument("--db", type=Path, default=None, help="Path to SQLite DB")
    parser.add_argument("--train-end", default=None, help="Training end date YYYY-MM-DD")
    parser.add_argument("--valid-months", type=int, default=12, help="Validation window (months)")
    parser.add_argument("--test-months", type=int, default=6, help="Test window (months)")
    parser.add_argument(
        "--loss",
        choices=["multi", "log_growth", "combo_nll", "plackett_luce"],
        default="multi",
        help=(
            "Loss function (default: multi = production all-markets objective: "
            "log_growth(単複 betting) + --combo-weight·combo_nll(連系 calibration)). "
            "log_growth = 単勝 fractional-Kelly return; combo_nll = 連系 calibration "
            "(proper scoring rule on the analytic-PL combo prob, folds combo "
            "calibration into the NN); plackett_luce = ranking (two-stage pretrain)."
        ),
    )
    parser.add_argument(
        "--combo-bet-type",
        choices=["馬連", "馬単", "三連複", "三連単", "all"],
        default="馬連",
        help=(
            "連系 type for combo_nll / the multi combo term (default: 馬連). "
            "'all' sums the NLL over all four combo types (slower on CUDA)."
        ),
    )
    parser.add_argument(
        "--combo-weight",
        type=float,
        default=0.01,
        help=(
            "Weight on the combo-calibration NLL term of --loss multi "
            "(default 0.01; combo_nll is ~10× the log_growth magnitude)."
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden layer size")
    parser.add_argument("--embed-dim", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument(
        "--n-transformer-layers",
        type=int,
        default=2,
        help="Number of stacked TransformerEncoderLayer blocks (default: 2)",
    )
    parser.add_argument(
        "--cat-embed-dim",
        type=int,
        default=4,
        help="Embedding size for each categorical column (default: 4)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (races per batch)")
    parser.add_argument("--max-epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--gradient-clip-val",
        type=float,
        default=1.0,
        help="Gradient norm clipping value (default: 1.0; set 0 to disable)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="EarlyStopping patience (epochs) on the --monitor metric",
    )
    parser.add_argument(
        "--monitor",
        choices=["valid_tansho_roi", "valid_fukusho_roi", "valid_ndcg3"],
        default="valid_tansho_roi",
        help=(
            "EarlyStopping / model-selection metric, all maximised "
            "(default: valid_tansho_roi). valid_tansho_roi / valid_fukusho_roi = "
            "real-odds betting return (deployment objective; pair with "
            "log_growth / multi). valid_ndcg3 = legacy ranking proxy."
        ),
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help=(
            "Warm-start weights from a saved NN model dir (two-stage fine-tuning, "
            "e.g. PL-pretrain → log_growth fine-tune).  Architecture must match."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Training device",
    )
    parser.add_argument(
        "--no-fit-temperature",
        action="store_true",
        default=False,
        help=(
            "Disable TemperatureScaler fitting after training.  Default is to fit "
            "temperature scaling on the validation set when it is non-empty."
        ),
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        default=False,
        help=(
            "Skip saving the model dir and the model_runs DB row; print metrics "
            "only.  For hyperparameter sweeps / feature A-B (keeps models/ and "
            "keiba.db untouched)."
        ),
    )
    parser.add_argument(
        "--use-history",
        action="store_true",
        default=False,
        help=(
            "Enable the per-past-race history sequence encoder (GRU) in addition "
            "to the aggregate features.  Builds leak-safe history once (slow cold)."
        ),
    )
    parser.add_argument(
        "--history-seq-len",
        type=int,
        default=15,
        help="Max past races per horse fed to the history encoder (default 15).",
    )
    args = parser.parse_args()

    result = train_nn(
        db=args.db,
        train_end=args.train_end,
        valid_months=args.valid_months,
        test_months=args.test_months,
        loss=args.loss,
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        fit_temperature=not args.no_fit_temperature,
        persist=not args.no_persist,
        n_transformer_layers=args.n_transformer_layers,
        cat_embed_dim=args.cat_embed_dim,
        weight_decay=args.weight_decay,
        gradient_clip_val=args.gradient_clip_val,
        early_stopping_patience=args.early_stopping_patience,
        monitor=args.monitor,
        init_from=args.init_from,
        combo_bet_type=args.combo_bet_type,
        combo_weight=args.combo_weight,
        use_history=args.use_history,
        history_seq_len=args.history_seq_len,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
