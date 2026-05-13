"""CLI: Train a PyTorch NN (Set Transformer) ranking model.

Usage:
    python -m keiba_ai.ai.nn.train_nn \\
        --train-end YYYY-MM-DD --valid-months 12 --test-months 6 \\
        --loss {plackett_luce,listmle,time_margin} \\
        --hidden-dim 64 --embed-dim 32 --n-heads 4 \\
        --batch-size 32 --max-epochs 100 --learning-rate 1e-3 \\
        --device {cpu,cuda} \\
        --db PATH
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping

from keiba_ai.ai.labels import assign_relevance
from keiba_ai.ai.nn.dataset import RaceDataset, collate_fn
from keiba_ai.ai.nn.loss import listmle_loss, plackett_luce_loss, time_margin_loss
from keiba_ai.ai.nn.model import RaceModel
from keiba_ai.ai.registry import save_nn_model
from keiba_ai.ai.splits import time_split
from keiba_ai.ai.temperature import TemperatureScaler
from keiba_ai.core.paths import data_dir, db_path
from keiba_ai.db.models.model_run import ModelRun
from keiba_ai.db.session import make_engine, session_scope
from keiba_ai.features.builder import CATEGORICAL_FEATURES, build_training_frame, get_active_features

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
    """Label-encode categorical string columns in-place so they become float32-compatible.

    Each unique string value is mapped to a consecutive integer.  NaN is mapped to -1.
    Numeric columns are coerced to float32 and NaN-filled with 0.

    This is intentionally simple: the model treats these as ordinary continuous inputs.
    For production quality, learned embeddings would be preferable, but label-encoding
    is sufficient for the initial training pipeline.

    Args:
        frame: DataFrame containing at least the columns listed in feature_cols.
        feature_cols: Columns to ensure are numeric.

    Returns:
        A copy of frame with the listed columns converted to float32.
    """
    frame = frame.copy()
    cat_set = set(CATEGORICAL_FEATURES)

    for col in feature_cols:
        if col not in frame.columns:
            frame[col] = 0.0
            continue
        if col in cat_set or frame[col].dtype == object:
            # Label-encode: NaN → -1, unique values → 0, 1, 2, ...
            unique_vals = [v for v in frame[col].dropna().unique()]
            mapping = {v: float(i) for i, v in enumerate(sorted(unique_vals, key=str))}
            frame[col] = frame[col].map(mapping).fillna(-1.0)
        else:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    return frame


def _compute_ndcg_nn(
    model: RaceModel,
    frame: pd.DataFrame,
    horse_feature_cols: list[str],
    race_feature_cols: list[str],
    at: int,
    device: torch.device,
) -> float:
    """Compute NDCG@at for the NN model across all races in frame."""
    if frame.empty:
        return float("nan")

    from sklearn.metrics import ndcg_score

    frame = frame.copy()
    frame["relevance"] = frame["finish_position"].map(assign_relevance)

    # RaceDataset groups by race_id (sort=True) → same order as groupby below
    dataset = RaceDataset(frame, horse_feature_cols, race_feature_cols)

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

            scores = model(hf, rf, mask_single)  # [1, n]
            race_scores.append(scores[0, :n].cpu().tolist())

    ndcg_vals: list[float] = []
    for (_, grp), scores in zip(frame.groupby("race_id", sort=True), race_scores):
        if len(grp) < 2:
            continue
        true_rel = grp["relevance"].values.reshape(1, -1)
        pred_sc = np.array(scores).reshape(1, -1)
        ndcg_vals.append(float(ndcg_score(true_rel, pred_sc, k=at)))

    return float(np.mean(ndcg_vals)) if ndcg_vals else float("nan")


def _fit_temperature_scaler_nn(
    model: RaceModel,
    valid_df: pd.DataFrame,
    horse_feature_cols: list[str],
    race_feature_cols: list[str],
    device: torch.device,
) -> TemperatureScaler:
    """Fit TemperatureScaler on valid_df using payback-maximising grid search.

    Per-race scores are extracted by running the NN in eval mode through
    RaceDataset (same indexing order as groupby("race_id", sort=True)).

    Args:
        model: Trained RaceModel (already in eval mode).
        valid_df: Validation DataFrame with finish_position, odds_win, payout_place.
        horse_feature_cols: Horse-level feature columns.
        race_feature_cols: Race-level feature columns.
        device: Torch device for inference.

    Returns:
        Fitted TemperatureScaler.
    """
    dataset = RaceDataset(valid_df, horse_feature_cols, race_feature_cols)

    scores_per_race: list[np.ndarray] = []
    finish_positions_per_race: list[np.ndarray] = []
    odds_win_per_race: list[np.ndarray] = []
    payout_place_per_race: list = []

    model.eval()
    # Iterate dataset by integer index — same groupby("race_id", sort=True) order
    # as RaceDataset._races, so grp and dataset[i] are always aligned.
    with torch.no_grad():
        for i, (_race_id, grp) in enumerate(valid_df.groupby("race_id", sort=True)):
            if len(grp) < 2:
                continue

            sample = dataset[i]
            n = sample["n_horses"]
            hf = sample["horse_features"].unsqueeze(0).to(device)
            rf = sample["race_features"].unsqueeze(0).to(device)
            mask_single = torch.zeros(1, n, dtype=torch.bool, device=device)
            mask_single[0, :n] = True

            raw_scores = model(hf, rf, mask_single)  # [1, n]
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
    model: RaceModel,
    frame: pd.DataFrame,
    horse_feature_cols: list[str],
    race_feature_cols: list[str],
    loss_fn_name: str,
    device: torch.device,
) -> float:
    """Compute mean loss over all races in frame (for test-set reporting)."""
    if frame.empty:
        return float("nan")

    dataset = RaceDataset(frame, horse_feature_cols, race_feature_cols)
    if len(dataset) == 0:
        return float("nan")

    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, shuffle=False)

    loss_fn = _build_loss_fn(loss_fn_name)

    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            hf = batch["horse_features"].to(device)
            rf = batch["race_features"].to(device)
            fp = batch["finish_positions"].to(device)
            ft = batch["finish_times"].to(device)
            mask = batch["mask"].to(device)

            scores = model(hf, rf, mask)

            if loss_fn_name == "time_margin":
                loss = loss_fn(scores, fp, ft, mask)
            else:
                loss = loss_fn(scores, fp, mask)

            if not torch.isnan(loss):
                total_loss += loss.item()
                n_batches += 1

    return total_loss / n_batches if n_batches > 0 else float("nan")


def _build_loss_fn(loss_name: str):
    """Return the loss callable for the given name."""
    if loss_name == "plackett_luce":
        return plackett_luce_loss
    if loss_name == "listmle":
        return listmle_loss
    if loss_name == "time_margin":
        return time_margin_loss
    raise ValueError(f"Unknown loss: {loss_name!r}. Choose from plackett_luce, listmle, time_margin")


class RaceLitModule(pl.LightningModule):
    """Lightning wrapper around RaceModel.

    Args:
        model: RaceModel instance
        loss_fn_name: one of "plackett_luce", "listmle", "time_margin"
        learning_rate: Adam optimizer learning rate
    """

    def __init__(
        self,
        model: RaceModel,
        loss_fn_name: str,
        learning_rate: float,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn_name = loss_fn_name
        self.loss_fn = _build_loss_fn(loss_fn_name)
        self.learning_rate = learning_rate

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        scores = self.model(
            batch["horse_features"],
            batch["race_features"],
            batch["mask"],
        )
        if self.loss_fn_name == "time_margin":
            loss = self.loss_fn(
                scores,
                batch["finish_positions"],
                batch["finish_times"],
                batch["mask"],
            )
        else:
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
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def train_nn(
    db: Path | None = None,
    train_end: str | None = None,
    valid_months: int = 12,
    test_months: int = 6,
    loss: str = "plackett_luce",
    hidden_dim: int = 64,
    embed_dim: int = 32,
    n_heads: int = 4,
    batch_size: int = 32,
    max_epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    fit_temperature: bool = True,
) -> dict:
    """Run the full NN training pipeline. Returns metrics dict."""
    resolved_db = db or db_path()
    engine = make_engine(resolved_db)

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

    # Encode categorical string columns to numeric so torch.tensor() can consume them.
    # Encoding is computed from the train set to avoid leakage, then applied to all splits.
    all_feat_for_encoding = horse_feature_cols + race_feature_cols
    train_df = _encode_categoricals(train_df, all_feat_for_encoding)
    if not valid_df.empty:
        valid_df = _encode_categoricals(valid_df, all_feat_for_encoding)
    if not test_df.empty:
        test_df = _encode_categoricals(test_df, all_feat_for_encoding)

    horse_feat_dim = len(horse_feature_cols)
    race_feat_dim = len(race_feature_cols)

    # Build datasets
    train_dataset = RaceDataset(train_df, horse_feature_cols, race_feature_cols)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
    )

    val_loader = None
    if not valid_df.empty:
        val_dataset = RaceDataset(valid_df, horse_feature_cols, race_feature_cols)
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

    race_model = RaceModel(
        horse_feat_dim=horse_feat_dim,
        race_feat_dim=race_feat_dim,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_heads=effective_n_heads,
    )

    lit_module = RaceLitModule(
        model=race_model,
        loss_fn_name=loss,
        learning_rate=learning_rate,
    )

    # Trainer callbacks
    callbacks = []
    if val_loader is not None:
        callbacks.append(EarlyStopping(monitor="valid_loss", patience=10, mode="min"))

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
    valid_loss = _compute_loss_on_dataset(
        race_model, valid_df, horse_feature_cols, race_feature_cols, loss, torch_device
    )
    test_loss = _compute_loss_on_dataset(
        race_model, test_df, horse_feature_cols, race_feature_cols, loss, torch_device
    )

    valid_ndcg1 = _compute_ndcg_nn(
        race_model, valid_df, horse_feature_cols, race_feature_cols, 1, torch_device
    ) if not valid_df.empty else float("nan")
    valid_ndcg3 = _compute_ndcg_nn(
        race_model, valid_df, horse_feature_cols, race_feature_cols, 3, torch_device
    ) if not valid_df.empty else float("nan")
    test_ndcg1 = _compute_ndcg_nn(
        race_model, test_df, horse_feature_cols, race_feature_cols, 1, torch_device
    ) if not test_df.empty else float("nan")
    test_ndcg3 = _compute_ndcg_nn(
        race_model, test_df, horse_feature_cols, race_feature_cols, 3, torch_device
    ) if not test_df.empty else float("nan")

    metrics = {
        "valid_loss": valid_loss,
        "test_loss": test_loss,
        "valid_ndcg1": valid_ndcg1,
        "valid_ndcg3": valid_ndcg3,
        "test_ndcg1": test_ndcg1,
        "test_ndcg3": test_ndcg3,
        "ndcg1": test_ndcg1 if not math.isnan(test_ndcg1) else valid_ndcg1,
        "ndcg3": test_ndcg3 if not math.isnan(test_ndcg3) else valid_ndcg3,
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
                    horse_feature_cols=horse_feature_cols,
                    race_feature_cols=race_feature_cols,
                    device=torch_device,
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

    meta_dict = {
        "model_type": "nn",
        "loss_type": loss,
        "params": {
            "hidden_dim": hidden_dim,
            "embed_dim": embed_dim,
            "n_heads": effective_n_heads,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "learning_rate": learning_rate,
            "device": device,
        },
        "metrics": metrics,
        "feature_columns": all_feature_cols,
        "horse_feature_cols": horse_feature_cols,
        "race_feature_cols": race_feature_cols,
        "train_range": train_range,
        "valid_range": valid_range,
        "test_range": test_range,
        "has_temperature_scaler": has_temperature_scaler,
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

    return {"model_dir": str(model_dir), **metrics}


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train keiba-ai NN (Set Transformer) ranking model")
    parser.add_argument("--db", type=Path, default=None, help="Path to SQLite DB")
    parser.add_argument("--train-end", default=None, help="Training end date YYYY-MM-DD")
    parser.add_argument("--valid-months", type=int, default=12, help="Validation window (months)")
    parser.add_argument("--test-months", type=int, default=6, help="Test window (months)")
    parser.add_argument(
        "--loss",
        choices=["plackett_luce", "listmle", "time_margin"],
        default="plackett_luce",
        help="Ranking loss function",
    )
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden layer size")
    parser.add_argument("--embed-dim", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (races per batch)")
    parser.add_argument("--max-epochs", type=int, default=100, help="Maximum training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
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
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
