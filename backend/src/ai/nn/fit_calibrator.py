"""Post-hoc isotonic calibration for an already-saved NN model.

Fits an IsotonicCalibrator on (NN_win_prob, is_winner) pairs collected from a
held-out window (typically the same valid_range used during training) and
writes nn_calibrator.pkl into the model directory so that subsequent
predict_race / load_model_full calls automatically apply the calibration.

Use case:
  An NN was trained without a calibration head and shows systematic mis-
  calibration on the diagnostic CLI (e.g. pred_rank 1 under-confidence,
  pred_rank 10+ over-confidence). This tool fixes that without retraining.

CLI:
  uv run python -m ai.nn.fit_calibrator \\
      --model data/models/<timestamp>-nn \\
      [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--force]

When --start / --end are omitted, the valid_range from meta.json is used.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np

from ai.calibrate import IsotonicCalibrator
from ai.predict import predict_race
from ai.registry import load_model_full
from core.paths import db_path
from db.session import make_engine, session_scope
from features.builder import build_training_frame

log = logging.getLogger(__name__)


def _parse_range(meta: dict) -> tuple[str | None, str | None]:
    """meta.json の valid_range ('start/end') を (start, end) に分解する。"""
    raw = meta.get("valid_range")
    if not raw or "/" not in raw:
        return None, None
    start, end = raw.split("/", 1)
    return start.strip() or None, end.strip() or None


def _collect_predictions(
    model_path: Path,
    db: Path | None,
    start: str | None,
    end: str | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run predict_race on every race in the window and return flat arrays.

    The bundle's nn_calibrator is forcibly cleared during this pass so that
    the collected probabilities are the raw (temperature-scaled but
    uncalibrated) outputs the IsotonicCalibrator should learn to correct.

    Returns:
        (win_probs, is_winner, n_races)
        win_probs and is_winner are 1-D arrays of equal length (one entry per
        evaluated horse). n_races is the number of races used.
    """
    resolved_db = db or db_path()
    engine = make_engine(resolved_db)
    bundle = load_model_full(model_path)
    if bundle.model_type != "nn":
        raise ValueError(
            f"--model must point to a NN model directory; got model_type={bundle.model_type}"
        )
    # Re-fit must operate on uncalibrated NN output so we strip any existing one.
    bundle.nn_calibrator = None

    log.info("Building feature frame from %s in window %s..%s", resolved_db, start, end)
    with session_scope(engine) as session:
        frame = build_training_frame(session, train_start=start, train_end=end)

    if frame.empty:
        raise RuntimeError(
            "No rows in evaluation window — cannot fit calibrator. "
            "Check --start / --end against the DB date range."
        )

    win_chunks: list[np.ndarray] = []
    win_chunks_outcome: list[np.ndarray] = []
    n_races = 0
    for _race_id, race_frame in frame.groupby("race_id"):
        if len(race_frame) < 2:
            continue
        preds = predict_race(bundle, race_frame)
        actual = race_frame[["horse_id", "finish_position"]]
        merged = preds.merge(actual, on="horse_id", how="left")
        # Drop rows with missing finish_position (DNF / scratched)
        merged = merged.dropna(subset=["finish_position"])
        if merged.empty:
            continue
        win_chunks.append(merged["win_prob"].to_numpy(dtype=np.float64))
        win_chunks_outcome.append(
            (merged["finish_position"] == 1).astype(np.float64).to_numpy()
        )
        n_races += 1

    if not win_chunks:
        raise RuntimeError("No usable races after filtering — cannot fit calibrator.")

    return (
        np.concatenate(win_chunks),
        np.concatenate(win_chunks_outcome),
        n_races,
    )


def _brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probs - outcomes) ** 2))


def _ece(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> float:
    """Equal-frequency ECE (matches ai.calibration_diagnosis)."""
    n = len(probs)
    if n == 0:
        return float("nan")
    sort_idx = np.argsort(probs)
    p_sorted = probs[sort_idx]
    o_sorted = outcomes[sort_idx]
    bin_edges = np.linspace(0, n, n_bins + 1, dtype=int)
    total = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if hi <= lo:
            continue
        weight = (hi - lo) / n
        total += weight * abs(p_sorted[lo:hi].mean() - o_sorted[lo:hi].mean())
    return float(total)


def fit_and_save(
    model_path: Path,
    db: Path | None = None,
    start: str | None = None,
    end: str | None = None,
    force: bool = False,
) -> dict:
    """Fit + persist nn_calibrator.pkl. Returns before/after diagnostics."""
    target = Path(model_path) / "nn_calibrator.pkl"
    if target.exists() and not force:
        raise FileExistsError(
            f"{target} already exists. Re-run with --force to overwrite."
        )

    meta_path = Path(model_path) / "meta.json"
    bundle_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if start is None and end is None:
        start, end = _parse_range(bundle_meta)
        if start is None:
            raise ValueError(
                "meta.json has no valid_range and no --start/--end was given."
            )

    log.info("Fit window: %s .. %s", start, end)
    raw_probs, outcomes, n_races = _collect_predictions(
        model_path=Path(model_path), db=db, start=start, end=end
    )
    log.info(
        "Collected %d (raw_prob, outcome) pairs across %d races", len(raw_probs), n_races
    )

    before_brier = _brier(raw_probs, outcomes)
    before_ece = _ece(raw_probs, outcomes)
    log.info("Before calibration: Brier=%.4f, ECE=%.4f", before_brier, before_ece)

    calibrator = IsotonicCalibrator()
    calibrator.fit(raw_probs, outcomes)

    # In-sample diagnostics (no race-level re-normalisation; matches the
    # _ece convention from ai.calibration_diagnosis on flat arrays).
    after_probs = calibrator.iso.predict(raw_probs)
    after_brier = _brier(after_probs, outcomes)
    after_ece = _ece(after_probs, outcomes)
    log.info("After  calibration: Brier=%.4f, ECE=%.4f", after_brier, after_ece)

    with target.open("wb") as f:
        pickle.dump(calibrator, f)
    log.info("Saved calibrator to %s", target)

    return {
        "model_path": str(model_path),
        "n_races": n_races,
        "n_entries": int(len(raw_probs)),
        "window": {"start": start, "end": end},
        "before": {"brier": before_brier, "ece": before_ece},
        "after": {"brier": after_brier, "ece": after_ece},
        "saved_to": str(target),
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a post-hoc IsotonicCalibrator on an NN model and save it alongside."
    )
    parser.add_argument("--model", type=Path, required=True, help="NN model directory")
    parser.add_argument("--db", type=Path, default=None, help="Path to SQLite DB")
    parser.add_argument("--start", default=None, help="Fit window start YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Fit window end YYYY-MM-DD")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing nn_calibrator.pkl.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    result = fit_and_save(
        model_path=args.model,
        db=args.db,
        start=args.start,
        end=args.end,
        force=args.force,
    )
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
