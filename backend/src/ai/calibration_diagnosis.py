"""Win-probability calibration diagnosis.

Run a trained model on a held-out window and quantify how far its
predicted win probabilities deviate from the actual winning rate.

Goal:
  Surface mis-calibration before/after a calibration-layer change.
  Specifically lets us confirm the user's intuition that 18-番人気の
  predicted_win_prob が実勝率より大幅に高い (over-estimation for
  longshots).

Outputs:
  - Per-prediction-rank bucket: N, mean predicted prob, actual win rate, ratio
  - Brier score (lower is better)
  - Expected Calibration Error (ECE) over 10 equal-frequency bins

CLI:
  uv run python -m ai.calibration_diagnosis \
      --model data/models/<timestamp> \
      --start 2024-10-01 --end 2024-12-31

NN models — inference happens inside
`predict_race(bundle, ...)` via bundle.model_type.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ai.predict import predict_race
from ai.registry import load_model_full
from core.logging import get_logger
from core.paths import db_path
from db.session import make_engine, session_scope
from features.builder import build_training_frame

if TYPE_CHECKING:
    from ai.registry import ModelBundle

log = get_logger(__name__)


def _per_rank_bucket(scored: pd.DataFrame) -> list[dict]:
    """Bucket scored entries by predicted-rank (1..max_runners) and return stats."""
    # groupby on an empty / column-less DataFrame raises KeyError, so guard explicitly.
    if scored.empty or "pred_rank" not in scored.columns:
        return []
    buckets: list[dict] = []
    grouped = scored.groupby("pred_rank")
    for rank, group in grouped:
        n = len(group)
        if n == 0:
            continue
        mean_pred = float(group["win_prob"].mean())
        # actual winners in this bucket: finish_position == 1
        actual_wins = int((group["finish_position"] == 1).sum())
        actual_rate = actual_wins / n
        ratio = (mean_pred / actual_rate) if actual_rate > 0 else float("inf")
        buckets.append(
            {
                "rank": int(rank),
                "n": n,
                "mean_pred_prob": round(mean_pred, 4),
                "actual_win_rate": round(actual_rate, 4),
                "ratio_pred_over_actual": round(ratio, 2) if ratio != float("inf") else None,
            }
        )
    return buckets


def _brier_score(scored: pd.DataFrame) -> float:
    """Brier score = mean((pred_prob - is_winner)^2). Lower is better."""
    is_winner = (scored["finish_position"] == 1).astype(np.float32).to_numpy()
    pred = scored["win_prob"].to_numpy(dtype=np.float32)
    return float(np.mean((pred - is_winner) ** 2))


def _expected_calibration_error(scored: pd.DataFrame, n_bins: int = 10) -> float:
    """ECE over equal-frequency bins on predicted probability."""
    pred = scored["win_prob"].to_numpy(dtype=np.float64)
    is_winner = (scored["finish_position"] == 1).astype(np.float64).to_numpy()
    n = len(pred)
    if n == 0:
        return float("nan")
    # Equal-frequency bins (quantile-based)
    sort_idx = np.argsort(pred)
    pred_sorted = pred[sort_idx]
    win_sorted = is_winner[sort_idx]
    bin_edges = np.linspace(0, n, n_bins + 1, dtype=int)
    total_err = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if hi <= lo:
            continue
        bin_mean_pred = pred_sorted[lo:hi].mean()
        bin_mean_actual = win_sorted[lo:hi].mean()
        weight = (hi - lo) / n
        total_err += weight * abs(bin_mean_pred - bin_mean_actual)
    return float(total_err)


def _score_all_races(
    bundle: ModelBundle,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Run predict_race per race and return combined long DataFrame.

    Bundle-aware: inference happens inside predict_race. The output
    schema is identical for both backends.

    Output columns: race_id, horse_id, pred_rank (1=top by score), win_prob,
                    finish_position.
    """
    out_rows: list[pd.DataFrame] = []
    for race_id, race_frame in frame.groupby("race_id"):
        if len(race_frame) < 2:
            continue
        preds = predict_race(bundle, race_frame)
        # predict_race sorts by score desc -> add pred_rank
        preds = preds.reset_index(drop=True)
        preds["pred_rank"] = preds.index + 1
        # Merge actual finish position
        actual = race_frame[["horse_id", "finish_position"]]
        merged = preds.merge(actual, on="horse_id", how="left")
        merged["race_id"] = race_id
        out_rows.append(merged)
    if not out_rows:
        return pd.DataFrame(
            columns=["race_id", "horse_id", "pred_rank", "win_prob", "finish_position"]
        )
    return pd.concat(out_rows, ignore_index=True)


def diagnose_calibration(
    model_path: Path,
    db: Path | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """Compute calibration diagnostics for a trained model.

    Args:
        model_path: Path to model directory (must contain model.txt + meta.json).
        db: Path to SQLite DB. None → default keiba.db.
        start: Inclusive evaluation start date (YYYY-MM-DD).
        end: Inclusive evaluation end date (YYYY-MM-DD).

    Returns:
        {
            "n_races": int,
            "n_entries": int,
            "rank_buckets": [...],
            "brier_score": float,
            "ece": float,
            "model_path": str,
            "window": {"start": ..., "end": ...},
        }
    """
    resolved_db = db or db_path()
    engine = make_engine(resolved_db)
    bundle = load_model_full(model_path)
    log.info("Diagnosing NN model at %s", model_path)
    log.info(
        "Building evaluation frame from %s in window %s..%s",
        resolved_db, start, end,
    )
    with session_scope(engine) as session:
        frame = build_training_frame(session, train_start=start, train_end=end)

    if frame.empty:
        log.warning("No rows in evaluation window — nothing to diagnose.")
        return {
            "n_races": 0,
            "n_entries": 0,
            "rank_buckets": [],
            "brier_score": float("nan"),
            "ece": float("nan"),
            "model_path": str(model_path),
            "window": {"start": start, "end": end},
        }

    log.info("Scoring %d entries across %d races...", len(frame), frame["race_id"].nunique())
    scored = _score_all_races(bundle, frame)

    if scored.empty:
        log.warning("Scored frame empty.")
        return {
            "n_races": 0,
            "n_entries": 0,
            "rank_buckets": [],
            "brier_score": float("nan"),
            "ece": float("nan"),
            "model_path": str(model_path),
            "window": {"start": start, "end": end},
        }

    # Drop entries with no finish_position (DNF / scratched)
    scored_finished = scored.dropna(subset=["finish_position"])

    return {
        "n_races": int(scored_finished["race_id"].nunique()),
        "n_entries": int(len(scored_finished)),
        "rank_buckets": _per_rank_bucket(scored_finished),
        "brier_score": round(_brier_score(scored_finished), 4),
        "ece": round(_expected_calibration_error(scored_finished), 4),
        "model_path": str(model_path),
        "window": {"start": start, "end": end},
    }


def _format_report(result: dict) -> str:
    """Render diagnose_calibration result as a human-readable text report."""
    lines = []
    lines.append("=== Win-probability calibration diagnosis ===")
    lines.append(f"model:    {result['model_path']}")
    lines.append(
        f"window:   {result['window']['start']} 〜 {result['window']['end']}"
        f"  ({result['n_races']} races, {result['n_entries']} entries)"
    )
    lines.append("")
    lines.append(
        f"{'pred_rank':>10}  {'N':>6}  {'mean_pred':>10}  "
        f"{'actual_rate':>12}  {'ratio':>10}"
    )
    lines.append("-" * 60)
    for b in result["rank_buckets"]:
        ratio_str = (
            f"{b['ratio_pred_over_actual']:.2f}x"
            if b["ratio_pred_over_actual"] is not None
            else "inf (zero actual)"
        )
        lines.append(
            f"{b['rank']:>10}  {b['n']:>6}  "
            f"{b['mean_pred_prob']:>10.4f}  {b['actual_win_rate']:>12.4f}  {ratio_str:>10}"
        )
    lines.append("")
    lines.append(f"Brier score: {result['brier_score']}")
    lines.append(f"ECE (10-bin equal-frequency): {result['ece']}")
    return "\n".join(lines)


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose win-probability calibration of a trained model."
    )
    parser.add_argument("--model", type=Path, required=True, help="Model directory")
    parser.add_argument("--db", type=Path, default=None, help="Path to SQLite DB")
    parser.add_argument("--start", default=None, help="Eval start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Eval end date YYYY-MM-DD")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit raw JSON instead of formatted report",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    result = diagnose_calibration(
        model_path=args.model,
        db=args.db,
        start=args.start,
        end=args.end,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(_format_report(result))


if __name__ == "__main__":
    _cli()
