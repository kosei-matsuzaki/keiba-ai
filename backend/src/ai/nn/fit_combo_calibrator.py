"""Post-hoc combo (連系) calibration for an already-saved NN model.

Runs ``fit_combo_calibrators_bundle`` against an existing NN bundle on a held-
out window (typically the same valid_range used during training) and writes
combo_calibrators.pkl into the model directory so that subsequent
predict_race_with_combinations / load_model_full calls automatically apply
per-bet-type isotonic correction to PL Monte Carlo combo probabilities.

Use case:
  combo_calibration_diagnosis reveals systematic over-estimation on connected
  bet types (馬連/ワイド/馬単/三連複/三連単). The win_prob calibrator alone
  doesn't address this because combos are derived from raw PL samples.

CLI:
  uv run python -m ai.nn.fit_combo_calibrator \\
      --model data/models/<timestamp>-nn \\
      [--start YYYY-MM-DD] [--end YYYY-MM-DD] \\
      [--n-samples 5000] [--conditional] [--force]

When --start / --end are omitted, valid_range from meta.json is used.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

from ai.calibrate import fit_combo_calibrators_bundle
from ai.registry import load_model_full
from core.bet_types import RENKEI_BET_TYPES
from core.paths import db_path
from db.session import make_engine, session_scope
from features.builder import build_training_frame

log = logging.getLogger(__name__)


def _parse_range(meta: dict) -> tuple[str | None, str | None]:
    raw = meta.get("valid_range")
    if not raw or "/" not in raw:
        return None, None
    start, end = raw.split("/", 1)
    return start.strip() or None, end.strip() or None


def fit_and_save(
    model_path: Path,
    db: Path | None = None,
    start: str | None = None,
    end: str | None = None,
    n_samples: int = 5_000,
    use_conditional: bool = False,
    force: bool = False,
) -> dict:
    """Fit + persist combo_calibrators.pkl. Returns a summary dict."""
    target = Path(model_path) / "combo_calibrators.pkl"
    if target.exists() and not force:
        raise FileExistsError(
            f"{target} already exists. Re-run with --force to overwrite."
        )

    meta_path = Path(model_path) / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if start is None and end is None:
        start, end = _parse_range(meta)
        if start is None:
            raise ValueError(
                "meta.json has no valid_range and no --start/--end was given."
            )
    log.info("Fit window: %s .. %s", start, end)

    resolved_db = db or db_path()
    engine = make_engine(resolved_db)
    bundle = load_model_full(Path(model_path))
    # Combo calibration must run on raw PL output, so clear any existing combo
    # calibrators while we collect fit data.
    bundle.combo_calibrators = None

    log.info("Building feature frame from %s", resolved_db)
    with session_scope(engine) as session:
        valid_frame = build_training_frame(session, train_start=start, train_end=end)

    if valid_frame.empty:
        raise RuntimeError(
            "No rows in fit window — check --start / --end against the DB date range."
        )

    log.info(
        "Fitting combo calibrators on %d races (n_samples=%d, use_conditional=%s)",
        valid_frame["race_id"].nunique(),
        n_samples,
        use_conditional,
    )
    cal = fit_combo_calibrators_bundle(
        valid_frame=valid_frame,
        bundle=bundle,
        n_samples=n_samples,
        use_conditional=use_conditional,
    )

    with target.open("wb") as f:
        pickle.dump(cal, f)
    log.info("Saved combo calibrators to %s", target)

    # Report which bet types were actually fitted (sample threshold inside fit).
    fitted: dict[str, bool] = {bt: cal.has(bt) for bt in RENKEI_BET_TYPES}

    return {
        "model_path": str(model_path),
        "n_races": int(valid_frame["race_id"].nunique()),
        "window": {"start": start, "end": end},
        "use_conditional": bool(use_conditional),
        "fitted_bet_types": fitted,
        "saved_to": str(target),
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Fit ComboCalibrators on an NN model and save them alongside.",
    )
    parser.add_argument("--model", type=Path, required=True, help="NN model directory")
    parser.add_argument("--db", type=Path, default=None, help="Path to SQLite DB")
    parser.add_argument("--start", default=None, help="Fit window start YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Fit window end YYYY-MM-DD")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5_000,
        help="MC samples for the per-race combo prediction. Default 5000.",
    )
    parser.add_argument(
        "--conditional",
        action="store_true",
        help="Use ConditionalIsotonicCalibrator (surface × n_runners bins) per bet type.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing combo_calibrators.pkl.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    result = fit_and_save(
        model_path=args.model,
        db=args.db,
        start=args.start,
        end=args.end,
        n_samples=args.n_samples,
        use_conditional=args.conditional,
        force=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
