"""CLI: Train a LightGBM lambdarank model and register it.

Usage:
    uv run python -m keiba_ai.ai.train [--train-end YYYY-MM-DD]
                                        [--valid-months 12]
                                        [--test-months 6]
                                        [--db PATH]
                                        [--params-json PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from keiba_ai.ai.calibrate import IsotonicCalibrator
from keiba_ai.ai.labels import assign_is_winner, assign_relevance
from keiba_ai.ai.registry import save_model
from keiba_ai.ai.splits import time_split
from keiba_ai.core.paths import db_path
from keiba_ai.db.models import ModelRun  # noqa: F401 — register table with Base
from keiba_ai.db.session import make_engine, session_scope
from keiba_ai.features.builder import (
    CATEGORICAL_FEATURES,
    build_training_frame,
    get_active_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_PARAMS: dict = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3],
    "lambdarank_truncation_level": 3,
    "num_leaves": 63,
    "learning_rate": 0.05,
    "min_data_in_leaf": 5,  # lowered from 50 to work with small synthetic data
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}


def _make_lgb_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    reference: lgb.Dataset | None = None,
) -> lgb.Dataset:
    """Build a LightGBM Dataset with group counts by race_id."""
    X = df[feature_cols].copy()
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype("category")

    y = df["relevance"].values.astype(np.float32)

    # group = count of entries per race, in the order they appear in df
    group = df.groupby("race_id", sort=False)["horse_id"].count().values

    return lgb.Dataset(
        X,
        label=y,
        group=group,
        feature_name=feature_cols,
        categorical_feature=categorical_cols,
        reference=reference,
        free_raw_data=False,
    )


def _compute_ndcg(model: lgb.Booster, df: pd.DataFrame, at: int) -> float:
    """Compute NDCG@at across all races in df."""
    if df.empty:
        return float("nan")

    # 学習時に使った特徴量を model から復元する（env flag 経由で odds 抜きに
    # した場合でも model.feature_name() が正しい列名を返す）
    feature_cols = list(model.feature_name())
    X = df[feature_cols].copy()
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")

    scores = model.predict(X)
    df = df.copy()
    df["_score"] = scores

    ndcg_vals: list[float] = []
    for _race_id, grp in df.groupby("race_id"):
        if len(grp) < 2:
            continue
        true_rel = grp["relevance"].values.reshape(1, -1)
        pred_scores = grp["_score"].values.reshape(1, -1)
        ndcg_vals.append(float(ndcg_score(true_rel, pred_scores, k=at)))

    return float(np.mean(ndcg_vals)) if ndcg_vals else float("nan")


def _train_binary_classifier_and_calibrator(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    base_params: dict,
) -> tuple[lgb.Booster, IsotonicCalibrator, dict]:
    """Train a binary classifier (objective=binary) on is_winner and a
    post-hoc isotonic calibrator on the validation set.

    The binary classifier outputs sigmoid-calibrated raw probabilities.
    The isotonic regression then corrects any residual systematic bias by
    learning monotonic mapping (raw_prob → empirical win rate) on valid set.

    Returns:
        (binary_model, calibrator, metrics)
        metrics keys: binary_logloss, binary_brier, binary_calibrated_brier
    """
    binary_params = {
        "objective": "binary",
        "metric": ["binary_logloss"],
        "learning_rate": base_params.get("learning_rate", 0.05),
        "num_leaves": base_params.get("num_leaves", 63),
        "min_data_in_leaf": base_params.get("min_data_in_leaf", 50),
        "feature_fraction": base_params.get("feature_fraction", 0.9),
        "bagging_fraction": base_params.get("bagging_fraction", 0.8),
        "bagging_freq": base_params.get("bagging_freq", 5),
        "verbose": -1,
    }

    # Build datasets with is_winner labels
    train_y_bin = train_df["finish_position"].map(assign_is_winner).values.astype(np.float32)
    train_X = train_df[feature_cols].copy()
    for col in CATEGORICAL_FEATURES:
        if col in train_X.columns:
            train_X[col] = train_X[col].astype("category")

    binary_train_data = lgb.Dataset(
        train_X,
        label=train_y_bin,
        categorical_feature=[
            c for c in CATEGORICAL_FEATURES if c in train_X.columns
        ] or None,
        free_raw_data=False,
    )

    binary_valid_sets = [binary_train_data]
    binary_valid_names = ["train"]
    binary_callbacks = [lgb.log_evaluation(period=50)]

    if not valid_df.empty:
        valid_y_bin = valid_df["finish_position"].map(assign_is_winner).values.astype(np.float32)
        valid_X = valid_df[feature_cols].copy()
        for col in CATEGORICAL_FEATURES:
            if col in valid_X.columns:
                valid_X[col] = valid_X[col].astype("category")
        binary_valid_data = lgb.Dataset(
            valid_X,
            label=valid_y_bin,
            categorical_feature=[
                c for c in CATEGORICAL_FEATURES if c in valid_X.columns
            ] or None,
            reference=binary_train_data,
            free_raw_data=False,
        )
        binary_valid_sets.append(binary_valid_data)
        binary_valid_names.append("valid")
        binary_callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=False))

    log.info("Training binary classifier (is_winner head)…")
    binary_model = lgb.train(
        binary_params,
        binary_train_data,
        num_boost_round=300,
        valid_sets=binary_valid_sets,
        valid_names=binary_valid_names,
        callbacks=binary_callbacks,
    )

    # Fit isotonic calibrator on the validation set if we have one,
    # otherwise on training set (less ideal but still better than nothing).
    calibrator = IsotonicCalibrator()
    metrics: dict[str, float] = {}

    if not valid_df.empty:
        valid_X_cal = valid_df[feature_cols].copy()
        for col in CATEGORICAL_FEATURES:
            if col in valid_X_cal.columns:
                valid_X_cal[col] = valid_X_cal[col].astype("category")
        valid_raw = binary_model.predict(valid_X_cal)
        valid_y = valid_df["finish_position"].map(assign_is_winner).values.astype(np.float32)
        calibrator.fit(valid_raw, valid_y)

        # Diagnostic metrics on valid (uncalibrated and calibrated)
        valid_calibrated = calibrator.predict(valid_raw, normalise=False)
        metrics["binary_brier"] = float(np.mean((valid_raw - valid_y) ** 2))
        metrics["binary_calibrated_brier"] = float(
            np.mean((valid_calibrated - valid_y) ** 2)
        )
    else:
        # Fallback to training set
        log.warning("Valid set empty — fitting calibrator on training set (overfit risk).")
        train_raw = binary_model.predict(train_X)
        calibrator.fit(train_raw, train_y_bin)
        train_calibrated = calibrator.predict(train_raw, normalise=False)
        metrics["binary_brier"] = float(np.mean((train_raw - train_y_bin) ** 2))
        metrics["binary_calibrated_brier"] = float(
            np.mean((train_calibrated - train_y_bin) ** 2)
        )

    log.info(
        "Binary classifier metrics: brier=%.4f, calibrated_brier=%.4f",
        metrics["binary_brier"],
        metrics["binary_calibrated_brier"],
    )
    return binary_model, calibrator, metrics


def train(
    db: Path | None = None,
    train_end: str | None = None,
    valid_months: int = 12,
    test_months: int = 6,
    params_json: Path | None = None,
) -> dict:
    """Run the full training pipeline. Returns metrics dict."""
    resolved_db = db or db_path()
    engine = make_engine(resolved_db)

    params = dict(DEFAULT_PARAMS)
    if params_json:
        params.update(json.loads(Path(params_json).read_text()))

    log.info("Building feature frame from %s", resolved_db)
    with session_scope(engine) as session:
        frame = build_training_frame(session)

    if frame.empty:
        raise RuntimeError("No training data found in the database.")

    frame["relevance"] = frame["finish_position"].map(assign_relevance)

    log.info("Total rows: %d | Races: %d", len(frame), frame["race_id"].nunique())
    train_df, valid_df, test_df = time_split(frame, train_end, valid_months, test_months)
    log.info(
        "Split → train=%d rows, valid=%d rows, test=%d rows",
        len(train_df),
        len(valid_df),
        len(test_df),
    )

    if train_df.empty:
        # Truly no training data after split — fall back to using everything.
        # This still leaks any test rows into training, but at that point the
        # split was so degenerate that we have nothing else to learn from.
        log.warning(
            "Train set is empty — using full frame for training (test will leak; "
            "consider widening the split window)."
        )
        train_df = frame.copy()
        valid_df = pd.DataFrame(columns=frame.columns)
    elif valid_df.empty:
        # Valid is just an early-stopping helper. Skipping it must NOT pull test
        # rows into training (that would silently leak test → 1.0 NDCG).
        log.info("Valid set is empty — proceeding without early stopping.")

    # 学習で使う特徴量列。KEIBA_EXCLUDE_ODDS_FEATURES=1 で 5 つの odds 派生を除外
    feature_cols = get_active_features()
    log.info(
        "Training with %d features (KEIBA_EXCLUDE_ODDS_FEATURES=%s)",
        len(feature_cols),
        os.environ.get("KEIBA_EXCLUDE_ODDS_FEATURES", "0"),
    )

    train_data = _make_lgb_dataset(train_df, feature_cols, CATEGORICAL_FEATURES)

    callbacks = [lgb.log_evaluation(period=50)]
    valid_sets: list[lgb.Dataset] = [train_data]
    valid_names: list[str] = ["train"]

    if not valid_df.empty:
        valid_data = _make_lgb_dataset(
            valid_df, feature_cols, CATEGORICAL_FEATURES, reference=train_data
        )
        valid_sets.append(valid_data)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=False))

    log.info("Starting LightGBM training…")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    # Evaluate on valid and test
    valid_ndcg1 = _compute_ndcg(model, valid_df, 1) if not valid_df.empty else float("nan")
    valid_ndcg3 = _compute_ndcg(model, valid_df, 3) if not valid_df.empty else float("nan")
    test_ndcg1 = _compute_ndcg(model, test_df, 1) if not test_df.empty else float("nan")
    test_ndcg3 = _compute_ndcg(model, test_df, 3) if not test_df.empty else float("nan")

    metrics = {
        "valid_ndcg1": valid_ndcg1,
        "valid_ndcg3": valid_ndcg3,
        "test_ndcg1": test_ndcg1,
        "test_ndcg3": test_ndcg3,
    }
    log.info("Lambdarank metrics: %s", metrics)

    # ── Phase 2: binary classifier + isotonic calibrator ──────────────────────
    # 既存の lambdarank score (順位用) に加えて、別 head として binary classifier
    # を学習し isotonic で post-hoc 補正する。これにより推論時の win_prob は
    # softmax(lambdarank scores) ではなく真の確率に近い値を返せるようになる。
    binary_model, calibrator, binary_metrics = _train_binary_classifier_and_calibrator(
        train_df, valid_df, feature_cols, params
    )
    metrics.update(binary_metrics)
    log.info("All metrics: %s", metrics)

    # Determine date ranges
    train_range = (
        f"{train_df['date'].min()}/{train_df['date'].max()}" if not train_df.empty else None
    )
    valid_range = (
        f"{valid_df['date'].min()}/{valid_df['date'].max()}" if not valid_df.empty else None
    )

    model_dir = save_model(
        model,
        params,
        train_range,
        valid_range,
        metrics,
        feature_columns=feature_cols,
        binary_model=binary_model,
        calibrator=calibrator,
    )
    log.info("Model saved to %s", model_dir)

    # Record in model_runs（odds 抜き判定をメモに残して A/B 比較しやすくする）
    odds_excluded = "odds_excluded" if len(feature_cols) < 30 else "odds_included"
    notes_str = f"M4 baseline lambdarank ({odds_excluded})"

    with session_scope(engine) as session:
        run = ModelRun(
            created_at=datetime.now(UTC).isoformat(),
            model_path=str(model_dir),
            params_json=json.dumps(params),
            train_range=train_range,
            valid_range=valid_range,
            metrics_json=json.dumps(metrics),
            notes=notes_str,
            is_active=0,
        )
        session.add(run)

    log.info("model_runs row inserted.")
    return {"model_dir": str(model_dir), **metrics}


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train keiba-ai LightGBM lambdarank model")
    parser.add_argument("--db", type=Path, default=None, help="Path to SQLite DB")
    parser.add_argument("--train-end", default=None, help="Training end date YYYY-MM-DD")
    parser.add_argument("--valid-months", type=int, default=12, help="Validation window (months)")
    parser.add_argument("--test-months", type=int, default=6, help="Test window (months)")
    parser.add_argument(
        "--params-json",
        type=Path,
        default=None,
        help=(
            "JSON file overriding LightGBM params. "
            "Default uses synthetic-friendly min_data_in_leaf=5; "
            'for production data set {"min_data_in_leaf": 50}.'
        ),
    )
    args = parser.parse_args()

    result = train(
        db=args.db,
        train_end=args.train_end,
        valid_months=args.valid_months,
        test_months=args.test_months,
        params_json=args.params_json,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
