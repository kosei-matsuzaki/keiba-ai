"""Hyperparameter tuning with Optuna.

Usage:
    uv run python -m ai.gbm.tune --n-trials 20
    uv run python -m ai.gbm.tune --n-trials 50 --train-end 2025-12-31
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.metrics import ndcg_score

from ai.labels import assign_relevance
from ai.splits import time_split
from core.paths import db_path
from db.models import ModelRun  # noqa: F401 — register table with Base
from db.session import make_engine, session_scope
from features.builder import CATEGORICAL_FEATURES, FEATURE_COLUMNS, build_training_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _make_lgb_dataset(
    df,
    reference: lgb.Dataset | None = None,
) -> lgb.Dataset:
    X = df[FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")
    y = df["relevance"].values.astype(np.float32)
    group = df.groupby("race_id", sort=False)["horse_id"].count().values
    return lgb.Dataset(
        X,
        label=y,
        group=group,
        feature_name=FEATURE_COLUMNS,
        categorical_feature=CATEGORICAL_FEATURES,
        reference=reference,
        free_raw_data=False,
    )


def _compute_ndcg3(model: lgb.Booster, df) -> float:
    if df.empty:
        return float("nan")
    X = df[FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")
    scores = model.predict(X)
    df = df.copy()
    df["_score"] = scores
    vals: list[float] = []
    for _, grp in df.groupby("race_id"):
        if len(grp) < 2:
            continue
        true_rel = grp["relevance"].values.reshape(1, -1)
        pred_scores = grp["_score"].values.reshape(1, -1)
        vals.append(float(ndcg_score(true_rel, pred_scores, k=3)))
    return float(np.mean(vals)) if vals else float("nan")


def objective(
    trial: optuna.Trial,
    train_df,
    valid_df,
) -> float:
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [3],
        "lambdarank_truncation_level": 3,
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
        "bagging_freq": 5,
        "verbose": -1,
    }

    train_data = _make_lgb_dataset(train_df)

    if not valid_df.empty:
        valid_data = _make_lgb_dataset(valid_df, reference=train_data)
        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
        model = lgb.train(
            params,
            train_data,
            num_boost_round=300,
            valid_sets=[valid_data],
            valid_names=["valid"],
            callbacks=callbacks,
        )
        return _compute_ndcg3(model, valid_df)
    else:
        model = lgb.train(params, train_data, num_boost_round=100)
        return _compute_ndcg3(model, train_df)


def tune(
    db: Path | None = None,
    train_end: str | None = None,
    valid_months: int = 12,
    test_months: int = 6,
    n_trials: int = 20,
    storage: str | None = None,
) -> dict:
    """Run Optuna hyperparameter search. Returns best_params and best_value."""
    resolved_db = db or db_path()
    engine = make_engine(resolved_db)

    log.info("Building feature frame from %s", resolved_db)
    with session_scope(engine) as session:
        frame = build_training_frame(session)

    if frame.empty:
        raise RuntimeError("No training data found in the database.")

    frame["relevance"] = frame["finish_position"].map(assign_relevance)
    train_df, valid_df, _test_df = time_split(frame, train_end, valid_months, test_months)

    if train_df.empty:
        log.warning("Train set empty — using full frame.")
        train_df = frame.copy()
        valid_df = frame.copy()

    log.info(
        "Optuna tuning: n_trials=%d train=%d valid=%d",
        n_trials,
        len(train_df),
        len(valid_df),
    )

    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name="keiba_lgb_tune",
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(trial, train_df, valid_df),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    result = {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
    }
    log.info("Best params: %s  |  best NDCG@3: %.4f", result["best_params"], result["best_value"])
    return result


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for keiba-ai")
    parser.add_argument("--db", type=Path, default=None, help="Path to SQLite DB")
    parser.add_argument("--train-end", default=None, help="Training end date YYYY-MM-DD")
    parser.add_argument("--valid-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL (e.g. sqlite:///optuna.db). Default: in-memory.",
    )
    args = parser.parse_args()

    result = tune(
        db=args.db,
        train_end=args.train_end,
        valid_months=args.valid_months,
        test_months=args.test_months,
        n_trials=args.n_trials,
        storage=args.storage,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
