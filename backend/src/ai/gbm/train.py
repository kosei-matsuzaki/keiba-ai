"""CLI: Train a LightGBM lambdarank or Plackett-Luce model and register it.

Usage:
    uv run python -m ai.gbm.train [--train-end YYYY-MM-DD]
                                        [--valid-months 12]
                                        [--test-months 6]
                                        [--db PATH]
                                        [--params-json PATH]
                                        [--loss {lambdarank,plackett_luce}]
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

from ai.calibrate import ConditionalIsotonicCalibrator, IsotonicCalibrator
from ai.cv import rolling_origin_splits
from ai.gbm.pl_loss import plackett_luce_eval_metric, plackett_luce_objective
from ai.labels import assign_is_winner, assign_relevance
from ai.registry import save_model
from ai.splits import time_split
from core.paths import db_path
from db.models import ModelRun  # noqa: F401 — register table with Base
from db.session import make_engine, session_scope
from features.builder import (
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


def _compute_recency_weights(df: pd.DataFrame, recency_lambda: float) -> np.ndarray | None:
    """Compute per-row sample weights based on race recency.

    Weight formula (λ > 0):
        age_years  = (latest_date_in_train - race_date).days / 365.25
        weight     = exp(-λ × age_years)

    The latest_date is derived from the df itself (not from any test period) to
    prevent data leakage.  When λ = 0 returns None, meaning uniform weights.
    """
    if recency_lambda <= 0.0:
        return None

    dates = pd.to_datetime(df["date"])
    latest_date = dates.max()
    age_years = (latest_date - dates).dt.days / 365.25
    weights = np.exp(-recency_lambda * age_years).astype(np.float32)
    return weights


def _make_lgb_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    reference: lgb.Dataset | None = None,
    recency_lambda: float = 0.0,
) -> lgb.Dataset:
    """Build a LightGBM Dataset with group counts by race_id (lambdarank).

    When recency_lambda > 0, each row is weighted by exp(-λ × age_years) so
    that older races contribute less to the lambdarank loss.
    """
    X = df[feature_cols].copy()
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype("category")

    y = df["relevance"].values.astype(np.float32)

    # group = count of entries per race, in the order they appear in df
    group = df.groupby("race_id", sort=False)["horse_id"].count().values

    sample_weight = _compute_recency_weights(df, recency_lambda)

    return lgb.Dataset(
        X,
        label=y,
        group=group,
        weight=sample_weight,
        feature_name=feature_cols,
        categorical_feature=categorical_cols,
        reference=reference,
        free_raw_data=False,
    )


def _make_lgb_dataset_pl(
    df: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
    reference: lgb.Dataset | None = None,
    recency_lambda: float = 0.0,
) -> lgb.Dataset:
    """Build a LightGBM Dataset for Plackett-Luce custom objective.

    Labels are raw finish_position values (1-based integers, NaN for
    non-finishers).  The PL objective reads these via ``get_label()`` to
    determine the ranking order within each race group.

    When recency_lambda > 0, sample weights are applied identically to the
    lambdarank path (older rows down-weighted by exp(-λ × age_years)).
    """
    X = df[feature_cols].copy()
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype("category")

    # Use raw finish_position as label; NaN (non-finishers) stay as NaN
    # (they'll be masked out inside the objective).
    y = df["finish_position"].values.astype(np.float32)

    group = df.groupby("race_id", sort=False)["horse_id"].count().values

    sample_weight = _compute_recency_weights(df, recency_lambda)

    return lgb.Dataset(
        X,
        label=y,
        group=group,
        weight=sample_weight,
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
    recency_lambda: float = 0.0,
) -> tuple[lgb.Booster, IsotonicCalibrator, dict]:
    """Train a binary classifier (objective=binary) on is_winner and a
    post-hoc isotonic calibrator on the validation set.

    The binary classifier outputs sigmoid-calibrated raw probabilities.
    The isotonic regression then corrects any residual systematic bias by
    learning monotonic mapping (raw_prob → empirical win rate) on valid set.

    When recency_lambda > 0, training rows are weighted by exp(-λ × age_years)
    to down-weight older races (same weight formula as the lambdarank head).

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

    train_sample_weight = _compute_recency_weights(train_df, recency_lambda)

    binary_train_data = lgb.Dataset(
        train_X,
        label=train_y_bin,
        weight=train_sample_weight,
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


def _fit_temperature_scaler(
    model: lgb.Booster,
    valid_df: pd.DataFrame,
    feature_cols: list[str],
    loss_type: str = "lambdarank",
):
    """Fit a TemperatureScaler on the validation set using payback-maximising grid search.

    Extracts per-race scores, finish_positions, odds_win and payout_place from
    valid_df, then delegates to TemperatureScaler.fit().

    Returns:
        Fitted TemperatureScaler.
    """
    from ai.temperature import TemperatureScaler

    scores_per_race: list[np.ndarray] = []
    finish_positions_per_race: list[np.ndarray] = []
    odds_win_per_race: list[np.ndarray] = []
    payout_place_per_race: list = []

    for _race_id, grp in valid_df.groupby("race_id", sort=False):
        if len(grp) < 2:
            continue

        X = grp[feature_cols].copy()
        for col in CATEGORICAL_FEATURES:
            if col in X.columns:
                X[col] = X[col].astype("category")
        scores = model.predict(X)

        finish_pos = grp["finish_position"].values.astype(float)
        odds_win = (
            grp["odds_win"].values.astype(float)
            if "odds_win" in grp.columns
            else np.full(len(grp), float("nan"))
        )

        # Parse payout_place JSON for the race (one row per race)
        payout_map: dict[int, int] | None = None
        if "payout_place" in grp.columns:
            raw_val = grp["payout_place"].dropna()
            if not raw_val.empty:
                import json
                try:
                    raw_dict = json.loads(raw_val.iloc[0])
                    payout_map = {int(k): int(v) for k, v in raw_dict.items()}
                except (json.JSONDecodeError, ValueError, TypeError):
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


def _train_single_split(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    params: dict,
    recency_lambda: float = 0.0,
    loss: str = "lambdarank",
    conditional_calibration: bool = False,
    fit_temperature: bool = True,
) -> tuple[
    lgb.Booster,
    lgb.Booster | None,
    IsotonicCalibrator | ConditionalIsotonicCalibrator | None,
    object,
    object,
    dict,
]:
    """Train lambdarank/PL + (optional) binary + calibrator + combo on one (train, valid, test) split.

    When recency_lambda > 0, sample weights are applied to both lambdarank
    and binary heads (older rows down-weighted by exp(-λ × age_years)).
    When loss == "plackett_luce", a custom PL objective is used and the binary
    classifier/calibrator are skipped (softmax(score) is the calibrated win
    probability).
    When conditional_calibration=True, isotonic / combo calibrators are fit
    per (surface, n_runners_bin) bucket with global fallback.
    When fit_temperature=True (default) and valid_df is not empty, a TemperatureScaler
    is fit on the validation set using 1D grid search over payback.

    Returns:
        (model, binary_model, calibrator, combo_calibrators, temperature_scaler, metrics)
        binary_model and calibrator are None in PL mode.
        temperature_scaler is None when valid_df is empty or fit_temperature=False.
    """
    if loss not in ("lambdarank", "plackett_luce"):
        raise ValueError(f"Unknown loss type: {loss!r}. Choose 'lambdarank' or 'plackett_luce'.")

    if recency_lambda > 0.0:
        log.info(
            "Recency weighting enabled: λ=%.4f (weight = exp(-λ × age_years))",
            recency_lambda,
        )

    use_pl = loss == "plackett_luce"

    if use_pl:
        # In PL mode, labels are raw finish_position values (1-based integers).
        # The PL objective uses them to determine the ordering within each race.
        train_data = _make_lgb_dataset_pl(
            train_df, feature_cols, CATEGORICAL_FEATURES, recency_lambda=recency_lambda
        )
    else:
        train_data = _make_lgb_dataset(
            train_df, feature_cols, CATEGORICAL_FEATURES, recency_lambda=recency_lambda
        )

    callbacks = [lgb.log_evaluation(period=50)]
    valid_sets: list[lgb.Dataset] = [train_data]
    valid_names: list[str] = ["train"]

    if not valid_df.empty:
        if use_pl:
            valid_data = _make_lgb_dataset_pl(
                valid_df, feature_cols, CATEGORICAL_FEATURES, reference=train_data
            )
        else:
            valid_data = _make_lgb_dataset(
                valid_df, feature_cols, CATEGORICAL_FEATURES, reference=train_data
            )
        valid_sets.append(valid_data)
        valid_names.append("valid")
        callbacks.append(lgb.early_stopping(stopping_rounds=50, verbose=False))

    if use_pl:
        # Build group sizes for each split (must match dataset construction order).
        train_group_sizes = train_df.groupby("race_id", sort=False)["horse_id"].count().tolist()
        pl_objective = plackett_luce_objective(train_group_sizes)

        # PL params: remove lambdarank-specific keys, set objective to custom.
        pl_params = {
            k: v
            for k, v in params.items()
            if k not in ("objective", "metric", "ndcg_eval_at", "lambdarank_truncation_level")
        }
        pl_params["objective"] = pl_objective

        # Use PL NLL as evaluation metric on valid set when available.
        if not valid_df.empty:
            valid_group_sizes = valid_df.groupby("race_id", sort=False)["horse_id"].count().tolist()
            pl_eval = plackett_luce_eval_metric(valid_group_sizes)
            # Also register on train set using same eval (for monitoring).
            train_eval = plackett_luce_eval_metric(train_group_sizes)
            feval = [train_eval, pl_eval]
            # LightGBM custom feval receives (preds, dataset); we need per-split
            # functions registered separately.  We use a single feval dict trick:
            # pass both under different dataset names via a wrapper.
            pl_params["metric"] = "custom"
        else:
            pl_params.pop("metric", None)
            feval = None

        log.info("Starting LightGBM training with Plackett-Luce objective…")
        model = lgb.train(
            pl_params,
            train_data,
            num_boost_round=300,
            valid_sets=valid_sets,
            valid_names=valid_names,
            feval=pl_eval if not valid_df.empty else None,
            callbacks=callbacks,
        )
    else:
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

    metrics: dict = {
        "valid_ndcg1": valid_ndcg1,
        "valid_ndcg3": valid_ndcg3,
        "test_ndcg1": test_ndcg1,
        "test_ndcg3": test_ndcg3,
        # UI/Dashboard が期待する flat な key (test 値を canonical 採用)。
        # evaluate.py --persist が後から payback_win など他のメトリクスを
        # merge する設計のため、ここでは ndcg のみで充分。
        "ndcg1": test_ndcg1 if not pd.isna(test_ndcg1) else valid_ndcg1,
        "ndcg3": test_ndcg3 if not pd.isna(test_ndcg3) else valid_ndcg3,
    }
    log.info("%s metrics: %s", loss, metrics)

    # ── Phase 2: binary classifier + isotonic calibrator ──────────────────────
    # Skipped in Plackett-Luce mode: softmax(score) within each race serves
    # directly as calibrated win probability, eliminating the need for a
    # separate binary head and its associated calibrator.
    if use_pl:
        binary_model = None
        calibrator = None
        log.info("PL mode: skipping binary classifier and calibrator.")
    else:
        # 既存の lambdarank score (順位用) に加えて、別 head として binary classifier
        # を学習し isotonic で post-hoc 補正する。これにより推論時の win_prob は
        # softmax(lambdarank scores) ではなく真の確率に近い値を返せるようになる。
        binary_model, calibrator, binary_metrics = _train_binary_classifier_and_calibrator(
            train_df, valid_df, feature_cols, params, recency_lambda=recency_lambda
        )
        metrics.update(binary_metrics)

        # ── conditional calibration: replace global isotonic with per-stratum one ─
        if conditional_calibration and not valid_df.empty:
            log.info("Fitting ConditionalIsotonicCalibrator (surface × n_runners bin)…")
            feature_cols_binary = list(binary_model.feature_name())
            valid_X_cond = valid_df[feature_cols_binary].copy()
            for col in CATEGORICAL_FEATURES:
                if col in valid_X_cond.columns:
                    valid_X_cond[col] = valid_X_cond[col].astype("category")
            valid_raw_cond = binary_model.predict(valid_X_cond)
            valid_y_cond = valid_df["finish_position"].map(
                lambda p: 1 if p == 1 else 0
            ).values.astype(np.float32)

            # Build conditions DataFrame from valid_df.
            n_runners_col = (
                valid_df["n_runners"]
                if "n_runners" in valid_df.columns
                else valid_df.groupby("race_id")["horse_id"].transform("count")
            )
            cond_df = pd.DataFrame(
                {
                    "surface": valid_df["surface"].values if "surface" in valid_df.columns else "unknown",
                    "n_runners": n_runners_col.values,
                }
            )
            cond_calibrator = ConditionalIsotonicCalibrator()
            cond_calibrator.fit(valid_raw_cond, valid_y_cond, cond_df)
            calibrator = cond_calibrator
            log.info(
                "ConditionalIsotonicCalibrator fitted with %d strata",
                len(cond_calibrator._calibrators),
            )

    log.info("All metrics: %s", metrics)

    # ── Phase A 後追加: 連系 馬券 (馬連 / ワイド / 馬単 / 三連複 / 三連単) の
    # PL 由来 combo prob は系統的に長配側で過大評価される (combo_calibration_diagnosis
    # 参照, ratio 2-7x)。馬券種ごとに isotonic 補正を学習して save_model で永続化する。
    # 学習は valid_df 上で行うので訓練データへのリークは無い。
    combo_calibrators = None
    if not valid_df.empty:
        log.info("Fitting combo calibrators on valid set (馬連/ワイド/馬単/三連複/三連単)…")
        from ai.calibrate import fit_combo_calibrators

        try:
            combo_calibrators = fit_combo_calibrators(
                valid_frame=valid_df,
                lambdarank_model=model,
                binary_model=binary_model,
                single_horse_calibrator=calibrator,
                n_samples=5_000,
                use_conditional=conditional_calibration,
            )
            log.info(
                "Combo calibrators fitted for: %s",
                combo_calibrators.fitted_bet_types,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("fit_combo_calibrators failed: %s — proceeding without combo cal", exc)
            combo_calibrators = None

    # ── Temperature scaling: fit per-bet-type temperature on valid set ────────
    # Skipped in lambdarank mode: the binary head + isotonic calibrator already
    # produces well-calibrated win probabilities (in [0, 1] with sum=1), and
    # applying softmax(probs / T) on top of that re-normalises a *probability
    # distribution* rather than a *score vector*, which flattens the top horse
    # and inflates the EV of mid-tier horses → over-betting → ROI collapse
    # (empirically observed: lambdarank payback_win 0.951 → 0.556 with T).
    # Temperature scaling is therefore only fit for plackett_luce mode, where
    # the model output is a raw log-utility score that needs softmax to become
    # a probability in the first place.
    temperature_scaler = None
    if fit_temperature and not valid_df.empty and loss == "plackett_luce":
        log.info("Fitting TemperatureScaler on valid set (PL mode)…")
        from ai.temperature import TemperatureScaler

        try:
            temperature_scaler = _fit_temperature_scaler(
                model=model,
                valid_df=valid_df,
                feature_cols=feature_cols,
                loss_type=loss,
            )
            log.info(
                "TemperatureScaler fitted: T_win=%.3f, T_place=%.3f",
                temperature_scaler.T_win,
                temperature_scaler.T_place,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("TemperatureScaler fit failed: %s — proceeding without temperature scaling", exc)
            temperature_scaler = None
    elif fit_temperature and loss == "lambdarank":
        log.info(
            "Skipping TemperatureScaler fit: lambdarank mode is already calibrated "
            "by the binary head + isotonic regression."
        )

    return model, binary_model, calibrator, combo_calibrators, temperature_scaler, metrics


def _aggregate_cv_metrics(per_fold: list[dict]) -> tuple[dict, dict]:
    """Compute mean and std across per-fold metric dicts.

    Non-numeric and NaN values are ignored in the aggregation.
    Returns (mean_dict, std_dict) with the same keys as per_fold entries.
    """
    if not per_fold:
        return {}, {}

    all_keys = per_fold[0].keys()
    mean_dict: dict = {}
    std_dict: dict = {}

    for key in all_keys:
        values = [
            v for fold in per_fold
            if (v := fold.get(key)) is not None and isinstance(v, (int, float)) and not pd.isna(v)
        ]
        if values:
            mean_dict[key] = float(np.mean(values))
            std_dict[key] = float(np.std(values, ddof=0))
        else:
            mean_dict[key] = float("nan")
            std_dict[key] = float("nan")

    return mean_dict, std_dict


def train(
    db: Path | None = None,
    train_end: str | None = None,
    valid_months: int = 12,
    test_months: int = 6,
    params_json: Path | None = None,
    cv_folds: int = 1,
    recency_lambda: float = 0.0,
    loss: str = "lambdarank",
    conditional_calibration: bool = False,
    fit_temperature: bool = True,
) -> dict:
    """Run the full training pipeline. Returns metrics dict.

    When cv_folds >= 2 the pipeline runs rolling-origin cross-validation:
    each fold trains on all data before its validation window, evaluates on
    a held-out test window, and the metrics are aggregated (mean + std).
    The model from the most-recent fold (fold 1) is saved and registered,
    matching the behaviour of the single-split path (cv_folds == 1).

    When cv_folds == 1 behaviour is identical to the original implementation.

    Args:
        recency_lambda: Exponential decay factor for recency-weighted sample
            weights.  When > 0, each training row is weighted by
            exp(-λ × age_years) where age_years is the number of years before
            the most recent race in the training set.  Set to 0.0 (default) to
            use uniform weights (backward-compatible behaviour).
        loss: "lambdarank" (default, backward-compatible) or "plackett_luce".
            In PL mode the binary classifier and calibrator are not trained;
            softmax(score) within each race gives calibrated win probabilities.
        conditional_calibration: If True, fit the isotonic calibrator and combo
            calibrators per (surface, n_runners_bin) bucket with global fallback
            on sparse strata.  Default False keeps backward-compatible global
            isotonic regression.
    """
    if loss not in ("lambdarank", "plackett_luce"):
        raise ValueError(f"Unknown loss type: {loss!r}. Choose 'lambdarank' or 'plackett_luce'.")

    resolved_db = db or db_path()
    engine = make_engine(resolved_db)

    params = dict(DEFAULT_PARAMS)
    if params_json:
        params.update(json.loads(Path(params_json).read_text(encoding="utf-8")))

    # Store recency_lambda in params so it is persisted to meta.json.
    params["recency_lambda"] = recency_lambda

    log.info("Building feature frame from %s", resolved_db)
    with session_scope(engine) as session:
        frame = build_training_frame(session)

    if frame.empty:
        raise RuntimeError("No training data found in the database.")

    frame["relevance"] = frame["finish_position"].map(assign_relevance)

    log.info("Total rows: %d | Races: %d", len(frame), frame["race_id"].nunique())

    # 学習で使う特徴量列。KEIBA_EXCLUDE_ODDS_FEATURES=1 で 5 つの odds 派生を除外
    feature_cols = get_active_features()
    log.info(
        "Training with %d features (KEIBA_EXCLUDE_ODDS_FEATURES=%s)",
        len(feature_cols),
        os.environ.get("KEIBA_EXCLUDE_ODDS_FEATURES", "0"),
    )

    if cv_folds >= 2:
        return _train_with_cv(
            frame=frame,
            engine=engine,
            feature_cols=feature_cols,
            params=params,
            valid_months=valid_months,
            test_months=test_months,
            n_folds=cv_folds,
            recency_lambda=recency_lambda,
            loss=loss,
            conditional_calibration=conditional_calibration,
            fit_temperature=fit_temperature,
        )

    # ── Single-split path (cv_folds == 1, original behaviour) ────────────────
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

    model, binary_model, calibrator, combo_calibrators, temperature_scaler, metrics = _train_single_split(
        train_df, valid_df, test_df, feature_cols, params,
        recency_lambda=recency_lambda, loss=loss,
        conditional_calibration=conditional_calibration,
        fit_temperature=fit_temperature,
    )

    return _save_and_register(
        engine=engine,
        model=model,
        binary_model=binary_model,
        calibrator=calibrator,
        combo_calibrators=combo_calibrators,
        temperature_scaler=temperature_scaler,
        params=params,
        feature_cols=feature_cols,
        train_df=train_df,
        valid_df=valid_df,
        metrics=metrics,
        loss=loss,
        conditional_calibration=conditional_calibration,
    )


def _train_with_cv(
    frame: pd.DataFrame,
    engine,
    feature_cols: list[str],
    params: dict,
    valid_months: int,
    test_months: int,
    n_folds: int,
    recency_lambda: float = 0.0,
    loss: str = "lambdarank",
    conditional_calibration: bool = False,
    fit_temperature: bool = True,
) -> dict:
    """Rolling-origin CV training.  Saves/registers only the most-recent fold's model."""
    log.info("Starting rolling-origin CV with %d folds.", n_folds)

    per_fold_metrics: list[dict] = []
    last_fold_artifacts: tuple | None = None  # (model, binary, cal, combo, ts, train_df, valid_df)

    for fold_idx, (train_df, valid_df, test_df) in enumerate(
        rolling_origin_splits(frame, n_folds, valid_months, test_months)
    ):
        fold_num = fold_idx + 1
        log.info(
            "CV fold %d/%d — train=%d rows, valid=%d rows, test=%d rows",
            fold_num,
            n_folds,
            len(train_df),
            len(valid_df),
            len(test_df),
        )

        model, binary_model, calibrator, combo_calibrators, temperature_scaler, fold_metrics = _train_single_split(
            train_df, valid_df, test_df, feature_cols, params,
            recency_lambda=recency_lambda, loss=loss,
            conditional_calibration=conditional_calibration,
            fit_temperature=fit_temperature,
        )
        per_fold_metrics.append(fold_metrics)

        # Fold 1 (most recent) is used for the final saved model.
        if fold_idx == 0:
            last_fold_artifacts = (
                model, binary_model, calibrator, combo_calibrators, temperature_scaler, train_df, valid_df
            )

    if not per_fold_metrics:
        raise RuntimeError("All CV folds were skipped (train sets empty). Widen the date range.")

    mean_metrics, std_metrics = _aggregate_cv_metrics(per_fold_metrics)
    log.info("CV mean metrics: %s", mean_metrics)
    log.info("CV std  metrics: %s", std_metrics)

    # Build the top-level metrics dict: canonical flat keys come from fold-1
    # (most recent), so downstream consumers that don't understand cv_metrics
    # still get meaningful numbers.
    assert last_fold_artifacts is not None
    model, binary_model, calibrator, combo_calibrators, temperature_scaler, train_df, valid_df = last_fold_artifacts

    # Fold-1 metrics are per_fold_metrics[0].
    canonical_metrics = dict(per_fold_metrics[0])
    canonical_metrics["cv_metrics"] = {
        "n_folds": len(per_fold_metrics),
        "mean": mean_metrics,
        "std": std_metrics,
        "per_fold": per_fold_metrics,
    }

    return _save_and_register(
        engine=engine,
        model=model,
        binary_model=binary_model,
        calibrator=calibrator,
        combo_calibrators=combo_calibrators,
        temperature_scaler=temperature_scaler,
        params=params,
        feature_cols=feature_cols,
        train_df=train_df,
        valid_df=valid_df,
        metrics=canonical_metrics,
        loss=loss,
        conditional_calibration=conditional_calibration,
    )


def _save_and_register(
    *,
    engine,
    model: lgb.Booster,
    binary_model: lgb.Booster | None,
    calibrator: IsotonicCalibrator | ConditionalIsotonicCalibrator | None,
    combo_calibrators,
    temperature_scaler=None,
    params: dict,
    feature_cols: list[str],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    metrics: dict,
    loss: str = "lambdarank",
    conditional_calibration: bool = False,
) -> dict:
    """Persist model files and insert a model_runs DB row. Returns result dict."""
    train_range = (
        f"{train_df['date'].min()}/{train_df['date'].max()}" if not train_df.empty else None
    )
    valid_range = (
        f"{valid_df['date'].min()}/{valid_df['date'].max()}" if not valid_df.empty else None
    )

    # In PL mode the stored params dict still contains the original DEFAULT_PARAMS
    # keys (num_leaves, learning_rate, etc.) which are safe to persist.  The
    # objective key is a Python callable and cannot be serialised; we drop it.
    serialisable_params = {k: v for k, v in params.items() if not callable(v)}

    model_dir = save_model(
        model,
        serialisable_params,
        train_range,
        valid_range,
        metrics,
        feature_columns=feature_cols,
        binary_model=binary_model,
        calibrator=calibrator,
        combo_calibrators=combo_calibrators,
        loss_type=loss,
        conditional_calibration=conditional_calibration,
        temperature_scaler=temperature_scaler,
    )
    log.info("Model saved to %s", model_dir)

    odds_excluded = "odds_excluded" if len(feature_cols) < 30 else "odds_included"
    notes_str = f"{loss} ({odds_excluded})"

    with session_scope(engine) as session:
        run = ModelRun(
            created_at=datetime.now(UTC).isoformat(),
            model_path=str(model_dir),
            params_json=json.dumps(serialisable_params),
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
    parser = argparse.ArgumentParser(
        description="Train keiba-ai LightGBM lambdarank or Plackett-Luce model"
    )
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
    parser.add_argument(
        "--recency-lambda",
        type=float,
        default=0.0,
        help=(
            "Exponential decay factor for recency-weighted sample weights. "
            "When λ > 0, each training row is weighted by exp(-λ × age_years) "
            "where age_years = (latest_date_in_train - race_date).days / 365.25. "
            "Default 0.0 disables weighting (uniform weights, backward-compatible)."
        ),
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=1,
        help=(
            "Number of rolling-origin CV folds (default 1 = single split, original behaviour). "
            "When >= 2, trains on each fold and saves aggregate cv_metrics alongside the "
            "most-recent fold's model."
        ),
    )
    parser.add_argument(
        "--loss",
        choices=["lambdarank", "plackett_luce"],
        default="lambdarank",
        help=(
            "Loss function to use. 'lambdarank' (default) uses the built-in LightGBM "
            "lambdarank objective with a separate binary classifier for win probabilities. "
            "'plackett_luce' uses a custom Plackett-Luce log-likelihood objective where "
            "softmax(score) directly gives calibrated win probabilities (no binary head)."
        ),
    )
    parser.add_argument(
        "--conditional-calibration",
        action="store_true",
        default=False,
        help=(
            "Use ConditionalIsotonicCalibrator (surface × n_runners bin) "
            "for isotonic calibration and combo calibrators.  Default is "
            "False (backward-compatible global isotonic regression)."
        ),
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

    result = train(
        db=args.db,
        train_end=args.train_end,
        valid_months=args.valid_months,
        test_months=args.test_months,
        params_json=args.params_json,
        cv_folds=args.cv_folds,
        recency_lambda=args.recency_lambda,
        loss=args.loss,
        conditional_calibration=args.conditional_calibration,
        fit_temperature=not args.no_fit_temperature,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
