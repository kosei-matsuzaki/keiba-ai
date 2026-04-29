"""Single-race and batch inference.

predict_race converts LightGBM raw scores to win_prob and place_prob using
softmax and the top-k cumulative approximation respectively.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from keiba_ai.ai.calibrate import softmax_within_race, top_k_cumulative_prob
from keiba_ai.features.builder import CATEGORICAL_FEATURES, FEATURE_COLUMNS


def _prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Extract and cast feature columns from frame."""
    X = frame[FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


def predict_race(model: lgb.Booster, frame: pd.DataFrame) -> pd.DataFrame:
    """Score all horses in a single race and return calibrated probabilities.

    Args:
        model: Trained LightGBM Booster.
        frame: Feature DataFrame for one race (output of build_inference_frame
               or a training-frame slice). Must contain FEATURE_COLUMNS.

    Returns:
        DataFrame with columns: horse_id, score, win_prob, place_prob.
        Sorted by score descending (top prediction first).
    """
    if frame.empty:
        return pd.DataFrame(columns=["horse_id", "score", "win_prob", "place_prob"])

    X = _prepare_features(frame)
    scores: np.ndarray = model.predict(X)

    win_probs = softmax_within_race(scores)
    place_probs = top_k_cumulative_prob(scores, k=3)

    result = pd.DataFrame(
        {
            "horse_id": frame["horse_id"].values,
            "score": scores,
            "win_prob": win_probs,
            "place_prob": place_probs,
        }
    )
    return result.sort_values("score", ascending=False).reset_index(drop=True)


def predict_race_with_shap(
    model: lgb.Booster,
    frame: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """Same as predict_race but adds a 'top_features' column (list[str]).

    Uses SHAP TreeExplainer to identify the most influential features for each
    horse. The top_n features by absolute SHAP value are returned per horse.

    Args:
        model: Trained LightGBM Booster.
        frame: Feature DataFrame for one race. Must contain FEATURE_COLUMNS.
        top_n: Number of top features to return per horse.

    Returns:
        DataFrame with columns: horse_id, score, win_prob, place_prob, top_features.
        Sorted by score descending.
    """
    import shap

    base = predict_race(model, frame)
    if frame.empty:
        base["top_features"] = pd.Series(dtype=object)
        return base

    X = _prepare_features(frame)

    explainer = shap.TreeExplainer(model)
    shap_values: np.ndarray = explainer.shap_values(X)

    top_features_list: list[list[str]] = []
    for i in range(len(X)):
        abs_vals = np.abs(shap_values[i])
        # argsort ascending, take last top_n in reverse
        sorted_idx = np.argsort(abs_vals)[::-1][:top_n]
        top_features_list.append([FEATURE_COLUMNS[j] for j in sorted_idx])

    # Align top_features with sorted prediction order via horse_id
    horse_to_features = dict(zip(frame["horse_id"].values, top_features_list, strict=False))
    base["top_features"] = base["horse_id"].map(horse_to_features)
    return base
