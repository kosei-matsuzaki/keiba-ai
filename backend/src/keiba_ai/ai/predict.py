"""Single-race and batch inference.

predict_race converts LightGBM raw scores to win_prob and place_prob using
softmax and the top-k cumulative approximation respectively.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from keiba_ai.ai.calibrate import softmax_within_race, top_k_cumulative_prob
from keiba_ai.features.builder import FEATURE_COLUMNS


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

    X = frame[FEATURE_COLUMNS].copy()
    # LightGBM handles NaN natively; cast categoricals to 'category' dtype
    from keiba_ai.features.builder import CATEGORICAL_FEATURES
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")

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
