"""GBDT stacking: augment NN feature frame with GBDT predictions.

Adds three per-horse columns derived from a separately-trained GBDT bundle:

* ``gbdt_score``      — lambdarank raw score
* ``gbdt_win_prob``   — calibrated win probability (race-softmax over scores)
* ``gbdt_place_prob`` — calibrated top-3 finishing probability

These columns can be appended to the NN's horse-level feature list so the
network learns to correct GBDT's residuals rather than rediscover what GBDT
already captures.

Note: GBDT predictions on data that overlaps GBDT's own training period are
in-sample (overfit). For strictly leak-free stacking, the GBDT should be
trained with a ``--train-end`` earlier than the NN's validation start, OR
use K-fold out-of-fold predictions. This module trusts the caller to pass
a GBDT whose train range is appropriate for the NN being trained.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from ai.predict import predict_race

if TYPE_CHECKING:
    from ai.registry import ModelBundle

log = logging.getLogger(__name__)

GBDT_FEATURE_COLUMNS: list[str] = [
    "gbdt_score",
    "gbdt_win_prob",
    "gbdt_place_prob",
]


def augment_frame_with_gbdt(
    frame: pd.DataFrame,
    gbdt_bundle: ModelBundle,
) -> pd.DataFrame:
    """Add per-horse GBDT prediction columns to ``frame``.

    Iterates the frame race-by-race, runs ``predict_race(gbdt_bundle, race_frame)``
    once per race and writes the (score, win_prob, place_prob) tuple back into
    each row using race_id + horse_id alignment.

    Rows for which GBDT prediction fails (e.g., race < 2 horses) get NaN in
    the GBDT columns. The NN preprocessor's numeric handling will turn these
    into 0 after standardization, which is acceptable for the rare failure
    modes (≤1 horse races shouldn't appear in training data).

    Args:
        frame: Feature DataFrame with at least ``race_id`` and ``horse_id``.
        gbdt_bundle: A loaded ModelBundle with ``model_type == "gbdt"``.

    Returns:
        A copy of frame with the three columns from ``GBDT_FEATURE_COLUMNS`` added.
    """
    if gbdt_bundle.model_type != "gbdt":
        raise ValueError(
            f"augment_frame_with_gbdt expects a GBDT bundle, got "
            f"model_type={gbdt_bundle.model_type!r}"
        )

    out = frame.copy()
    for col in GBDT_FEATURE_COLUMNS:
        out[col] = float("nan")

    if frame.empty:
        return out

    for race_id, race_frame in frame.groupby("race_id"):
        if len(race_frame) < 2:
            continue
        try:
            preds = predict_race(gbdt_bundle, race_frame)
        except Exception as exc:  # noqa: BLE001
            log.warning("GBDT predict failed for race %s: %s", race_id, exc)
            continue

        pred_lookup: dict[str, tuple[float, float, float]] = {
            str(row["horse_id"]): (
                float(row["score"]),
                float(row["win_prob"]),
                float(row["place_prob"]),
            )
            for _, row in preds.iterrows()
        }
        for idx in race_frame.index:
            hid = str(frame.at[idx, "horse_id"])
            if hid in pred_lookup:
                s, w, p = pred_lookup[hid]
                out.at[idx, "gbdt_score"] = s
                out.at[idx, "gbdt_win_prob"] = w
                out.at[idx, "gbdt_place_prob"] = p

    return out
