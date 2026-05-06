"""Single-race and batch inference.

predict_race converts LightGBM raw scores to win_prob and place_prob using
softmax and a place-probability estimator selected by KEIBA_PLACE_PROB_METHOD.

KEIBA_PLACE_PROB_METHOD:
  plackett_luce  (default) — Plackett-Luce Monte Carlo, per-horse probabilities
  heuristic                — legacy top_k_cumulative_prob approximation

predict_race_with_combinations extends predict_race with EV calculations for
all combination bet types (馬連, ワイド, 馬単, 三連複, 三連単).
"""

from __future__ import annotations

import os
from itertools import combinations, permutations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session  # noqa: F401 — kept for API compatibility

from keiba_ai.ai.calibrate import (
    IsotonicCalibrator,
    compute_all_combination_probs,
    plackett_luce_place_prob,
    softmax_within_race,
    top_k_cumulative_prob,
)
from keiba_ai.ai.types import CombinationPrediction
from keiba_ai.features.builder import CATEGORICAL_FEATURES, FEATURE_COLUMNS

_PLACE_PROB_METHOD = os.environ.get("KEIBA_PLACE_PROB_METHOD", "plackett_luce")


def _prepare_features(
    frame: pd.DataFrame, model: lgb.Booster | None = None
) -> pd.DataFrame:
    """Extract and cast feature columns from frame.

    model が渡された場合、学習時の feature_name() を使って列を選ぶ
    （odds 抜きモデルでも正しく動作する）。model=None のときは
    後方互換のため FEATURE_COLUMNS を使う。
    """
    if model is not None:
        cols = list(model.feature_name())
    else:
        cols = FEATURE_COLUMNS
    X = frame[cols].copy()
    for col in CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


def _compute_place_prob(scores: np.ndarray) -> np.ndarray:
    """Dispatch to the configured place-probability estimator.

    Reads KEIBA_PLACE_PROB_METHOD at call time so that monkeypatching the
    module-level variable in tests takes immediate effect.
    """
    method = os.environ.get("KEIBA_PLACE_PROB_METHOD", _PLACE_PROB_METHOD)
    if method == "heuristic":
        return top_k_cumulative_prob(scores, k=3)
    # Default: plackett_luce
    return plackett_luce_place_prob(scores, k=3, n_samples=10_000)


def predict_race(
    model: lgb.Booster,
    frame: pd.DataFrame,
    binary_model: lgb.Booster | None = None,
    calibrator: IsotonicCalibrator | None = None,
) -> pd.DataFrame:
    """Score all horses in a single race and return calibrated probabilities.

    Args:
        model: Trained LightGBM lambdarank Booster (順位用).
        frame: Feature DataFrame for one race (output of build_inference_frame
               or a training-frame slice). Must contain FEATURE_COLUMNS.
        binary_model: Optional binary classifier (objective=binary) trained
            in parallel. When provided together with `calibrator`, win_prob
            is computed as `calibrator(binary_model.predict(X))` instead of
            softmax(lambdarank scores). This produces well-calibrated
            probabilities suitable for EV calculation.
        calibrator: Optional IsotonicCalibrator fit on the validation set.
            Used in tandem with `binary_model`.

    Returns:
        DataFrame with columns: horse_id, score, win_prob, place_prob.
        Sorted by score descending (top prediction first).

    Notes:
        - Lambdarank `score` is always returned (用途: 順位ソート, NDCG 評価).
        - place_prob は引き続き Plackett-Luce で lambdarank scores から計算する。
          binary head は単勝確率にしか使わない。
    """
    if frame.empty:
        return pd.DataFrame(columns=["horse_id", "score", "win_prob", "place_prob"])

    X = _prepare_features(frame, model=model)
    scores: np.ndarray = model.predict(X)

    if binary_model is not None and calibrator is not None:
        # Calibrated win_prob path (Phase 2 onward)
        # binary_model may have been trained with a different feature subset;
        # use its own feature_name() instead of the lambdarank model's.
        X_binary = _prepare_features(frame, model=binary_model)
        raw_win = binary_model.predict(X_binary)
        win_probs = calibrator.predict(raw_win, normalise=True)
    else:
        # Backward-compat: softmax(lambdarank scores)
        win_probs = softmax_within_race(scores)

    place_probs = _compute_place_prob(scores)

    result = pd.DataFrame(
        {
            "horse_id": frame["horse_id"].values,
            "score": scores,
            "win_prob": win_probs,
            "place_prob": place_probs,
        }
    )
    return result.sort_values("score", ascending=False).reset_index(drop=True)


def derive_wide_prob_from_triple(
    triple_prob: dict[frozenset, float],
    n: int,
) -> np.ndarray:
    """Derive wide (ワイド) probabilities from triple_prob.

    ワイド: both horse i and horse j finish in the top-3 (order irrelevant).
    This equals the sum of triple_prob over all triples that contain both i and j.

    wide_prob[(i,j)] = sum_{k != i,j} triple_prob[{i,j,k}]

    The result is a symmetric matrix with 0 on the diagonal. Each off-diagonal
    entry [i,j] is P(both i and j are in the top-3).

    This is a Monte Carlo approximation derived from the same PL samples used
    for three-horse combinations — it is internally consistent with triple_prob
    but is an approximation of the true PL marginal (exact computation would
    require summing over all triples analytically, which matches this approach
    when triple_prob was derived from MC samples).

    Args:
        triple_prob: Dict mapping frozenset of 3 horse indices to probability.
        n: Number of horses in the race.

    Returns:
        Symmetric ndarray of shape (n, n) where result[i,j] = P(i and j in top-3).
    """
    out = np.zeros((n, n))
    for fs, p in triple_prob.items():
        members = sorted(fs)
        # Each triple contributes to all 3 pairs within it
        for a, b in [
            (members[0], members[1]),
            (members[0], members[2]),
            (members[1], members[2]),
        ]:
            out[a, b] += p
            out[b, a] += p
    return out


def predict_race_with_combinations(
    model: lgb.Booster,
    frame: pd.DataFrame,
    session: Session | None = None,
    n_samples: int = 10_000,
    rng: np.random.Generator | None = None,
    top_k_combinations: int | None = None,
    race_odds: dict[str, dict[str, float]] | None = None,
    race_odds_sources: dict[str, dict[str, str]] | None = None,
    binary_model: lgb.Booster | None = None,
    calibrator: IsotonicCalibrator | None = None,
) -> dict[str, list[CombinationPrediction]]:
    """Extend predict_race with EV calculations for all combination bet types.

    Calls predict_race internally (does not modify it) and augments the result
    with Plackett-Luce probability estimates for 馬連, ワイド, 馬単, 三連複,
    三連単, plus confirmed odds from race_odds when available.

    Args:
        model: Trained LightGBM Booster.
        frame: Feature DataFrame for one race (output of build_inference_frame).
            Must contain FEATURE_COLUMNS and 'post_position'.
        session: Unused; retained for API compatibility.
        n_samples: Number of Plackett-Luce Monte Carlo samples.
        rng: Optional Generator for reproducibility.
        top_k_combinations: If set, each bet type list is truncated to the top-K
            entries by EV descending (None rows sort last). Useful for 三連単 (up
            to 4896 combos for 18 horses). None returns all combinations.
        race_odds: Confirmed odds dict from compute_race_odds or compute_past_race_odds.
            Format: {bet_type: {combo: odds}}.
            If provided, per-combo est_odds is taken from race_odds when available;
            combos not present in race_odds get est_odds=None and ev=None.
            If None (default), all combos get est_odds=None and ev=None.

    Returns:
        Dict mapping bet_type name to list of CombinationPrediction.
        Keys: '単勝', '複勝', '馬連', 'ワイド', '馬単', '三連複', '三連単'.
        Each list is sorted by ev descending (None ev rows are placed last).
    """
    if frame.empty:
        return {
            bt: []
            for bt in ["単勝", "複勝", "馬連", "ワイド", "馬単", "三連複", "三連単"]
        }

    base_df = predict_race(model, frame, binary_model=binary_model, calibrator=calibrator)

    # Normalise race_odds — None means no confirmed odds data available
    confirmed: dict[str, dict[str, float]] = race_odds if race_odds is not None else {}
    sources_map: dict[str, dict[str, str]] = (
        race_odds_sources if race_odds_sources is not None else {}
    )

    # Compute all PL combination probs in one MC pass (k=3 for triple support).
    # predict_race sorts by score, so we re-align probabilities back to frame order via horse_id.
    horse_to_score = dict(zip(base_df["horse_id"].values, base_df["score"].values))
    horse_to_win = dict(zip(base_df["horse_id"].values, base_df["win_prob"].values))
    horse_to_place = dict(zip(base_df["horse_id"].values, base_df["place_prob"].values))

    # Build aligned arrays in frame order (one entry per post_position)
    frame_scores = np.array([horse_to_score[hid] for hid in frame["horse_id"].values])
    frame_win_probs = np.array([horse_to_win[hid] for hid in frame["horse_id"].values])
    frame_place_probs = np.array([horse_to_place[hid] for hid in frame["horse_id"].values])
    post_positions = frame["post_position"].values  # post_position per horse, same order as frame

    n = len(frame_scores)
    combo_probs = compute_all_combination_probs(frame_scores, k=3, n_samples=n_samples, rng=rng)

    wide_matrix = derive_wide_prob_from_triple(combo_probs["triple"], n)
    ordered_triple: np.ndarray = combo_probs["ordered_triple"]

    result: dict[str, list[CombinationPrediction]] = {}

    def _est_odds(bet_type: str, combo: str) -> float | None:
        """race_odds から該当 combo の odds を引く。無ければ None。
        baseline へのフォールバックはしない。
        """
        return confirmed.get(bet_type, {}).get(combo)

    def _est_source(bet_type: str, combo: str, has_odds: bool) -> str:
        """est_odds_source の決定ロジック。

        - source dict に明示的な値があればそれを使う ("confirmed" / "implied")
        - 値が無く est_odds が取れた場合 → 後方互換で "confirmed"
        - est_odds 取得不能 → "unknown"
        """
        explicit = sources_map.get(bet_type, {}).get(combo)
        if explicit is not None:
            return explicit
        return "confirmed" if has_odds else "unknown"

    def _sort_key(cp: CombinationPrediction) -> tuple[int, float]:
        """ev が None の行は末尾固定（ev=−∞ 扱い）。"""
        if cp.ev is None:
            return (1, 0.0)
        return (0, -cp.ev)

    # ── 単勝 ──────────────────────────────────────────────────────────────────
    tansho_list: list[CombinationPrediction] = []
    for idx in range(n):
        prob = float(frame_win_probs[idx])
        pp = int(post_positions[idx])
        combo = str(pp)
        est = _est_odds("単勝", combo)
        ev = prob * est if est is not None else None
        tansho_list.append(CombinationPrediction(
            combo=combo,
            prob=prob,
            est_odds=est,
            est_odds_source=_est_source("単勝", combo, est is not None),
            ev=ev,
            post_positions=(pp,),
        ))
    tansho_list.sort(key=_sort_key)
    result["単勝"] = tansho_list[:top_k_combinations] if top_k_combinations else tansho_list

    # ── 複勝 ──────────────────────────────────────────────────────────────────
    fukusho_list: list[CombinationPrediction] = []
    for idx in range(n):
        prob = float(frame_place_probs[idx])
        pp = int(post_positions[idx])
        combo = str(pp)
        est = _est_odds("複勝", combo)
        ev = prob * est if est is not None else None
        fukusho_list.append(CombinationPrediction(
            combo=combo,
            prob=prob,
            est_odds=est,
            est_odds_source=_est_source("複勝", combo, est is not None),
            ev=ev,
            post_positions=(pp,),
        ))
    fukusho_list.sort(key=_sort_key)
    result["複勝"] = fukusho_list[:top_k_combinations] if top_k_combinations else fukusho_list

    # ── 馬連 ──────────────────────────────────────────────────────────────────
    pair_matrix: np.ndarray = combo_probs["pair"]
    umaren_list: list[CombinationPrediction] = []
    for i, j in combinations(range(n), 2):
        prob = float(pair_matrix[i, j])
        pp_i = int(post_positions[i])
        pp_j = int(post_positions[j])
        pp_lo, pp_hi = (pp_i, pp_j) if pp_i <= pp_j else (pp_j, pp_i)
        combo = f"{pp_lo}-{pp_hi}"
        est = _est_odds("馬連", combo)
        ev = prob * est if est is not None else None
        umaren_list.append(CombinationPrediction(
            combo=combo,
            prob=prob,
            est_odds=est,
            est_odds_source=_est_source("馬連", combo, est is not None),
            ev=ev,
            post_positions=(pp_lo, pp_hi),
        ))
    umaren_list.sort(key=_sort_key)
    result["馬連"] = umaren_list[:top_k_combinations] if top_k_combinations else umaren_list

    # ── ワイド ────────────────────────────────────────────────────────────────
    wide_list: list[CombinationPrediction] = []
    for i, j in combinations(range(n), 2):
        prob = float(wide_matrix[i, j])
        pp_i = int(post_positions[i])
        pp_j = int(post_positions[j])
        pp_lo, pp_hi = (pp_i, pp_j) if pp_i <= pp_j else (pp_j, pp_i)
        combo = f"{pp_lo}-{pp_hi}"
        est = _est_odds("ワイド", combo)
        ev = prob * est if est is not None else None
        wide_list.append(CombinationPrediction(
            combo=combo,
            prob=prob,
            est_odds=est,
            est_odds_source=_est_source("ワイド", combo, est is not None),
            ev=ev,
            post_positions=(pp_lo, pp_hi),
        ))
    wide_list.sort(key=_sort_key)
    result["ワイド"] = wide_list[:top_k_combinations] if top_k_combinations else wide_list

    # ── 馬単 ──────────────────────────────────────────────────────────────────
    ordered_pair_matrix: np.ndarray = combo_probs["ordered_pair"]
    umatan_list: list[CombinationPrediction] = []
    for i, j in permutations(range(n), 2):
        prob = float(ordered_pair_matrix[i, j])
        pp_i = int(post_positions[i])
        pp_j = int(post_positions[j])
        combo = f"{pp_i}→{pp_j}"
        est = _est_odds("馬単", combo)
        ev = prob * est if est is not None else None
        umatan_list.append(CombinationPrediction(
            combo=combo,
            prob=prob,
            est_odds=est,
            est_odds_source=_est_source("馬単", combo, est is not None),
            ev=ev,
            post_positions=(pp_i, pp_j),
        ))
    umatan_list.sort(key=_sort_key)
    result["馬単"] = umatan_list[:top_k_combinations] if top_k_combinations else umatan_list

    # ── 三連複 ────────────────────────────────────────────────────────────────
    triple_prob: dict[frozenset, float] = combo_probs["triple"]
    sanrenpuku_list: list[CombinationPrediction] = []
    for i, j, k in combinations(range(n), 3):
        fs = frozenset({i, j, k})
        prob = float(triple_prob.get(fs, 0.0))
        pp_i = int(post_positions[i])
        pp_j = int(post_positions[j])
        pp_k = int(post_positions[k])
        pps = tuple(sorted([pp_i, pp_j, pp_k]))
        combo = f"{pps[0]}-{pps[1]}-{pps[2]}"
        est = _est_odds("三連複", combo)
        ev = prob * est if est is not None else None
        sanrenpuku_list.append(CombinationPrediction(
            combo=combo,
            prob=prob,
            est_odds=est,
            est_odds_source=_est_source("三連複", combo, est is not None),
            ev=ev,
            post_positions=pps,
        ))
    sanrenpuku_list.sort(key=_sort_key)
    result["三連複"] = sanrenpuku_list[:top_k_combinations] if top_k_combinations else sanrenpuku_list

    # ── 三連単 ────────────────────────────────────────────────────────────────
    sanrentan_list: list[CombinationPrediction] = []
    for i, j, k in permutations(range(n), 3):
        prob = float(ordered_triple[i, j, k])
        pp_i = int(post_positions[i])
        pp_j = int(post_positions[j])
        pp_k = int(post_positions[k])
        combo = f"{pp_i}→{pp_j}→{pp_k}"
        est = _est_odds("三連単", combo)
        ev = prob * est if est is not None else None
        sanrentan_list.append(CombinationPrediction(
            combo=combo,
            prob=prob,
            est_odds=est,
            est_odds_source=_est_source("三連単", combo, est is not None),
            ev=ev,
            post_positions=(pp_i, pp_j, pp_k),
        ))
    sanrentan_list.sort(key=_sort_key)
    result["三連単"] = sanrentan_list[:top_k_combinations] if top_k_combinations else sanrentan_list

    return result


def predict_race_with_shap(
    model: lgb.Booster,
    frame: pd.DataFrame,
    top_n: int = 3,
    binary_model: lgb.Booster | None = None,
    calibrator: IsotonicCalibrator | None = None,
) -> pd.DataFrame:
    """Same as predict_race but adds a 'top_features' column (list[str]).

    Uses SHAP TreeExplainer to identify the most influential features for each
    horse. The top_n features by absolute SHAP value are returned per horse.

    Args:
        model: Trained LightGBM Booster (lambdarank).
        frame: Feature DataFrame for one race. Must contain FEATURE_COLUMNS.
        top_n: Number of top features to return per horse.
        binary_model / calibrator: Optional. Forwarded to predict_race so the
            returned win_prob comes from the calibrated path when available.

    Returns:
        DataFrame with columns: horse_id, score, win_prob, place_prob, top_features.
        Sorted by score descending.
    """
    import shap

    base = predict_race(model, frame, binary_model=binary_model, calibrator=calibrator)
    if frame.empty:
        base["top_features"] = pd.Series(dtype=object)
        return base

    X = _prepare_features(frame, model=model)
    feature_names = list(model.feature_name())

    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X)

    # LightGBM lambdarank は (n_samples, n_features) の 2D を返すが、
    # multi-output モデル（例: 多クラス分類）では list of 2D / 3D が返る場合があるため
    # 第 1 出力 (primary) を採用するガードを入れる。
    if isinstance(raw_shap, list):
        shap_values = np.asarray(raw_shap[0])
    else:
        shap_values = np.asarray(raw_shap)
    if shap_values.ndim == 3:
        shap_values = shap_values[..., 0]
    if shap_values.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP value shape {shap_values.shape}; expected 2D (n_samples, n_features)"
        )

    top_features_list: list[list[str]] = []
    for i in range(len(X)):
        abs_vals = np.abs(shap_values[i])
        # argsort ascending, take last top_n in reverse
        sorted_idx = np.argsort(abs_vals)[::-1][:top_n]
        top_features_list.append([feature_names[j] for j in sorted_idx])

    # Align top_features with sorted prediction order via horse_id
    horse_to_features = dict(zip(frame["horse_id"].values, top_features_list, strict=False))
    base["top_features"] = base["horse_id"].map(horse_to_features)
    return base
