"""Single-race and batch inference for NN models.

公開 API (bundle-aware):
    predict_race(bundle, frame)
    predict_race_with_combinations(bundle, frame, ...)
    predict_race_with_shap(bundle, frame, top_n=...)

bundle は registry.load_model_full() が返す NN ModelBundle。win_prob /
place_prob への変換と各 馬券種の EV 計算 (馬連 / ワイド / 馬単 / 三連複 /
三連単) を行う。place_prob の実装は KEIBA_PLACE_PROB_METHOD で選択:
  plackett_luce  (default) — Plackett-Luce Monte Carlo, per-horse probabilities
  heuristic                — legacy top_k_cumulative_prob approximation

SHAP による特徴量寄与は廃止済み。predict_race_with_shap は互換のため残置し、
常に top_features=[] を返す。
"""

from __future__ import annotations

import os
from itertools import combinations, permutations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ai.calibrate import (
    compute_all_combination_probs,
    compute_place_prob,
    softmax_within_race,
    top_k_cumulative_prob,
)
from ai.types import CombinationPrediction
from core.bet_types import COMBINATION_BET_TYPES

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from ai.registry import ModelBundle

_PLACE_PROB_METHOD = os.environ.get("KEIBA_PLACE_PROB_METHOD", "plackett_luce")


def _compute_place_prob(scores: np.ndarray, place_temperature: float = 1.0) -> np.ndarray:
    """Dispatch to the configured place-probability estimator.

    Reads KEIBA_PLACE_PROB_METHOD at call time so that monkeypatching the
    module-level variable in tests takes immediate effect.

    Args:
        scores: Raw model scores for n horses.
        place_temperature: Temperature divisor for PL score scaling (TemperatureScaler).
            1.0 = no scaling (default, backward-compatible).
    """
    method = os.environ.get("KEIBA_PLACE_PROB_METHOD", _PLACE_PROB_METHOD)
    if method == "heuristic":
        # Temperature not applied for heuristic method (legacy path)
        return top_k_cumulative_prob(scores, k=3)
    # Default: plackett_luce with optional temperature scaling
    return compute_place_prob(scores, k=3, n_samples=10_000, place_temperature=place_temperature)


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


# ---------------------------------------------------------------------------
# Bundle-aware inference (NN)
# ---------------------------------------------------------------------------


def predict_race(
    bundle: ModelBundle,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Score all horses in a single race using a NN ModelBundle.

    Args:
        bundle: ModelBundle loaded via registry.load_model_full().
        frame:  Feature DataFrame for one race. Must contain horse_id and
                the feature columns used at training time.

    Returns:
        DataFrame with columns: horse_id, score, win_prob, place_prob.
        Sorted by score descending.
    """
    return _predict_race_nn(bundle, frame)


def _predict_race_nn(bundle: ModelBundle, frame: pd.DataFrame) -> pd.DataFrame:
    """NN inference for a single race.

    Builds a single-batch tensor from frame, runs the RaceModel forward pass,
    converts scores to win_prob (softmax) and place_prob (Plackett-Luce MC),
    and returns a DataFrame with columns horse_id, score, win_prob, place_prob.

    torch は遅延 import — torch が入っていない環境では呼ばれない想定。
    """
    import torch  # noqa: PLC0415 — intentional lazy import

    if frame.empty:
        return pd.DataFrame(columns=["horse_id", "score", "win_prob", "place_prob"])

    horse_feature_cols: list[str] = bundle.nn_horse_feature_cols or []
    race_feature_cols: list[str] = bundle.nn_race_feature_cols or []
    all_feat_cols = horse_feature_cols + race_feature_cols

    if bundle.nn_preprocessor is not None:
        encoded = bundle.nn_preprocessor.transform(frame)
    else:
        # Legacy fallback: NN models saved before preprocessor.pkl was introduced.
        # The mapping is computed from the single race only, which means the
        # categorical encoding will not match what the model was trained with.
        from ai.nn.train_nn import _encode_categoricals  # noqa: PLC0415
        encoded = _encode_categoricals(frame, all_feat_cols)

    n_horses = len(encoded)

    # Horse features: [1, n_horses, horse_feat_dim]
    if horse_feature_cols:
        hf_np = encoded[horse_feature_cols].values.astype("float32")
    else:
        hf_np = np.zeros((n_horses, 0), dtype="float32")
    hf = torch.tensor(hf_np, dtype=torch.float32).unsqueeze(0)  # [1, n, F]

    # Race features: [1, race_feat_dim]
    if race_feature_cols:
        rf_np = encoded[race_feature_cols].iloc[0].values.astype("float32")
    else:
        rf_np = np.zeros(0, dtype="float32")
    rf = torch.tensor(rf_np, dtype=torch.float32).unsqueeze(0)  # [1, R]

    mask = torch.ones(1, n_horses, dtype=torch.bool)  # all valid

    model = bundle.nn_model
    assert model is not None, "nn_model is None in NN bundle"

    with torch.no_grad():
        scores_t = model(hf, rf, mask)  # [1, n_horses]
    scores: np.ndarray = scores_t[0, :n_horses].cpu().numpy()

    # Masked -inf positions (shouldn't occur for single-race batch) → replace with min
    finite_mask = np.isfinite(scores)
    if not finite_mask.all():
        scores = np.where(finite_mask, scores, scores[finite_mask].min() if finite_mask.any() else 0.0)

    ts = bundle.temperature_scaler
    win_probs = ts.transform_win(scores) if ts is not None else softmax_within_race(scores)

    place_temperature = ts.T_place if ts is not None else 1.0
    place_probs = _compute_place_prob(scores, place_temperature=place_temperature)

    result = pd.DataFrame(
        {
            "horse_id": frame["horse_id"].values,
            "score": scores,
            "win_prob": win_probs,
            "place_prob": place_probs,
        }
    )
    return result.sort_values("score", ascending=False).reset_index(drop=True)


def predict_race_with_combinations(
    bundle: ModelBundle,
    frame: pd.DataFrame,
    session: Session | None = None,
    n_samples: int = 10_000,
    rng: np.random.Generator | None = None,
    top_k_combinations: int | None = None,
    race_odds: dict[str, dict[str, float]] | None = None,
    race_odds_sources: dict[str, dict[str, str]] | None = None,
) -> dict[str, list[CombinationPrediction]]:
    """Predict + per-combination EV for all 馬券種 using a NN ModelBundle.

    Computes base predictions via the NN forward pass, then derives the
    Plackett-Luce combination probabilities and EVs.

    Args:
        bundle:    ModelBundle loaded via registry.load_model_full().
        frame:     Feature DataFrame for one race.
        session:   Unused; retained for API compatibility.
        n_samples: Number of Plackett-Luce Monte Carlo samples.
        rng:       Optional Generator for reproducibility.
        top_k_combinations: Truncate each bet type list to top-K by EV.
        race_odds: Confirmed odds dict (bet_type → combo → odds).
        race_odds_sources: Source labels for odds.

    Returns:
        Dict mapping bet_type name to list of CombinationPrediction.
    """
    base_df = _predict_race_nn(bundle, frame)
    if frame.empty or frame["post_position"].isna().any():
        return {bt: [] for bt in COMBINATION_BET_TYPES}
    return _combinations_from_base(
        base_df=base_df,
        frame=frame,
        n_samples=n_samples,
        rng=rng,
        top_k_combinations=top_k_combinations,
        race_odds=race_odds,
        race_odds_sources=race_odds_sources,
    )


def predict_race_with_shap(
    bundle: ModelBundle,
    frame: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """Predict + (NN) empty top_features.

    SHAP による特徴量寄与は廃止済み。top_features には空リストを入れて返す
    (UI 側で「説明なし」表示にできる)。``top_n`` はシグネチャ互換のため残すが未使用。

    Returns:
        DataFrame with columns: horse_id, score, win_prob, place_prob, top_features.
    """
    result_df = predict_race(bundle, frame)
    result_df["top_features"] = [[] for _ in range(len(result_df))]
    return result_df


def _combinations_from_base(
    base_df: pd.DataFrame,
    frame: pd.DataFrame,
    n_samples: int,
    rng: np.random.Generator | None,
    top_k_combinations: int | None,
    race_odds: dict[str, dict[str, float]] | None,
    race_odds_sources: dict[str, dict[str, str]] | None,
) -> dict[str, list[CombinationPrediction]]:
    """Shared combination computation given a pre-computed base_df.

    base_df must have columns: horse_id, score, win_prob, place_prob.
    frame must have columns: horse_id, post_position.

    This mirrors the combination logic used by predict_race_with_combinations but
    accepts an already-computed base_df of NN scores.
    """
    confirmed: dict[str, dict[str, float]] = race_odds if race_odds is not None else {}
    sources_map: dict[str, dict[str, str]] = (
        race_odds_sources if race_odds_sources is not None else {}
    )

    horse_to_score = dict(zip(base_df["horse_id"].values, base_df["score"].values, strict=True))
    horse_to_win = dict(zip(base_df["horse_id"].values, base_df["win_prob"].values, strict=True))
    horse_to_place = dict(zip(base_df["horse_id"].values, base_df["place_prob"].values, strict=True))

    frame_scores = np.array([horse_to_score[hid] for hid in frame["horse_id"].values])
    frame_win_probs = np.array([horse_to_win[hid] for hid in frame["horse_id"].values])
    frame_place_probs = np.array([horse_to_place[hid] for hid in frame["horse_id"].values])
    post_positions = frame["post_position"].values

    n = len(frame_scores)
    combo_probs = compute_all_combination_probs(frame_scores, k=3, n_samples=n_samples, rng=rng)

    wide_matrix = derive_wide_prob_from_triple(combo_probs["triple"], n)
    ordered_triple: np.ndarray = combo_probs["ordered_triple"]

    result: dict[str, list[CombinationPrediction]] = {}

    def _est_odds(bet_type: str, combo: str) -> float | None:
        return confirmed.get(bet_type, {}).get(combo)

    def _calibrate(bet_type: str, prob: float) -> float:
        # Combo calibration is learned inside the NN (combo_nll / multi loss);
        # no external isotonic post-processing.  Kept as identity for call-site
        # symmetry with the per-bet-type combo construction below.
        return prob

    def _est_source(bet_type: str, combo: str, has_odds: bool) -> str:
        explicit = sources_map.get(bet_type, {}).get(combo)
        if explicit is not None:
            return explicit
        return "confirmed" if has_odds else "unknown"

    def _sort_key(cp: CombinationPrediction) -> tuple[int, float]:
        if cp.ev is None:
            return (1, 0.0)
        return (0, -cp.ev)

    # 単勝
    tansho_list: list[CombinationPrediction] = []
    for idx in range(n):
        prob = float(frame_win_probs[idx])
        pp = int(post_positions[idx])
        combo = str(pp)
        est = _est_odds("単勝", combo)
        ev = prob * est if est is not None else None
        tansho_list.append(CombinationPrediction(
            combo=combo, prob=prob, est_odds=est,
            est_odds_source=_est_source("単勝", combo, est is not None),
            ev=ev, post_positions=(pp,),
        ))
    tansho_list.sort(key=_sort_key)
    result["単勝"] = tansho_list[:top_k_combinations] if top_k_combinations else tansho_list

    # 複勝
    fukusho_list: list[CombinationPrediction] = []
    for idx in range(n):
        prob = float(frame_place_probs[idx])
        pp = int(post_positions[idx])
        combo = str(pp)
        est = _est_odds("複勝", combo)
        ev = prob * est if est is not None else None
        fukusho_list.append(CombinationPrediction(
            combo=combo, prob=prob, est_odds=est,
            est_odds_source=_est_source("複勝", combo, est is not None),
            ev=ev, post_positions=(pp,),
        ))
    fukusho_list.sort(key=_sort_key)
    result["複勝"] = fukusho_list[:top_k_combinations] if top_k_combinations else fukusho_list

    # 馬連
    pair_matrix: np.ndarray = combo_probs["pair"]
    umaren_list: list[CombinationPrediction] = []
    for i, j in combinations(range(n), 2):
        prob = _calibrate("馬連", float(pair_matrix[i, j]))
        pp_i, pp_j = int(post_positions[i]), int(post_positions[j])
        pp_lo, pp_hi = (pp_i, pp_j) if pp_i <= pp_j else (pp_j, pp_i)
        combo = f"{pp_lo}-{pp_hi}"
        est = _est_odds("馬連", combo)
        ev = prob * est if est is not None else None
        umaren_list.append(CombinationPrediction(
            combo=combo, prob=prob, est_odds=est,
            est_odds_source=_est_source("馬連", combo, est is not None),
            ev=ev, post_positions=(pp_lo, pp_hi),
        ))
    umaren_list.sort(key=_sort_key)
    result["馬連"] = umaren_list[:top_k_combinations] if top_k_combinations else umaren_list

    # ワイド
    wide_list: list[CombinationPrediction] = []
    for i, j in combinations(range(n), 2):
        prob = _calibrate("ワイド", float(wide_matrix[i, j]))
        pp_i, pp_j = int(post_positions[i]), int(post_positions[j])
        pp_lo, pp_hi = (pp_i, pp_j) if pp_i <= pp_j else (pp_j, pp_i)
        combo = f"{pp_lo}-{pp_hi}"
        est = _est_odds("ワイド", combo)
        ev = prob * est if est is not None else None
        wide_list.append(CombinationPrediction(
            combo=combo, prob=prob, est_odds=est,
            est_odds_source=_est_source("ワイド", combo, est is not None),
            ev=ev, post_positions=(pp_lo, pp_hi),
        ))
    wide_list.sort(key=_sort_key)
    result["ワイド"] = wide_list[:top_k_combinations] if top_k_combinations else wide_list

    # 馬単
    ordered_pair_matrix: np.ndarray = combo_probs["ordered_pair"]
    umatan_list: list[CombinationPrediction] = []
    for i, j in permutations(range(n), 2):
        prob = _calibrate("馬単", float(ordered_pair_matrix[i, j]))
        pp_i, pp_j = int(post_positions[i]), int(post_positions[j])
        combo = f"{pp_i}→{pp_j}"
        est = _est_odds("馬単", combo)
        ev = prob * est if est is not None else None
        umatan_list.append(CombinationPrediction(
            combo=combo, prob=prob, est_odds=est,
            est_odds_source=_est_source("馬単", combo, est is not None),
            ev=ev, post_positions=(pp_i, pp_j),
        ))
    umatan_list.sort(key=_sort_key)
    result["馬単"] = umatan_list[:top_k_combinations] if top_k_combinations else umatan_list

    # 三連複
    triple_prob: dict[frozenset, float] = combo_probs["triple"]
    sanrenpuku_list: list[CombinationPrediction] = []
    for i, j, k in combinations(range(n), 3):
        fs = frozenset({i, j, k})
        prob = _calibrate("三連複", float(triple_prob.get(fs, 0.0)))
        pp_i, pp_j, pp_k = int(post_positions[i]), int(post_positions[j]), int(post_positions[k])
        pps = tuple(sorted([pp_i, pp_j, pp_k]))
        combo = f"{pps[0]}-{pps[1]}-{pps[2]}"
        est = _est_odds("三連複", combo)
        ev = prob * est if est is not None else None
        sanrenpuku_list.append(CombinationPrediction(
            combo=combo, prob=prob, est_odds=est,
            est_odds_source=_est_source("三連複", combo, est is not None),
            ev=ev, post_positions=pps,
        ))
    sanrenpuku_list.sort(key=_sort_key)
    result["三連複"] = sanrenpuku_list[:top_k_combinations] if top_k_combinations else sanrenpuku_list

    # 三連単
    sanrentan_list: list[CombinationPrediction] = []
    for i, j, k in permutations(range(n), 3):
        prob = _calibrate("三連単", float(ordered_triple[i, j, k]))
        pp_i, pp_j, pp_k = int(post_positions[i]), int(post_positions[j]), int(post_positions[k])
        combo = f"{pp_i}→{pp_j}→{pp_k}"
        est = _est_odds("三連単", combo)
        ev = prob * est if est is not None else None
        sanrentan_list.append(CombinationPrediction(
            combo=combo, prob=prob, est_odds=est,
            est_odds_source=_est_source("三連単", combo, est is not None),
            ev=ev, post_positions=(pp_i, pp_j, pp_k),
        ))
    sanrentan_list.sort(key=_sort_key)
    result["三連単"] = sanrentan_list[:top_k_combinations] if top_k_combinations else sanrentan_list

    return result
