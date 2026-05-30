"""Single-race and batch inference.

API レイヤ:
  - 公開 (bundle-aware, 推奨):
      predict_race(bundle, frame)
      predict_race_with_combinations(bundle, frame, ...)
      predict_race_with_shap(bundle, frame, top_n=...)
    bundle.model_type で GBDT / NN を自動切替。呼び出し側が model_type を
    気にする必要がない。新規コードは原則これを使う。NN モデルで SHAP を
    要求した場合は top_features=[] が返る (SHAP TreeExplainer は GBDT 限定)。

  - GBDT 固有 (低レイヤ, 学習時用): predict_race_gbdt /
    predict_race_with_combinations_gbdt / predict_race_with_shap_gbdt
    Booster を直接受け取る。train.py / calibrate.py / evaluate.py など
    bundle 組み立て前の学習パイプラインで使う。

predict_race_gbdt は LightGBM raw scores を softmax + place 推定器で
win_prob / place_prob に変換する。KEIBA_PLACE_PROB_METHOD で実装を選択:
  plackett_luce  (default) — Plackett-Luce Monte Carlo, per-horse probabilities
  heuristic                — legacy top_k_cumulative_prob approximation

predict_race_with_combinations_gbdt は EV 計算 (馬連 / ワイド / 馬単 /
三連複 / 三連単) を追加で行う。
"""

from __future__ import annotations

import os
from itertools import combinations, permutations
from typing import TYPE_CHECKING

import lightgbm as lgb
import numpy as np
import pandas as pd

from ai.calibrate import (
    ComboCalibrators,
    ConditionalIsotonicCalibrator,
    IsotonicCalibrator,
    compute_all_combination_probs,
    compute_place_prob,
    softmax_within_race,
    top_k_cumulative_prob,
)
from ai.types import CombinationPrediction
from features.builder import CATEGORICAL_FEATURES, FEATURE_COLUMNS

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from ai.registry import ModelBundle
    from ai.temperature import TemperatureScaler

_PLACE_PROB_METHOD = os.environ.get("KEIBA_PLACE_PROB_METHOD", "plackett_luce")


def _prepare_features(
    frame: pd.DataFrame, model: lgb.Booster | None = None
) -> pd.DataFrame:
    """Extract and cast feature columns from frame.

    model が渡された場合、学習時の feature_name() を使って列を選ぶ
    （odds 抜きモデルでも正しく動作する）。model=None のときは
    後方互換のため FEATURE_COLUMNS を使う。

    LightGBM は object dtype を受け付けない (TypeError: pandas dtypes must
    be int, float or bool)。inference frame で None / 混在型が入ると pandas
    が object として推論してしまうため、CATEGORICAL_FEATURES 以外は明示的に
    pd.to_numeric (errors='coerce') で float に強制する。
    """
    cols = list(model.feature_name()) if model is not None else FEATURE_COLUMNS
    X = frame[cols].copy()
    for col in X.columns:
        if col in CATEGORICAL_FEATURES:
            X[col] = X[col].astype("category")
        elif X[col].dtype == "object":
            X[col] = pd.to_numeric(X[col], errors="coerce")
    return X


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


def predict_race_gbdt(
    model: lgb.Booster,
    frame: pd.DataFrame,
    binary_model: lgb.Booster | None = None,
    calibrator: IsotonicCalibrator | ConditionalIsotonicCalibrator | None = None,
    loss_type: str | None = None,
    temperature_scaler: TemperatureScaler | None = None,
    place_calibrator: IsotonicCalibrator | None = None,
) -> pd.DataFrame:
    """Score all horses in a single race and return calibrated probabilities.

    Args:
        model: Trained LightGBM Booster (lambdarank or Plackett-Luce).
        frame: Feature DataFrame for one race (output of build_inference_frame
               or a training-frame slice). Must contain FEATURE_COLUMNS.
        binary_model: Optional binary classifier (objective=binary) trained
            in parallel. When provided together with `calibrator`, win_prob
            is computed as `calibrator(binary_model.predict(X))` instead of
            softmax(scores). Ignored in Plackett-Luce mode.
        calibrator: Optional IsotonicCalibrator fit on the validation set.
            Used in tandem with `binary_model`. Ignored in Plackett-Luce mode.
        loss_type: "lambdarank" or "plackett_luce" (or None for backward
            compatibility, treated as "lambdarank").  When "plackett_luce",
            win_prob is softmax(score) within the race — no binary head needed.
        temperature_scaler: Optional TemperatureScaler for post-hoc probability
            calibration.  In PL mode, win_prob = softmax(score / T_win);
            in lambdarank mode, T_win is applied after isotonic calibration
            as an additional multiplicative adjustment.  place_prob always
            uses T_place for PL MC score scaling.  None = identity (backward-compat).

    Returns:
        DataFrame with columns: horse_id, score, win_prob, place_prob.
        Sorted by score descending (top prediction first).

    Notes:
        - The model `score` is always returned for ranking / NDCG evaluation.
        - place_prob is computed from Plackett-Luce MC using model scores for
          both loss types.
        - When loss_type is None and binary_model/calibrator are also None,
          falls back to softmax(scores) for backward compatibility.
    """
    if frame.empty:
        return pd.DataFrame(columns=["horse_id", "score", "win_prob", "place_prob"])

    X = _prepare_features(frame, model=model)
    scores: np.ndarray = model.predict(X)

    is_pl_mode = loss_type == "plackett_luce"

    if is_pl_mode:
        # Plackett-Luce model: softmax(score / T_win) is the calibrated win probability.
        if temperature_scaler is not None:
            win_probs = temperature_scaler.transform_win(scores)
        else:
            win_probs = softmax_within_race(scores)
    elif binary_model is not None and calibrator is not None:
        # Lambdarank + binary head path (Phase 2 onward).
        # binary_model may have been trained with a different feature subset;
        # use its own feature_name() instead of the lambdarank model's.
        X_binary = _prepare_features(frame, model=binary_model)
        raw_win = binary_model.predict(X_binary)
        if isinstance(calibrator, ConditionalIsotonicCalibrator):
            # Build per-entry conditions for the conditional calibrator.
            n_runners_val = (
                int(frame["n_runners"].iloc[0])
                if "n_runners" in frame.columns
                else len(frame)
            )
            cond_df = pd.DataFrame(
                {
                    "surface": (
                        frame["surface"].values
                        if "surface" in frame.columns
                        else ["unknown"] * len(frame)
                    ),
                    "n_runners": n_runners_val,
                }
            )
            win_probs = calibrator.predict(raw_win, cond_df, normalise=True)
        else:
            win_probs = calibrator.predict(raw_win, normalise=True)
        # NOTE: temperature_scaler is intentionally NOT applied here.
        # In lambdarank mode the win_probs are already calibrated probabilities
        # (binary head + isotonic), and applying softmax(probs / T) on top
        # re-normalises a probability distribution as if it were a score vector,
        # flattening the top horse and inflating mid-tier EV → over-betting.
        # train.py also skips fitting a TemperatureScaler in lambdarank mode,
        # so this branch is normally not reached with a non-None scaler, but
        # we keep the guard out of the lambdarank path defensively.
    else:
        # Backward-compat: softmax(lambdarank scores)
        win_probs = softmax_within_race(scores)
        if temperature_scaler is not None:
            win_probs = temperature_scaler.transform_win(win_probs)

    place_temperature = temperature_scaler.T_place if temperature_scaler is not None else 1.0
    place_probs = _compute_place_prob(scores, place_temperature=place_temperature)

    # Post-hoc isotonic calibration of place_prob. Unlike win_prob, place
    # probabilities don't sum to 1 over a race (3 horses place), so we apply a
    # plain monotonic mapping without per-race re-normalisation (normalise=False).
    if place_calibrator is not None:
        place_probs = place_calibrator.predict(place_probs, normalise=False)

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


def predict_race_with_combinations_gbdt(
    model: lgb.Booster,
    frame: pd.DataFrame,
    session: Session | None = None,
    n_samples: int = 10_000,
    rng: np.random.Generator | None = None,
    top_k_combinations: int | None = None,
    race_odds: dict[str, dict[str, float]] | None = None,
    race_odds_sources: dict[str, dict[str, str]] | None = None,
    binary_model: lgb.Booster | None = None,
    calibrator: IsotonicCalibrator | ConditionalIsotonicCalibrator | None = None,
    combo_calibrators: ComboCalibrators | None = None,
    loss_type: str | None = None,
    temperature_scaler: TemperatureScaler | None = None,
) -> dict[str, list[CombinationPrediction]]:
    """Extend predict_race_gbdt with EV calculations for all combination bet types.

    Calls predict_race_gbdt internally (does not modify it) and augments the result
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
        race_odds: Confirmed odds dict from compute_race_odds_with_sources.
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

    # post_position が 1 つでも欠けると combo 文字列を作れず int(None) で
    # crash する (shutuba 取込中の半端なデータで起こる)。combinations は
    # 全頭の post_position が揃って初めて意味を持つので、欠けがあれば全
    # bet_type で空リストを返す。caller (route) は HTTPException でなく
    # 静かに「組合わせ予想なし」を返せる。
    if frame["post_position"].isna().any():
        return {
            bt: []
            for bt in ["単勝", "複勝", "馬連", "ワイド", "馬単", "三連複", "三連単"]
        }

    base_df = predict_race_gbdt(
        model, frame,
        binary_model=binary_model,
        calibrator=calibrator,
        loss_type=loss_type,
        temperature_scaler=temperature_scaler,
    )

    # Normalise race_odds — None means no confirmed odds data available
    confirmed: dict[str, dict[str, float]] = race_odds if race_odds is not None else {}
    sources_map: dict[str, dict[str, str]] = (
        race_odds_sources if race_odds_sources is not None else {}
    )

    # Compute all PL combination probs in one MC pass (k=3 for triple support).
    # predict_race_gbdt sorts by score, so we re-align probabilities back to frame order via horse_id.
    horse_to_score = dict(zip(base_df["horse_id"].values, base_df["score"].values, strict=True))
    horse_to_win = dict(zip(base_df["horse_id"].values, base_df["win_prob"].values, strict=True))
    horse_to_place = dict(zip(base_df["horse_id"].values, base_df["place_prob"].values, strict=True))

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

    # Build a single-row conditions DataFrame for this race (used only when
    # the combo_calibrators contains ConditionalIsotonicCalibrator instances).
    _race_surface = str(frame["surface"].iloc[0]) if "surface" in frame.columns else "unknown"
    _race_n_runners = int(frame["n_runners"].iloc[0]) if "n_runners" in frame.columns else n
    _race_conditions = pd.DataFrame(
        {"surface": [_race_surface], "n_runners": [_race_n_runners]}
    )

    def _calibrate(bet_type: str, prob: float) -> float:
        """連系 馬券種に combo_calibrators が指定されていれば prob を補正する。
        単勝・複勝は呼ばれない (上の専用 path で処理済み)。"""
        if combo_calibrators is None:
            return prob
        if not combo_calibrators.has(bet_type):
            return prob
        # スカラー単発 transform。EV 計算前にここで上書きしておくと
        # ev = prob * est で自動的に calibrated EV になる。
        # ConditionalIsotonicCalibrator の場合は race-level conditions を渡す。
        adjusted = float(
            combo_calibrators.predict(bet_type, np.array([prob]), conditions=_race_conditions)[0]
        )
        # 万一 0 未満になる極端値を 0 にクランプ (iso は y_min=0 だが念のため)
        return max(0.0, min(1.0, adjusted))

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
        prob = _calibrate("馬連", float(pair_matrix[i, j]))
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
        prob = _calibrate("ワイド", float(wide_matrix[i, j]))
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
        prob = _calibrate("馬単", float(ordered_pair_matrix[i, j]))
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
        prob = _calibrate("三連複", float(triple_prob.get(fs, 0.0)))
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
        prob = _calibrate("三連単", float(ordered_triple[i, j, k]))
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


def predict_race_with_shap_gbdt(
    model: lgb.Booster,
    frame: pd.DataFrame,
    top_n: int = 3,
    binary_model: lgb.Booster | None = None,
    calibrator: IsotonicCalibrator | ConditionalIsotonicCalibrator | None = None,
    loss_type: str | None = None,
    temperature_scaler: TemperatureScaler | None = None,
) -> pd.DataFrame:
    """Same as predict_race_gbdt but adds a 'top_features' column (list[str]).

    Uses SHAP TreeExplainer to identify the most influential features for each
    horse. The top_n features by absolute SHAP value are returned per horse.

    Args:
        model: Trained LightGBM Booster (lambdarank or Plackett-Luce).
        frame: Feature DataFrame for one race. Must contain FEATURE_COLUMNS.
        top_n: Number of top features to return per horse.
        binary_model / calibrator: Optional. Forwarded to predict_race_gbdt so the
            returned win_prob comes from the calibrated path when available.
        loss_type: Forwarded to predict_race_gbdt for win_prob computation branching.

    Returns:
        DataFrame with columns: horse_id, score, win_prob, place_prob, top_features.
        Sorted by score descending.
    """
    import shap

    base = predict_race_gbdt(
        model, frame,
        binary_model=binary_model,
        calibrator=calibrator,
        loss_type=loss_type,
        temperature_scaler=temperature_scaler,
    )
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
    shap_values = np.asarray(raw_shap[0]) if isinstance(raw_shap, list) else np.asarray(raw_shap)
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


# ---------------------------------------------------------------------------
# Bundle-aware inference (GBDT or NN dispatch)
# ---------------------------------------------------------------------------


def predict_race(
    bundle: ModelBundle,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Score all horses in a single race using a ModelBundle.

    Dispatches to either the GBDT or NN inference path depending on
    bundle.model_type.

    Args:
        bundle: ModelBundle loaded via registry.load_model_full().
        frame:  Feature DataFrame for one race. Must contain horse_id and
                the feature columns used at training time.

    Returns:
        DataFrame with columns: horse_id, score, win_prob, place_prob.
        Sorted by score descending.
    """
    if bundle.model_type == "nn":
        return _predict_race_nn(bundle, frame)

    return predict_race_gbdt(
        bundle.lambdarank,
        frame,
        binary_model=bundle.binary,
        calibrator=bundle.calibrator,
        loss_type=bundle.meta.get("loss_type"),
        temperature_scaler=bundle.temperature_scaler,
        place_calibrator=bundle.place_calibrator,
    )


def _predict_race_nn(bundle: ModelBundle, frame: pd.DataFrame) -> pd.DataFrame:
    """NN inference for a single race.

    Builds a single-batch tensor from frame, runs the RaceModel forward pass,
    converts scores to win_prob (softmax) and place_prob (Plackett-Luce MC),
    and returns a DataFrame with the same schema as predict_race_gbdt.

    torch は遅延 import — torch が入っていない環境では呼ばれない想定。
    """
    import torch  # noqa: PLC0415 — intentional lazy import

    if frame.empty:
        return pd.DataFrame(columns=["horse_id", "score", "win_prob", "place_prob"])

    horse_feature_cols: list[str] = bundle.nn_horse_feature_cols or []
    race_feature_cols: list[str] = bundle.nn_race_feature_cols or []
    all_feat_cols = horse_feature_cols + race_feature_cols

    # GBDT stacking: apply the same augmentation the NN saw at training time
    # BEFORE the preprocessor (which was fit on augmented frames).
    if bundle.nn_gbdt_bundle is not None:
        from ai.nn.stacking import augment_frame_with_gbdt  # noqa: PLC0415
        frame = augment_frame_with_gbdt(frame, bundle.nn_gbdt_bundle)

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

    # Post-hoc isotonic calibration on win_prob (fixes NN over-confidence on
    # longshots / under-confidence on top picks). predict() re-normalises per
    # race so the output is still a valid distribution.
    if bundle.nn_calibrator is not None:
        win_probs = bundle.nn_calibrator.predict(win_probs, normalise=True)

    place_temperature = ts.T_place if ts is not None else 1.0
    place_probs = _compute_place_prob(scores, place_temperature=place_temperature)

    # Post-hoc isotonic calibration of place_prob (no per-race renormalisation).
    if bundle.place_calibrator is not None:
        place_probs = bundle.place_calibrator.predict(place_probs, normalise=False)

    # GBDT ensemble (inference-time blending of win/place prob; ranking score
    # stays as the NN's so combinations and ordering are unaffected).  Skipped
    # when weight==1.0 (pure NN) or no ensemble bundle is configured.
    if (
        bundle.nn_ensemble_gbdt_bundle is not None
        and bundle.nn_ensemble_weight < 1.0
    ):
        w = float(bundle.nn_ensemble_weight)
        gbdt_preds = predict_race(bundle.nn_ensemble_gbdt_bundle, frame)
        gbdt_by_horse: dict[str, tuple[float, float]] = {
            str(row["horse_id"]): (float(row["win_prob"]), float(row["place_prob"]))
            for _, row in gbdt_preds.iterrows()
        }
        nn_horse_ids = frame["horse_id"].values
        blended_win = np.empty_like(win_probs)
        blended_place = np.empty_like(place_probs)
        for i, hid in enumerate(nn_horse_ids):
            g_win, g_place = gbdt_by_horse.get(str(hid), (win_probs[i], place_probs[i]))
            blended_win[i] = w * win_probs[i] + (1.0 - w) * g_win
            blended_place[i] = w * place_probs[i] + (1.0 - w) * g_place
        win_probs = blended_win
        place_probs = blended_place

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
    """Bundle-aware wrapper around predict_race_with_combinations_gbdt.

    Dispatches to GBDT or NN inference path depending on bundle.model_type.
    The combination probability computation (Plackett-Luce MC) is shared
    between both paths — only the base predict_race_gbdt call differs.

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
    if bundle.model_type == "nn":
        # For NN: compute base predictions via _predict_race_nn, then reuse
        # the GBDT combination path with the NN scores directly.
        base_df = _predict_race_nn(bundle, frame)
        if frame.empty or frame["post_position"].isna().any():
            return {
                bt: []
                for bt in ["単勝", "複勝", "馬連", "ワイド", "馬単", "三連複", "三連単"]
            }
        # Build a surrogate frame aligned with base_df for the combination engine.
        # predict_race_with_combinations_gbdt expects the original frame (with post_position),
        # so we use it directly but pass a dummy GBDT model.  Instead, we replicate
        # the core combination logic here using NN scores.
        return _combinations_from_base(
            base_df=base_df,
            frame=frame,
            n_samples=n_samples,
            rng=rng,
            top_k_combinations=top_k_combinations,
            race_odds=race_odds,
            race_odds_sources=race_odds_sources,
            combo_calibrators=bundle.combo_calibrators,
        )

    # GBDT 経路: 既存関数に委譲
    return predict_race_with_combinations_gbdt(
        bundle.lambdarank,
        frame,
        session=session,
        n_samples=n_samples,
        rng=rng,
        top_k_combinations=top_k_combinations,
        race_odds=race_odds,
        race_odds_sources=race_odds_sources,
        binary_model=bundle.binary,
        calibrator=bundle.calibrator,
        combo_calibrators=bundle.combo_calibrators,
        loss_type=bundle.meta.get("loss_type"),
        temperature_scaler=bundle.temperature_scaler,
    )


def predict_race_with_shap(
    bundle: ModelBundle,
    frame: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """Bundle-aware: predict + SHAP top features.

    SHAP TreeExplainer は GBDT 限定の機能。NN モデルの場合は top_features に
    空リストを入れて返す (UI 側で「説明なし」表示にできる)。

    Returns:
        DataFrame with columns: horse_id, score, win_prob, place_prob, top_features.
    """
    if bundle.model_type == "nn":
        result_df = predict_race(bundle, frame)
        result_df["top_features"] = [[] for _ in range(len(result_df))]
        return result_df

    return predict_race_with_shap_gbdt(
        bundle.lambdarank,
        frame,
        top_n=top_n,
        binary_model=bundle.binary,
        calibrator=bundle.calibrator,
        loss_type=bundle.meta.get("loss_type"),
        temperature_scaler=bundle.temperature_scaler,
    )


def _combinations_from_base(
    base_df: pd.DataFrame,
    frame: pd.DataFrame,
    n_samples: int,
    rng: np.random.Generator | None,
    top_k_combinations: int | None,
    race_odds: dict[str, dict[str, float]] | None,
    race_odds_sources: dict[str, dict[str, str]] | None,
    combo_calibrators: ComboCalibrators | None,
) -> dict[str, list[CombinationPrediction]]:
    """Shared combination computation given a pre-computed base_df.

    base_df must have columns: horse_id, score, win_prob, place_prob.
    frame must have columns: horse_id, post_position.

    This mirrors the combination logic in predict_race_with_combinations_gbdt but
    accepts an already-computed base_df instead of a LightGBM model.
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
        if combo_calibrators is None:
            return prob
        if not combo_calibrators.has(bet_type):
            return prob
        adjusted = float(combo_calibrators.predict(bet_type, np.array([prob]))[0])
        return max(0.0, min(1.0, adjusted))

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
