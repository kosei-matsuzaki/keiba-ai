"""Probability calibration helpers.

softmax_within_race: standard approach used in M4.
top_k_cumulative_prob: simple approximation for place probability.
plackett_luce_*: Plackett-Luce Monte Carlo probability estimators.
compute_all_combination_probs: compute all combination probabilities in one pass.
compute_analytical_combo_probs: closed-form (no MC) combination probabilities.
IsotonicCalibrator: post-hoc calibrator for binary-classifier win probabilities.
"""

from __future__ import annotations

from itertools import combinations, permutations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from core.bet_types import RENKEI_BET_TYPES


def softmax_within_race(scores: np.ndarray) -> np.ndarray:
    """Convert raw LightGBM lambdarank scores to win probabilities via softmax.

    Subtracts max for numerical stability before exponentiation.
    """
    shifted = scores - scores.max()
    exp_s = np.exp(shifted)
    return exp_s / exp_s.sum()


def top_k_cumulative_prob(scores: np.ndarray, k: int = 3) -> np.ndarray:
    """Approximate each horse's probability of finishing in the top-k.

    For each horse i:
      - If i is in the top-k by win_prob: place_prob[i] = sum of win_prob for
        the top-k group (a strict upper bound: i is one of the top-k).
      - Otherwise: place_prob[i] = sum of the top-(k-1) win_probs (a strict
        lower bound: i would have to displace one of the existing top-k to
        get there, so its prob is at most the leftover share of the top-k
        bucket after removing the weakest top-k horse).

    Top-k horses thus get a strictly higher place_prob than non-top-k horses
    whenever the k-th win_prob is non-zero. This is a heuristic — see
    plackett_luce_place_prob for the theoretically correct alternative.
    """
    win_probs = softmax_within_race(scores)
    n = len(win_probs)
    effective_k = min(k, n)

    order = np.argsort(-win_probs)  # descending
    top_set = set(order[:effective_k].tolist())
    place_probs = np.zeros(n)

    top_k_mass = float(win_probs[order[:effective_k]].sum())
    # Non-top-k horses cannot be in top-k unless they displace the weakest
    # top-k member, so their cumulative bound is sum of top-(k-1).
    top_km1_mass = float(win_probs[order[: effective_k - 1]].sum()) if effective_k >= 1 else 0.0

    for i in range(n):
        place_probs[i] = top_k_mass if i in top_set else top_km1_mass

    # Clip to [0, 1] due to approximation error
    return np.clip(place_probs, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Plackett-Luce Monte Carlo
# ---------------------------------------------------------------------------

def sample_top_k(
    scores: np.ndarray,
    k: int = 3,
    n_samples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw n_samples ordered top-k sequences under the Plackett-Luce model.

    Uses the Gumbel-Top-k trick: adding i.i.d. Gumbel(0,1) noise to
    log-probabilities and taking the top-k by perturbed value is mathematically
    equivalent to sequential Plackett-Luce sampling.  This is fully vectorised
    over all samples at once — no Python loop per sample.

    Theory: if p = softmax(scores), then
        perturbed_i = log(p_i) + Gumbel_i
                    = shifted_i + Gumbel_i        (shift cancels in argmax)
    and argtop-k of perturbed gives an exact PL draw.  Gumbel samples are
    generated via the inverse-CDF transform: G = -log(-log(U)), U ~ Uniform.

    Args:
        scores: Raw model scores for n horses (any shape (n,)).
        k: Number of positions to fill per sample.
        n_samples: Number of Monte Carlo draws.
        rng: Optional Generator for reproducibility.  Defaults to
            np.random.default_rng() (non-reproducible).

    Returns:
        Array of shape (n_samples, k).  result[s, r] is the horse index
        that finished in position r+1 in sample s (0-indexed position).

    Raises:
        ValueError: If scores is empty or k exceeds the number of horses.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(scores)
    if n == 0:
        raise ValueError("scores must not be empty")
    if k > n:
        raise ValueError(f"k={k} exceeds number of horses {n}")

    # Numerical stability: shift by max so scores act as log-probabilities
    # relative to the max (the softmax constant cancels in argtop-k).
    shifted = scores - scores.max()

    # Gumbel samples via inverse-CDF: -log(-log(U)), U in (tiny, 1).
    # tiny lower bound avoids log(0); upper bound 1.0 is exclusive by default.
    u = rng.uniform(low=np.finfo(float).tiny, high=1.0, size=(n_samples, n))
    gumbels = -np.log(-np.log(u))
    perturbed = shifted[None, :] + gumbels  # shape: (n_samples, n)

    # Partial sort is sufficient: argpartition finds top-k bucket in O(n),
    # then argsort within the bucket restores descending order.
    if k == n:
        top_k_idx = np.argsort(-perturbed, axis=1)
    else:
        partition = np.argpartition(-perturbed, k, axis=1)[:, :k]
        sub_perturbed = np.take_along_axis(-perturbed, partition, axis=1)
        order_within = np.argsort(sub_perturbed, axis=1)
        top_k_idx = np.take_along_axis(partition, order_within, axis=1)

    return top_k_idx.astype(np.intp)


def compute_place_prob(
    scores: np.ndarray,
    k: int = 3,
    n_samples: int = 10_000,
    rng: np.random.Generator | None = None,
    place_temperature: float = 1.0,
) -> np.ndarray:
    """Estimate each horse's probability of finishing in the top-k.

    Thin wrapper around plackett_luce_place_prob that applies optional
    temperature scaling to the scores before sampling.  When place_temperature
    is 1.0 (default) the function is identical to plackett_luce_place_prob.

    Args:
        scores: Raw model scores for n horses.
        k: Top-k threshold (default 3 for place bet).
        n_samples: Monte Carlo sample count.
        rng: Optional Generator for reproducibility.
        place_temperature: Temperature divisor applied to scores before PL MC.
            > 1 flattens the distribution (suppresses over-betting),
            < 1 sharpens it, = 1 is the identity.

    Returns:
        Array of shape (n,) where result[i] = P(horse i finishes in top k).
    """
    scaled = scores / place_temperature if place_temperature != 1.0 else scores
    return plackett_luce_place_prob(scaled, k=k, n_samples=n_samples, rng=rng)


def plackett_luce_place_prob(
    scores: np.ndarray,
    k: int = 3,
    n_samples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Estimate each horse's probability of finishing in the top-k.

    Uses Plackett-Luce Monte Carlo sampling.  Unlike top_k_cumulative_prob,
    each horse receives a distinct probability derived from actual simulated
    orderings rather than a shared cumulative mass shared by all top-k horses.

    Args:
        scores: Raw model scores for n horses.
        k: Top-k threshold (default 3 for place bet).
        n_samples: Monte Carlo sample count.
        rng: Optional Generator for reproducibility.

    Returns:
        Array of shape (n,) where result[i] = P(horse i finishes in top k).
        Values are in [0, 1] and sum to approximately k.
    """
    samples = sample_top_k(scores, k=k, n_samples=n_samples, rng=rng)
    n = len(scores)
    counts = np.zeros(n, dtype=np.float64)
    # samples shape: (n_samples, k) — every horse index that appeared counts
    np.add.at(counts, samples.ravel(), 1.0)
    return counts / n_samples


def plackett_luce_position_prob(
    scores: np.ndarray,
    max_position: int = 3,
    n_samples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Estimate P(horse i finishes exactly at position r) for r in 1..max_position.

    Args:
        scores: Raw model scores for n horses.
        max_position: Number of positions to estimate (1-indexed).
        n_samples: Monte Carlo sample count.
        rng: Optional Generator for reproducibility.

    Returns:
        Array of shape (n, max_position).  result[i, r] = P(horse i is at
        position r+1, i.e. 1-indexed position r+1).  Column 0 = 1st place.
    """
    samples = sample_top_k(scores, k=max_position, n_samples=n_samples, rng=rng)
    n = len(scores)
    counts = np.zeros((n, max_position), dtype=np.float64)
    for pos in range(samples.shape[1]):
        np.add.at(counts[:, pos], samples[:, pos], 1.0)
    return counts / n_samples


def plackett_luce_pair_prob(
    scores: np.ndarray,
    n_samples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Estimate P({i, j} are both in top-2) — unordered pair (馬連).

    Args:
        scores: Raw model scores for n horses.
        n_samples: Monte Carlo sample count.
        rng: Optional Generator for reproducibility.

    Returns:
        Symmetric array of shape (n, n).  result[i, j] = result[j, i] =
        P(i and j both finish in top 2).  Diagonal is 0.
    """
    samples = sample_top_k(scores, k=2, n_samples=n_samples, rng=rng)
    n = len(scores)
    counts = np.zeros((n, n), dtype=np.float64)
    i_idx = samples[:, 0]
    j_idx = samples[:, 1]
    np.add.at(counts, (i_idx, j_idx), 1.0)
    np.add.at(counts, (j_idx, i_idx), 1.0)
    return counts / n_samples


def plackett_luce_ordered_pair_prob(
    scores: np.ndarray,
    n_samples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Estimate P(i finishes 1st AND j finishes 2nd) — ordered pair (馬単).

    Args:
        scores: Raw model scores for n horses.
        n_samples: Monte Carlo sample count.
        rng: Optional Generator for reproducibility.

    Returns:
        Array of shape (n, n).  result[i, j] = P(i=1st, j=2nd).
        Diagonal is 0.  Not symmetric.
    """
    samples = sample_top_k(scores, k=2, n_samples=n_samples, rng=rng)
    n = len(scores)
    counts = np.zeros((n, n), dtype=np.float64)
    np.add.at(counts, (samples[:, 0], samples[:, 1]), 1.0)
    return counts / n_samples


def plackett_luce_triple_prob(
    scores: np.ndarray,
    n_samples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> dict[frozenset, float]:
    """Estimate P({i, j, k} are all in top-3) — unordered triple (3連複).

    Args:
        scores: Raw model scores for n horses.
        n_samples: Monte Carlo sample count.
        rng: Optional Generator for reproducibility.

    Returns:
        Dict mapping frozenset({i, j, k}) to empirical probability.
        Only combinations that appeared at least once are present.
        Uses frozenset keys (not ndarray) because the bet is symmetric —
        order within the top-3 group is irrelevant.  For order-sensitive
        probabilities see plackett_luce_ordered_triple_prob.
    """
    samples = sample_top_k(scores, k=3, n_samples=n_samples, rng=rng)
    counts: dict[frozenset, int] = {}
    for row in samples:
        key = frozenset(row.tolist())
        counts[key] = counts.get(key, 0) + 1
    return {key: cnt / n_samples for key, cnt in counts.items()}


def plackett_luce_ordered_triple_prob(
    scores: np.ndarray,
    n_samples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Estimate P(i=1st, j=2nd, k=3rd) — ordered triple (3連単).

    Args:
        scores: Raw model scores for n horses.
        n_samples: Monte Carlo sample count.
        rng: Optional Generator for reproducibility.

    Returns:
        Array of shape (n, n, n).  result[i, j, k] = P(i=1st, j=2nd, k=3rd).
        Elements where any two indices are equal are 0.
        Uses ndarray (not dict) because the bet is order-sensitive — each
        distinct permutation (i,j,k) is a separate outcome.  For the
        order-insensitive version see plackett_luce_triple_prob.
    """
    samples = sample_top_k(scores, k=3, n_samples=n_samples, rng=rng)
    n = len(scores)
    counts = np.zeros((n, n, n), dtype=np.float64)
    np.add.at(counts, (samples[:, 0], samples[:, 1], samples[:, 2]), 1.0)
    return counts / n_samples


def compute_all_combination_probs(
    scores: np.ndarray,
    k: int = 3,
    n_samples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Compute all Plackett-Luce combination probabilities in a single pass.

    Calls sample_top_k once with the given k and derives all applicable
    probability types from the same sample matrix, avoiding redundant draws.

    Args:
        scores: Raw model scores for n horses.
        k: Top-k positions to simulate (default 3).  Affects which outputs
            are included: pair/ordered_pair require k>=2; triple/ordered_triple
            require k==3 exactly (those keys are omitted for other k values).
        n_samples: Monte Carlo sample count shared across all outputs.
        rng: Optional Generator for reproducibility.

    Returns:
        Dict that always contains:
          "place"         -> np.ndarray shape (n,)        — P(top-k)
          "position"      -> np.ndarray shape (n, k)      — P(exact position)

        Additionally when k >= 2:
          "pair"          -> np.ndarray shape (n, n)      — unordered top-2
          "ordered_pair"  -> np.ndarray shape (n, n)      — ordered top-2

        Additionally when k == 3:
          "triple"        -> dict[frozenset, float]       — unordered top-3
          "ordered_triple"-> np.ndarray shape (n, n, n)   — ordered top-3
    """
    if rng is None:
        rng = np.random.default_rng()

    samples = sample_top_k(scores, k=k, n_samples=n_samples, rng=rng)
    n = len(scores)

    # place: horse appeared in any of the k positions
    place_counts = np.zeros(n, dtype=np.float64)
    np.add.at(place_counts, samples.ravel(), 1.0)
    place = place_counts / n_samples

    # position: per-column breakdown across all k positions
    position = np.zeros((n, k), dtype=np.float64)
    for pos in range(k):
        np.add.at(position[:, pos], samples[:, pos], 1.0)
    position /= n_samples

    result: dict = {"place": place, "position": position}

    # pair and ordered_pair: only available when k >= 2
    if k >= 2:
        pair = np.zeros((n, n), dtype=np.float64)
        np.add.at(pair, (samples[:, 0], samples[:, 1]), 1.0)
        np.add.at(pair, (samples[:, 1], samples[:, 0]), 1.0)
        pair /= n_samples
        result["pair"] = pair

        ordered_pair = np.zeros((n, n), dtype=np.float64)
        np.add.at(ordered_pair, (samples[:, 0], samples[:, 1]), 1.0)
        ordered_pair /= n_samples
        result["ordered_pair"] = ordered_pair

    # triple and ordered_triple: only meaningful for exactly k=3
    if k == 3:
        triple: dict[frozenset, int] = {}
        for row in samples:
            key = frozenset(row.tolist())
            triple[key] = triple.get(key, 0) + 1
        result["triple"] = {key: cnt / n_samples for key, cnt in triple.items()}

        ordered_triple = np.zeros((n, n, n), dtype=np.float64)
        np.add.at(ordered_triple, (samples[:, 0], samples[:, 1], samples[:, 2]), 1.0)
        ordered_triple /= n_samples
        result["ordered_triple"] = ordered_triple

    return result


# ---------------------------------------------------------------------------
# Analytical (closed-form) Plackett-Luce combination probabilities
# ---------------------------------------------------------------------------
#
# 同じ k=3 PL モデルだが Monte Carlo の代わりに閉じた解析式で正確に計算する。
# MC のサンプル数有限による量子化（10K samples だと最低確率粒度 = 1e-4 で
# 三連単などの低確率 combo が 1/10000, 2/10000... の離散値しか取れない）を
# 完全に解消する。
#
# PL の漸進公式:
#   P(i 1着)              = p_i
#   P(j 2着 | i 1着)      = p_j / (1 - p_i)              (j ≠ i)
#   P(k 3着 | i, j 1-2着) = p_k / (1 - p_i - p_j)        (k ≠ i, j)


def compute_analytical_combo_probs(scores: np.ndarray) -> dict:
    """compute_all_combination_probs の解析式版。MC ノイズなし、k=3 固定。

    入力 scores の softmax を勝率 p に正規化し、PL の漸進式から各券種確率を
    閉じた形で算出する。

    Args:
        scores: PL utility score の配列 (shape (n,))

    Returns:
        compute_all_combination_probs(..., k=3) と同じ 5 キーの dict:
          place         shape (n,)        — P(top-3)
          pair          shape (n, n)      — P(unordered top-2 = {i,j})
          ordered_pair  shape (n, n)      — P(i 1着, j 2着)
          triple        dict[frozenset, float] — P(unordered top-3 = {i,j,k})
          ordered_triple shape (n, n, n)  — P(i 1着, j 2着, k 3着)

    Notes:
        - 計算量 O(n³)。n=18 で 5832 ops、MC (10K samples) より速い。
        - prob ≤ 0 / 確率質量逸脱の case では該当 entry を 0 にして安全に扱う。
    """
    # Softmax to get win probabilities; defensive renormalize.
    shifted = scores - scores.max()
    p = np.exp(shifted)
    p = p / p.sum()
    n = len(p)

    # ── ordered_pair: P(i 1着, j 2着) = p_i * p_j / (1 - p_i) ───────────────
    one_minus_p = 1.0 - p
    # avoid division by zero (only when p_i == 1, which shouldn't occur for n>=2)
    safe_denom = np.where(one_minus_p > 1e-15, one_minus_p, np.inf)
    ordered_pair = (p[:, None] * p[None, :]) / safe_denom[:, None]
    np.fill_diagonal(ordered_pair, 0.0)

    # ── pair (unordered top-2): symmetric sum of ordered_pair ──────────────
    pair = ordered_pair + ordered_pair.T

    # ── ordered_triple: P(i 1着, j 2着, k 3着) ─────────────────────────────
    #   = ordered_pair[i, j] * p_k / (1 - p_i - p_j)
    # Build the 3D tensor with explicit loops (n is small, ~18 max).
    ordered_triple = np.zeros((n, n, n), dtype=np.float64)
    for i in range(n):
        if one_minus_p[i] <= 1e-15:
            continue
        base_i = p[i] / one_minus_p[i]
        for j in range(n):
            if j == i:
                continue
            denom_ij = 1.0 - p[i] - p[j]
            if denom_ij <= 1e-15:
                continue
            base_ij = base_i * p[j] / denom_ij
            for k in range(n):
                if k in (i, j):
                    continue
                ordered_triple[i, j, k] = base_ij * p[k]

    # ── triple (unordered top-3): sum over 6 permutations of (i, j, k) ─────
    triple_dict: dict[frozenset, float] = {}
    for i, j, k in combinations(range(n), 3):
        total = 0.0
        for a, b, c in permutations((i, j, k)):
            total += ordered_triple[a, b, c]
        if total > 0.0:
            triple_dict[frozenset({i, j, k})] = total

    # ── place: P(i top-3) = P(i 1着) + P(i 2着) + P(i 3着) ─────────────────
    # P(i 2着) = Σ_a ordered_pair[a, i]   (column sum)
    # P(i 3着) = Σ_{a, b} ordered_triple[a, b, i]
    p_2nd = ordered_pair.sum(axis=0)
    p_3rd = ordered_triple.sum(axis=(0, 1))
    place = p + p_2nd + p_3rd

    return {
        "place": place,
        "pair": pair,
        "ordered_pair": ordered_pair,
        "triple": triple_dict,
        "ordered_triple": ordered_triple,
    }


# ---------------------------------------------------------------------------
# Isotonic calibration for win-probability outputs
# ---------------------------------------------------------------------------


class IsotonicCalibrator:
    """Post-hoc isotonic regression calibrator for binary-classifier win probs.

    Wraps sklearn's IsotonicRegression and adds within-race re-normalisation
    so calibrated probabilities sum to 1 per race (preserves the rank-1
    softmax constraint).

    Usage:
        cal = IsotonicCalibrator()
        cal.fit(raw_valid_probs, valid_is_winner)         # 1-D arrays
        race_probs = cal.predict(race_raw_probs)          # one race at a time
                                                          # (re-normalised)
        bulk_probs = cal.predict(bulk_raw_probs)          # if normalise=False

    Persisted via pickle alongside the NN model dir.
    """

    def __init__(self) -> None:
        self.iso = IsotonicRegression(
            out_of_bounds="clip",
            y_min=0.0,
            y_max=1.0,
            increasing=True,
        )
        self.fitted = False

    def fit(self, raw_probs: np.ndarray, outcomes: np.ndarray) -> None:
        """Fit on (raw_prob, outcome) pairs.

        Args:
            raw_probs: 1-D array of raw probabilities from the binary classifier.
            outcomes: 1-D array of 0/1 (is_winner) outcomes, same length.
        """
        raw = np.asarray(raw_probs, dtype=np.float64).ravel()
        out = np.asarray(outcomes, dtype=np.float64).ravel()
        if raw.shape != out.shape:
            raise ValueError(
                f"raw_probs shape {raw.shape} != outcomes shape {out.shape}"
            )
        self.iso.fit(raw, out)
        self.fitted = True

    def predict(
        self, raw_probs: np.ndarray, normalise: bool = True
    ) -> np.ndarray:
        """Apply calibration. With normalise=True (default), the result is
        re-scaled so it sums to 1 — only meaningful when raw_probs is one
        race's worth of horses.

        Args:
            raw_probs: 1-D array of raw probabilities from the binary classifier.
            normalise: If True, divide by sum so result is a proper distribution
                over the race. Set False for bulk diagnostic / metric use.

        Returns:
            Calibrated probabilities (same shape as input).
        """
        if not self.fitted:
            raise RuntimeError("IsotonicCalibrator must be fit() before predict()")
        raw = np.asarray(raw_probs, dtype=np.float64).ravel()
        calibrated = self.iso.predict(raw)
        if normalise:
            total = calibrated.sum()
            if total > 0:
                calibrated = calibrated / total
        return calibrated


# ---------------------------------------------------------------------------
# Conditional isotonic calibration — surface × n_runners bin 別
# ---------------------------------------------------------------------------


def _n_runners_bin(n: int) -> int:
    """Map n_runners to a discrete bin index (0..3).

    Bins: 0 = <=8, 1 = 9-12, 2 = 13-15, 3 = 16+.
    """
    if n <= 8:
        return 0
    if n <= 12:
        return 1
    if n <= 15:
        return 2
    return 3


class ConditionalIsotonicCalibrator:
    """Post-hoc isotonic calibrator stratified by (surface, n_runners_bin).

    Fits one IsotonicRegression per (surface × n_runners_bin) stratum where
    sample count meets min_samples_per_bin.  Strata with too few samples fall
    back to a global IsotonicRegression fit on all data.

    n_runners_bin mapping: <=8 → 0, 9-12 → 1, 13-15 → 2, 16+ → 3.

    Usage:
        cal = ConditionalIsotonicCalibrator()
        cal.fit(raw_valid_probs, valid_is_winner, conditions_df)
        calibrated = cal.predict(raw_probs, conditions_df)

    conditions_df must have columns ['surface', 'n_runners'], one row per
    element of raw_probs / target.
    """

    def __init__(self, min_samples_per_bin: int = 100) -> None:
        self._calibrators: dict[tuple, IsotonicRegression] = {}
        self._global: IsotonicRegression = IsotonicRegression(
            out_of_bounds="clip", y_min=0.0, y_max=1.0, increasing=True
        )
        self._global_fitted: bool = False
        self._min_samples_per_bin = min_samples_per_bin

    def fit(
        self,
        raw: np.ndarray,
        target: np.ndarray,
        conditions: pd.DataFrame,
    ) -> None:
        """Fit global fallback and per-stratum calibrators.

        Args:
            raw: 1-D array of raw probabilities.
            target: 1-D array of 0/1 outcomes (e.g. is_winner).
            conditions: DataFrame with columns ['surface', 'n_runners'],
                same length as raw/target.
        """
        raw_arr = np.asarray(raw, dtype=np.float64).ravel()
        tgt_arr = np.asarray(target, dtype=np.float64).ravel()

        if raw_arr.shape != tgt_arr.shape:
            raise ValueError(
                f"raw shape {raw_arr.shape} != target shape {tgt_arr.shape}"
            )
        if len(raw_arr) != len(conditions):
            raise ValueError(
                f"raw length {len(raw_arr)} != conditions length {len(conditions)}"
            )

        # Global fallback — fit on all data.
        self._global.fit(raw_arr, tgt_arr)
        self._global_fitted = True

        # Per-stratum fits.
        bins = conditions["n_runners"].apply(_n_runners_bin).values
        surfaces = conditions["surface"].values

        # Group by (surface, bin).
        # Build a structured array of (surface_str, bin_int) keys.
        unique_keys = set(zip(surfaces.tolist(), bins.tolist(), strict=True))
        for surf, b in unique_keys:
            mask = (surfaces == surf) & (bins == b)
            count = int(mask.sum())
            if count < self._min_samples_per_bin:
                continue
            iso = IsotonicRegression(
                out_of_bounds="clip", y_min=0.0, y_max=1.0, increasing=True
            )
            iso.fit(raw_arr[mask], tgt_arr[mask])
            self._calibrators[(surf, b)] = iso

    def predict(
        self,
        raw: np.ndarray,
        conditions: pd.DataFrame,
        normalise: bool = False,
    ) -> np.ndarray:
        """Apply per-stratum calibration with global fallback.

        Args:
            raw: 1-D array of raw probabilities.
            conditions: DataFrame with columns ['surface', 'n_runners'],
                same length as raw.
            normalise: If True, divide result by its sum so it forms a proper
                distribution (only meaningful when raw covers one full race).

        Returns:
            Calibrated probabilities (same shape as raw).
        """
        if not self._global_fitted:
            raise RuntimeError(
                "ConditionalIsotonicCalibrator must be fit() before predict()"
            )
        raw_arr = np.asarray(raw, dtype=np.float64).ravel()
        result = np.empty_like(raw_arr)

        bins = conditions["n_runners"].apply(_n_runners_bin).values
        surfaces = conditions["surface"].values

        for i, (surf, b) in enumerate(zip(surfaces.tolist(), bins.tolist(), strict=True)):
            iso = self._calibrators.get((surf, b), self._global)
            result[i] = float(iso.predict(raw_arr[i : i + 1])[0])

        if normalise:
            total = result.sum()
            if total > 0:
                result = result / total

        return result


# ---------------------------------------------------------------------------
# 連系 馬券 (馬連 / ワイド / 馬単 / 三連複 / 三連単) の確率 calibration
# ---------------------------------------------------------------------------


class ComboCalibrators:
    """連系 馬券種ごとに isotonic 補正を持つコンテナ。

    Plackett-Luce MC で導出した combo prob は系統的に低 prob 帯で過大評価、
    高 prob 帯で過小評価される (combo_calibration_diagnosis 参照)。
    馬券種ごとに (PL_prob, hit) で iso fit すれば、個別 calibrator の
    transform で歪みを吸収できる。

    `predict` は **正規化を行わない** 点が IsotonicCalibrator と異なる
    (連系 combo は race 内合計が 1 にならないので): 単純にスカラー値の
    monotone 補正のみ。

    use_conditional=True のとき内部の isotonic を ConditionalIsotonicCalibrator
    に切り替える。各馬券種ごとに conditions (surface × n_runners) 別の補正を
    学習・適用できる。
    """

    def __init__(self, use_conditional: bool = False) -> None:
        # When use_conditional is True, values are ConditionalIsotonicCalibrator;
        # otherwise, they are plain IsotonicRegression (backward-compat).
        self._calibrators: dict[str, IsotonicRegression | ConditionalIsotonicCalibrator] = {}
        self._use_conditional = use_conditional

    def fit_for(
        self,
        bet_type: str,
        raw_probs: np.ndarray,
        outcomes: np.ndarray,
        conditions: pd.DataFrame | None = None,
    ) -> None:
        """1 馬券種分の calibrator を学習。

        Args:
            bet_type: Bet type label (e.g. '馬連').
            raw_probs: 1-D array of raw PL probabilities.
            outcomes: 1-D array of 0/1 hit outcomes.
            conditions: DataFrame with ['surface', 'n_runners'] required when
                use_conditional=True.  Ignored when use_conditional=False.
        """
        raw = np.asarray(raw_probs, dtype=np.float64).ravel()
        out = np.asarray(outcomes, dtype=np.float64).ravel()
        if raw.shape != out.shape:
            raise ValueError(
                f"raw_probs shape {raw.shape} != outcomes shape {out.shape}"
            )
        if raw.size == 0:
            return

        if self._use_conditional:
            if conditions is None:
                raise ValueError(
                    "conditions DataFrame required when use_conditional=True"
                )
            cal: IsotonicRegression | ConditionalIsotonicCalibrator = (
                ConditionalIsotonicCalibrator()
            )
            cal.fit(raw, out, conditions)
        else:
            iso = IsotonicRegression(
                out_of_bounds="clip",
                y_min=0.0,
                y_max=1.0,
                increasing=True,
            )
            iso.fit(raw, out)
            cal = iso
        self._calibrators[bet_type] = cal

    def predict(
        self,
        bet_type: str,
        raw_probs: np.ndarray,
        conditions: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """馬券種に対応した calibrator で transform。fit していない bet_type
        は raw を返す (後方互換)。

        Args:
            bet_type: Bet type label.
            raw_probs: 1-D array of raw probabilities.
            conditions: Required when the stored calibrator is a
                ConditionalIsotonicCalibrator (i.e. use_conditional=True was
                used at fit time).
        """
        raw = np.asarray(raw_probs, dtype=np.float64).ravel()
        cal = self._calibrators.get(bet_type)
        if cal is None:
            return raw
        if isinstance(cal, ConditionalIsotonicCalibrator):
            if conditions is None:
                raise ValueError(
                    "conditions DataFrame required for ConditionalIsotonicCalibrator predict"
                )
            return cal.predict(raw, conditions)
        return cal.predict(raw)

    def has(self, bet_type: str) -> bool:
        return bet_type in self._calibrators

    @property
    def use_conditional(self) -> bool:
        return self._use_conditional

    @property
    def fitted_bet_types(self) -> list[str]:
        return list(self._calibrators.keys())


def _build_conditions_for_race(race_frame: pd.DataFrame) -> pd.DataFrame:
    """Build a per-entry conditions DataFrame from race_frame.

    Extracts 'surface' (entry-level column already in the feature frame) and
    'n_runners' (computed from race size and broadcast to all entries).
    Returns a DataFrame with columns ['surface', 'n_runners'] aligned to
    race_frame's index.
    """
    n = len(race_frame)
    surface = race_frame["surface"].values if "surface" in race_frame.columns else ["unknown"] * n
    n_runners_val = int(race_frame["n_runners"].iloc[0]) if "n_runners" in race_frame.columns else n
    return pd.DataFrame(
        {"surface": surface, "n_runners": n_runners_val},
        index=race_frame.index,
    )


def fit_combo_calibrators_bundle(
    valid_frame,
    bundle,  # ModelBundle
    n_samples: int = 5_000,
    rng: np.random.Generator | None = None,
    use_conditional: bool = False,
) -> ComboCalibrators:
    """Bundle-aware variant: collect (PL_prob, hit) pairs per bet type and fit iso.

    Mirrors :func:`fit_combo_calibrators` but routes combo prediction through
    :func:`ai.predict.predict_race_with_combinations` so it works for both
    GBDT and NN bundles.  Used by ``ai.nn.train_nn`` after temperature
    scaling so that the persisted ``combo_calibrators.pkl`` corrects the
    raw PL joint probabilities that drive ワイド / 三連複 EV decisions.

    Args:
        valid_frame: Validation feature frame (race_id, post_position,
            finish_position, etc.) — same shape used elsewhere.
        bundle: A loaded ModelBundle (GBDT or NN).
        n_samples: MC samples for the per-race combo prediction.
        rng: Optional random generator for reproducibility.
        use_conditional: Pass-through to ComboCalibrators
            (surface × n_runners-conditional iso when True).

    Returns:
        A fitted ComboCalibrators (only bet types with ≥100 samples are fitted).
    """
    from ai.predict import predict_race_with_combinations  # noqa: PLC0415

    bet_types = list(RENKEI_BET_TYPES)
    records: dict[str, list[tuple[float, int]]] = {bt: [] for bt in bet_types}
    cond_records: dict[str, list[tuple[str, int]]] = {bt: [] for bt in bet_types}

    for _race_id, race_frame in valid_frame.groupby("race_id"):
        if len(race_frame) < 4:
            continue
        if race_frame["post_position"].isna().any():
            continue
        finished = race_frame.dropna(subset=["finish_position"])
        finished = finished[finished["finish_position"].astype(int).isin([1, 2, 3])]
        if len(finished) < 3:
            continue
        by_finish = {
            int(row["finish_position"]): int(row["post_position"])
            for _, row in finished.iterrows()
        }
        pp1, pp2, pp3 = by_finish.get(1), by_finish.get(2), by_finish.get(3)
        if pp1 is None or pp2 is None or pp3 is None:
            continue

        try:
            combo_map = predict_race_with_combinations(
                bundle, race_frame, n_samples=n_samples, rng=rng,
            )
        except Exception:  # noqa: BLE001
            continue

        surf = str(race_frame["surface"].iloc[0]) if "surface" in race_frame.columns else "unknown"
        n_runners_val = (
            int(race_frame["n_runners"].iloc[0])
            if "n_runners" in race_frame.columns
            else len(race_frame)
        )

        for bt in bet_types:
            for cp in combo_map.get(bt, []):
                hit = _is_combo_hit(bt, cp.combo, pp1, pp2, pp3)
                records[bt].append((float(cp.prob), 1 if hit else 0))
                cond_records[bt].append((surf, n_runners_val))

    cal = ComboCalibrators(use_conditional=use_conditional)
    for bt, recs in records.items():
        if len(recs) < 100:
            continue
        raw = np.asarray([r[0] for r in recs], dtype=np.float64)
        out = np.asarray([r[1] for r in recs], dtype=np.float64)
        if use_conditional:
            conds = cond_records[bt]
            cond_df = pd.DataFrame(conds, columns=["surface", "n_runners"])
            cal.fit_for(bt, raw, out, conditions=cond_df)
        else:
            cal.fit_for(bt, raw, out)
    return cal


def _is_combo_hit(bet_type: str, combo: str, pp1: int, pp2: int, pp3: int) -> bool:
    """連系 combo が実 top-3 と一致するか。combo_calibration_diagnosis と同じロジック。"""
    try:
        if bet_type == "馬連":
            pps = sorted(int(x) for x in combo.split("-"))
            return pps == sorted([pp1, pp2])
        if bet_type == "ワイド":
            pps = {int(x) for x in combo.split("-")}
            return pps.issubset({pp1, pp2, pp3})
        if bet_type == "馬単":
            parts = [int(x) for x in combo.split("→")]
            return parts == [pp1, pp2]
        if bet_type == "三連複":
            pps = sorted(int(x) for x in combo.split("-"))
            return pps == sorted([pp1, pp2, pp3])
        if bet_type == "三連単":
            parts = [int(x) for x in combo.split("→")]
            return parts == [pp1, pp2, pp3]
    except (ValueError, TypeError):
        return False
    return False
