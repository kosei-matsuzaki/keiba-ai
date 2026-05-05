"""Probability calibration helpers.

softmax_within_race: standard approach used in M4.
top_k_cumulative_prob: simple approximation for place probability.
plackett_luce_*: Plackett-Luce Monte Carlo probability estimators.
compute_all_combination_probs: compute all combination probabilities in one pass.
"""

from __future__ import annotations

import numpy as np


def softmax_within_race(scores: np.ndarray) -> np.ndarray:
    """Convert raw LightGBM lambdarank scores to win probabilities via softmax.

    Subtracts max for numerical stability before exponentiation.
    """
    shifted = scores - scores.max()
    exp_s = np.exp(shifted)
    return exp_s / exp_s.sum()


def top_k_cumulative_prob(scores: np.ndarray, k: int = 3) -> np.ndarray:
    """Approximate each horse's probability of finishing in the top-k.

    Simple approach: for each horse i, place_prob[i] = sum of win_prob for
    the top-k horses when horse i is included.  In practice we use the
    complementary view: place_prob[i] ≈ sum_{j in top-k} win_prob[j] if
    i is in top-k, else sum_{j in top-k+1, j!=i} win_prob[j].

    This is a heuristic approximation — see plackett_luce_place_prob for
    a theoretically correct (but expensive) alternative.
    """
    win_probs = softmax_within_race(scores)
    n = len(win_probs)
    effective_k = min(k, n)

    order = np.argsort(-win_probs)  # descending
    place_probs = np.zeros(n)

    for i in range(n):
        if i in order[:effective_k]:
            # Horse is in top-k by win_prob: cumulative prob of the top-k group
            place_probs[i] = win_probs[order[:effective_k]].sum()
        else:
            # Approximate: sum of top-(k-1) excluding this horse
            top_without_i = [j for j in order if j != i][:effective_k]
            place_probs[i] = win_probs[top_without_i].sum()

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
