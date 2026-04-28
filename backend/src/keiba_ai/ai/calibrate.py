"""Probability calibration helpers.

softmax_within_race: standard approach used in M4.
top_k_cumulative_prob: simple approximation for place probability.
plackett_luce_place_prob: stub for future Plackett-Luce Monte Carlo.
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


def plackett_luce_place_prob(scores: np.ndarray, k: int = 3, n_samples: int = 10_000) -> np.ndarray:
    """Estimate top-k placement probabilities via Plackett-Luce Monte Carlo.

    This is a future extension — not used in M4.
    """
    raise NotImplementedError(
        "Plackett-Luce Monte Carlo is planned for a future milestone. "
        "Use top_k_cumulative_prob for M4."
    )
