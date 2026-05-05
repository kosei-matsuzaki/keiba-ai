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


def plackett_luce_place_prob(scores: np.ndarray, k: int = 3, n_samples: int = 10_000) -> np.ndarray:
    """Estimate top-k placement probabilities via Plackett-Luce Monte Carlo.

    This is a future extension — not used in M4.
    """
    raise NotImplementedError(
        "Plackett-Luce Monte Carlo is planned for a future milestone. "
        "Use top_k_cumulative_prob for M4."
    )
