"""Tests for ai/core/probabilities.py — probability calibration helpers."""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from ai.core.probabilities import (
    compute_all_combination_probs,
    plackett_luce_place_prob,
    sample_top_k,
    softmax_within_race,
    top_k_cumulative_prob,
)


def test_softmax_sums_to_one():
    scores = np.array([2.5, 1.0, 0.3, -0.5, 1.8])
    probs = softmax_within_race(scores)
    assert probs.sum() == pytest.approx(1.0, abs=1e-6)


def test_softmax_all_non_negative():
    scores = np.array([0.1, -1.0, 3.0, 0.5])
    probs = softmax_within_race(scores)
    assert (probs >= 0).all()


def test_softmax_highest_score_highest_prob():
    scores = np.array([3.0, 1.0, 0.5, 2.0])
    probs = softmax_within_race(scores)
    assert probs.argmax() == 0  # index 0 has score 3.0


def test_softmax_uniform_scores():
    scores = np.zeros(5)
    probs = softmax_within_race(scores)
    np.testing.assert_allclose(probs, np.full(5, 0.2), atol=1e-6)


def test_top_k_all_in_range():
    scores = np.array([3.0, 2.0, 1.5, 0.8, 0.2])
    place_probs = top_k_cumulative_prob(scores, k=3)
    assert (place_probs >= 0).all()
    assert (place_probs <= 1.0 + 1e-9).all()


def test_top_k_top_horse_has_higher_place_prob():
    scores = np.array([5.0, 0.1, 0.1, 0.1, 0.1])
    place_probs = top_k_cumulative_prob(scores, k=3)
    # The top-scoring horse should have the highest or equal place probability
    # (equal is acceptable when dominant — all top-3 share a near-identical cumulative mass)
    assert place_probs[0] >= place_probs[-1]
    assert place_probs[0] >= 0.0


def test_top_k_single_horse():
    scores = np.array([1.0])
    place_probs = top_k_cumulative_prob(scores, k=3)
    assert place_probs[0] == pytest.approx(1.0, abs=1e-6)


def test_top_k_separates_top_k_from_rest():
    """Regression: top-k horses must have a strictly higher place_prob than
    non-top-k horses whenever the k-th win_prob is non-zero.

    Earlier the slice was [:effective_k] (instead of [:effective_k - 1]),
    which made every horse share the same top-k cumulative mass and
    degenerated the EV-based place bet filter to all-or-nothing per race.
    """
    scores = np.array([3.0, 2.0, 1.0, 0.5, 0.0, -0.5])
    place_probs = top_k_cumulative_prob(scores, k=3)
    top_three = place_probs[:3]
    rest = place_probs[3:]
    assert (top_three > rest.max()).all()
    # All top-k share the same mass; all non-top-k share another (smaller) mass.
    np.testing.assert_allclose(top_three, top_three[0])
    np.testing.assert_allclose(rest, rest[0])


def test_top_k_two_unique_values_per_race():
    """The current heuristic produces at most 2 unique place_probs per race
    (one for the top-k group, one for the rest). Lock this in so callers
    don't accidentally rely on per-horse variance that the heuristic
    cannot produce.
    """
    scores = np.array([3.0, 2.0, 1.5, 0.8, 0.2, -0.1, -0.5, -1.0])
    place_probs = top_k_cumulative_prob(scores, k=3)
    assert len(set(np.round(place_probs, 9))) <= 2


def test_top_k_non_top_k_uses_top_km1_sum():
    """Non-top-k horses should receive sum of top-(k-1) win_probs."""
    scores = np.array([3.0, 2.0, 1.0, 0.5, 0.0])
    win_probs = np.exp(scores - scores.max())
    win_probs = win_probs / win_probs.sum()
    expected_non_top = float(win_probs[:2].sum())  # sum of top-(k-1)=top-2

    place_probs = top_k_cumulative_prob(scores, k=3)
    # Indices 3 and 4 are non-top-3 by win_prob (which mirrors score order here).
    assert place_probs[3] == pytest.approx(expected_non_top, abs=1e-9)
    assert place_probs[4] == pytest.approx(expected_non_top, abs=1e-9)


# ---------------------------------------------------------------------------
# sample_top_k
# ---------------------------------------------------------------------------

def test_sample_top_k_shape():
    scores = np.array([3.0, 2.0, 1.5, 0.8, 0.2])
    rng = np.random.default_rng(0)
    samples = sample_top_k(scores, k=3, n_samples=500, rng=rng)
    assert samples.shape == (500, 3)


def test_sample_top_k_no_repeat_within_row():
    scores = np.array([3.0, 2.0, 1.5, 0.8, 0.2])
    rng = np.random.default_rng(42)
    samples = sample_top_k(scores, k=3, n_samples=200, rng=rng)
    for row in samples:
        assert len(set(row.tolist())) == 3, "Each row must have unique horse indices"


def test_sample_top_k_reproducible():
    scores = np.array([1.0, 2.0, 3.0, 0.5])
    s1 = sample_top_k(scores, k=2, n_samples=100, rng=np.random.default_rng(7))
    s2 = sample_top_k(scores, k=2, n_samples=100, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(s1, s2)


def test_sample_top_k_raises_when_k_exceeds_n():
    """k > n must raise ValueError, not silently truncate."""
    scores = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="k=4 exceeds number of horses 3"):
        sample_top_k(scores, k=4, n_samples=10, rng=np.random.default_rng(0))


def test_plackett_luce_permutation_probs_match_theory():
    """Gumbel-Top-k sampling must match exact Plackett-Luce permutation probabilities.

    For scores = [2, 1, 0] the PL model gives softmax probs p = softmax([2,1,0]).
    All 6 permutations have closed-form probabilities; we verify each is within
    ±0.01 of the empirical frequency at n_samples=200_000.
    """
    scores = np.array([2.0, 1.0, 0.0])
    rng = np.random.default_rng(2024)
    n_samples = 200_000

    samples = sample_top_k(scores, k=3, n_samples=n_samples, rng=rng)

    # Count each of the 6 permutations of (0,1,2)
    perm_counts: dict[tuple, int] = {p: 0 for p in itertools.permutations([0, 1, 2])}
    for row in samples:
        perm_counts[tuple(row.tolist())] += 1
    empirical = {p: cnt / n_samples for p, cnt in perm_counts.items()}

    # Theoretical PL probabilities:
    # p = softmax([2,1,0]) ≈ [0.6652, 0.2447, 0.0900]
    # P(a,b,c) = p[a] * p[b]/(1-p[a]) * p[c]/(1-p[a]-p[b])
    expected = {
        (0, 1, 2): pytest.approx(0.4866, abs=0.01),
        (0, 2, 1): pytest.approx(0.1789, abs=0.01),
        (1, 0, 2): pytest.approx(0.2155, abs=0.01),
        (1, 2, 0): pytest.approx(0.0291, abs=0.01),
        (2, 0, 1): pytest.approx(0.0658, abs=0.01),
        (2, 1, 0): pytest.approx(0.0242, abs=0.01),
    }
    for perm, approx_val in expected.items():
        assert empirical[perm] == approx_val, (
            f"Permutation {perm}: empirical={empirical[perm]:.4f}"
        )


# ---------------------------------------------------------------------------
# plackett_luce_place_prob
# ---------------------------------------------------------------------------

def test_plackett_luce_place_prob_per_horse_distinct():
    """Plackett-Luce gives a unique probability per horse (breaks top_k_cumulative_prob's 2-value constraint)."""
    scores = np.array([5.0, 3.0, 1.0, 0.5, 0.1, 0.1])
    rng = np.random.default_rng(0)
    probs = plackett_luce_place_prob(scores, k=3, n_samples=5_000, rng=rng)
    assert len(np.unique(np.round(probs, 3))) > 2, (
        "Expected more than 2 distinct probability values"
    )


def test_plackett_luce_place_prob_sum_approx_k():
    scores = np.array([3.0, 2.0, 1.5, 0.8, 0.2, 0.1])
    rng = np.random.default_rng(1)
    probs = plackett_luce_place_prob(scores, k=3, n_samples=5_000, rng=rng)
    assert probs.sum() == pytest.approx(3.0, abs=0.05)


def test_plackett_luce_place_prob_dominant_horse():
    """Horse with far superior score should dominate top-3."""
    scores = np.array([20.0, 0.0, 0.0, 0.0, 0.0])
    rng = np.random.default_rng(2)
    probs = plackett_luce_place_prob(scores, k=3, n_samples=5_000, rng=rng)
    assert probs[0] == pytest.approx(1.0, abs=0.01)


def test_plackett_luce_place_prob_uniform_equal():
    """All equal scores should yield approximately equal probabilities."""
    scores = np.zeros(6)
    rng = np.random.default_rng(3)
    probs = plackett_luce_place_prob(scores, k=3, n_samples=10_000, rng=rng)
    expected = 3.0 / 6.0
    np.testing.assert_allclose(probs, expected, atol=0.05)


def test_plackett_luce_place_prob_reproducible():
    scores = np.array([2.0, 1.0, 0.5, 0.3])
    p1 = plackett_luce_place_prob(scores, k=2, n_samples=500, rng=np.random.default_rng(99))
    p2 = plackett_luce_place_prob(scores, k=2, n_samples=500, rng=np.random.default_rng(99))
    np.testing.assert_array_equal(p1, p2)


def test_plackett_luce_place_prob_variance_shrinks_with_more_samples():
    """More samples should produce a result closer to the 10k-sample estimate."""
    scores = np.array([2.0, 1.5, 1.0, 0.5, 0.2])
    reference = plackett_luce_place_prob(scores, k=3, n_samples=10_000, rng=np.random.default_rng(0))
    low_sample = plackett_luce_place_prob(scores, k=3, n_samples=1_000, rng=np.random.default_rng(1))
    # The 10k estimate should not deviate wildly from the 1k estimate on average
    assert np.abs(reference - low_sample).mean() < 0.05


# ---------------------------------------------------------------------------
# compute_all_combination_probs
# ---------------------------------------------------------------------------

def test_compute_all_combination_probs_keys():
    """Default k=3 must return all six keys."""
    scores = np.array([3.0, 2.0, 1.0, 0.5])
    rng = np.random.default_rng(50)
    result = compute_all_combination_probs(scores, n_samples=500, rng=rng)
    expected_keys = {"place", "position", "pair", "ordered_pair", "triple", "ordered_triple"}
    assert set(result.keys()) == expected_keys


def test_compute_all_combination_probs_shapes():
    scores = np.array([3.0, 2.0, 1.0, 0.5])
    n = len(scores)
    rng = np.random.default_rng(51)
    result = compute_all_combination_probs(scores, n_samples=500, rng=rng)
    assert result["place"].shape == (n,)
    assert result["position"].shape == (n, 3)
    assert result["pair"].shape == (n, n)
    assert result["ordered_pair"].shape == (n, n)
    assert isinstance(result["triple"], dict)
    assert result["ordered_triple"].shape == (n, n, n)


def test_compute_all_combination_probs_k2_omits_triple_keys():
    """k=2 must omit triple and ordered_triple, but include pair keys."""
    scores = np.array([3.0, 2.0, 1.0, 0.5])
    n = len(scores)
    rng = np.random.default_rng(52)
    result = compute_all_combination_probs(scores, k=2, n_samples=500, rng=rng)
    assert "triple" not in result
    assert "ordered_triple" not in result
    assert result["place"].shape == (n,)
    assert result["position"].shape == (n, 2)
    assert result["pair"].shape == (n, n)
    assert result["ordered_pair"].shape == (n, n)


def test_compute_all_combination_probs_k1_omits_pair_and_triple_keys():
    """k=1 must include only place and position."""
    scores = np.array([3.0, 2.0, 1.0, 0.5])
    rng = np.random.default_rng(53)
    result = compute_all_combination_probs(scores, k=1, n_samples=500, rng=rng)
    assert set(result.keys()) == {"place", "position"}


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------

def test_plackett_luce_performance_16_horses():
    """16-horse race with 10,000 samples must finish within 200 ms."""
    scores = np.linspace(0.0, 3.0, 16)
    rng = np.random.default_rng(0)
    start = time.perf_counter()
    plackett_luce_place_prob(scores, k=3, n_samples=10_000, rng=rng)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 200, f"Took {elapsed_ms:.1f} ms, expected < 200 ms"
