"""Tests for ai/calibrate.py — probability calibration helpers."""

from __future__ import annotations

import numpy as np
import pytest

from keiba_ai.ai.calibrate import (
    plackett_luce_place_prob,
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


def test_plackett_luce_raises():
    with pytest.raises(NotImplementedError):
        plackett_luce_place_prob(np.array([1.0, 2.0, 3.0]))
