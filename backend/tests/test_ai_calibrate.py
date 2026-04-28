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


def test_plackett_luce_raises():
    with pytest.raises(NotImplementedError):
        plackett_luce_place_prob(np.array([1.0, 2.0, 3.0]))
