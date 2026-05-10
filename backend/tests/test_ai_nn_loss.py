"""Tests for keiba_ai.ai.nn.loss (plackett_luce_loss, listmle_loss, time_margin_loss)."""

from __future__ import annotations

import torch
import pytest

from keiba_ai.ai.nn.loss import listmle_loss, plackett_luce_loss, time_margin_loss


def _make_mask(n: int, total: int) -> torch.Tensor:
    """1D mask: first n True."""
    m = torch.zeros(total, dtype=torch.bool)
    m[:n] = True
    return m


# ---------------------------------------------------------------------------
# plackett_luce_loss
# ---------------------------------------------------------------------------


class TestPlackettLuceLoss:
    def test_perfect_scores_beat_reversed(self):
        """Model assigning high score to winner should have lower loss."""
        positions = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        # Perfect: scores match inverse of position (winner highest)
        good_scores = torch.tensor([[3.0, 2.0, 1.0]], requires_grad=True)
        bad_scores = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)

        good_loss = plackett_luce_loss(good_scores, positions, mask)
        bad_loss = plackett_luce_loss(bad_scores, positions, mask)

        assert good_loss.item() < bad_loss.item()

    def test_gradient_not_none(self):
        positions = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        scores = torch.tensor([[3.0, 2.0, 1.0]], requires_grad=True)
        loss = plackett_luce_loss(scores, positions, mask)
        loss.backward()
        assert scores.grad is not None

    def test_all_mask_false_returns_nan(self):
        """A race with all horses masked should return NaN."""
        scores = torch.tensor([[1.0, 2.0, 3.0]])
        positions = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.zeros(1, 3, dtype=torch.bool)
        loss = plackett_luce_loss(scores, positions, mask)
        assert torch.isnan(loss)

    def test_nan_positions_excluded(self):
        """NaN finish_positions are treated as missing and excluded."""
        positions = torch.tensor([[1.0, float("nan"), 3.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        scores = torch.tensor([[3.0, 2.0, 1.0]], requires_grad=True)
        # Should not raise and should produce a finite scalar
        loss = plackett_luce_loss(scores, positions, mask)
        assert torch.isfinite(loss)

    def test_batch_of_two(self):
        """Works over a batch and returns scalar."""
        positions = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        mask = torch.ones(2, 3, dtype=torch.bool)
        scores = torch.tensor([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]], requires_grad=True)
        loss = plackett_luce_loss(scores, positions, mask)
        assert loss.shape == ()


# ---------------------------------------------------------------------------
# listmle_loss
# ---------------------------------------------------------------------------


class TestListMLELoss:
    def test_perfect_scores_beat_reversed(self):
        positions = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)

        good_scores = torch.tensor([[3.0, 2.0, 1.0]], requires_grad=True)
        bad_scores = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)

        good_loss = listmle_loss(good_scores, positions, mask)
        bad_loss = listmle_loss(bad_scores, positions, mask)

        assert good_loss.item() < bad_loss.item()

    def test_gradient_not_none(self):
        positions = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        scores = torch.tensor([[3.0, 2.0, 1.0]], requires_grad=True)
        loss = listmle_loss(scores, positions, mask)
        loss.backward()
        assert scores.grad is not None

    def test_all_mask_false_returns_nan(self):
        scores = torch.tensor([[1.0, 2.0, 3.0]])
        positions = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.zeros(1, 3, dtype=torch.bool)
        loss = listmle_loss(scores, positions, mask)
        assert torch.isnan(loss)


# ---------------------------------------------------------------------------
# time_margin_loss
# ---------------------------------------------------------------------------


class TestTimeMarginLoss:
    def _perfect_batch(self):
        """3 horses with positions [1,2,3] and times [10,11,12]. Scores respect order."""
        positions = torch.tensor([[1.0, 2.0, 3.0]])
        times = torch.tensor([[10.0, 11.0, 12.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        # score_1 > score_2 > score_3, margin = delta_time * scale
        # With scale=1 margins are [1, 2, 1]; scores diff = [2, 4, 2] >= all margins
        scores = torch.tensor([[4.0, 2.0, 0.0]])
        return scores, positions, times, mask

    def test_zero_loss_when_all_pairs_correct(self):
        """No valid pair violated → loss should be 0."""
        scores, positions, times, mask = self._perfect_batch()
        # Increase margin to 0 by using scale=0
        loss = time_margin_loss(scores, positions, times, mask, scale=0.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_gradient_backward_works(self):
        scores = torch.tensor([[4.0, 2.0, 0.0]], requires_grad=True)
        positions = torch.tensor([[1.0, 2.0, 3.0]])
        times = torch.tensor([[10.0, 11.0, 12.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        loss = time_margin_loss(scores, positions, times, mask, scale=0.0)
        loss.backward()
        assert scores.grad is not None

    def test_scale_affects_loss(self):
        """Larger scale → larger margin → higher loss when pairs are borderline correct."""
        positions = torch.tensor([[1.0, 2.0]])
        times = torch.tensor([[10.0, 12.0]])  # delta = 2
        mask = torch.ones(1, 2, dtype=torch.bool)
        # score diff = 1 (winner - loser); margin at scale=0.5 → 1.0 (tie) → loss 0
        # margin at scale=2.0 → 4.0 → loss = 4 - 1 = 3
        scores_small_gap = torch.tensor([[1.5, 0.5]], requires_grad=True)

        loss_small_scale = time_margin_loss(
            scores_small_gap, positions, times, mask, scale=0.5
        )
        scores_small_gap2 = torch.tensor([[1.5, 0.5]])
        loss_large_scale = time_margin_loss(
            scores_small_gap2, positions, times, mask, scale=2.0
        )
        assert loss_large_scale.item() > loss_small_scale.item()

    def test_nan_times_treated_as_zero_margin(self):
        """NaN finish_times → margin = 0, hinge reduces to max(0, -(s_i - s_j))."""
        positions = torch.tensor([[1.0, 2.0]])
        times = torch.tensor([[float("nan"), float("nan")]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        # Winner has higher score → hinge 0
        scores = torch.tensor([[2.0, 1.0]])
        loss = time_margin_loss(scores, positions, times, mask, scale=1.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_all_mask_false_returns_nan(self):
        scores = torch.tensor([[2.0, 1.0]])
        positions = torch.tensor([[1.0, 2.0]])
        times = torch.tensor([[10.0, 11.0]])
        mask = torch.zeros(1, 2, dtype=torch.bool)
        loss = time_margin_loss(scores, positions, times, mask)
        assert torch.isnan(loss)
