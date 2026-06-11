"""Tests for ai.model.loss (plackett_luce, log_growth, combo_nll, multi)."""

from __future__ import annotations

import math

import numpy as np
import torch

from ai.model.loss import (
    _pl_exacta,
    _winning_combo_prob,
    combo_nll_loss,
    log_growth_loss,
    multi_objective_loss,
    plackett_luce_loss,
)

# ---------------------------------------------------------------------------
# plackett_luce_loss (two-stage pretrain objective)
# ---------------------------------------------------------------------------


class TestPlackettLuceLoss:
    def test_perfect_scores_beat_reversed(self):
        positions = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        good = torch.tensor([[3.0, 2.0, 1.0]], requires_grad=True)
        bad = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        assert plackett_luce_loss(good, positions, mask).item() < plackett_luce_loss(
            bad, positions, mask
        ).item()

    def test_gradient_not_none(self):
        positions = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        scores = torch.tensor([[3.0, 2.0, 1.0]], requires_grad=True)
        plackett_luce_loss(scores, positions, mask).backward()
        assert scores.grad is not None

    def test_all_mask_false_returns_nan(self):
        scores = torch.tensor([[1.0, 2.0, 3.0]])
        positions = torch.tensor([[1.0, 2.0, 3.0]])
        mask = torch.zeros(1, 3, dtype=torch.bool)
        assert torch.isnan(plackett_luce_loss(scores, positions, mask))


# ---------------------------------------------------------------------------
# log_growth_loss (単勝 betting return)
# ---------------------------------------------------------------------------


def test_log_growth_matches_manual_and_differentiable():
    s = torch.tensor([[2.0, 1.0, 0.5, 0.0]], requires_grad=True)
    pos = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    odds = torch.tensor([[4.0, 6.0, 10.0, 20.0]])
    mask = torch.tensor([[True, True, True, True]])
    loss = log_growth_loss(s, pos, odds, mask, kelly_fraction=0.25)
    p0 = math.exp(2) / (math.exp(2) + math.exp(1) + math.exp(0.5) + 1)
    expected = -math.log(1 + 0.25 * (p0 * 4.0 - 1))
    assert abs(loss.item() - expected) < 1e-5
    loss.backward()
    assert s.grad.norm().item() > 0


def test_log_growth_nan_winner_odds_skips():
    loss = log_growth_loss(
        torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 2.0]]),
        torch.tensor([[float("nan"), 2.0]]), torch.tensor([[True, True]]),
    )
    assert math.isnan(loss.item())


# ---------------------------------------------------------------------------
# combo_nll_loss (連系 calibration) + analytic-PL combo prob
# ---------------------------------------------------------------------------


def test_combo_nll_all_types_and_all():
    pos = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[True, True, True, True]])
    for bt in ("馬連", "馬単", "三連複", "三連単", "all"):
        s = torch.tensor([[2.0, 1.0, 0.5, 0.0]], requires_grad=True)
        nll = combo_nll_loss(s, pos, mask, bet_type=bt)
        assert torch.isfinite(nll) and nll.item() > 0
        nll.backward()
        assert s.grad.norm().item() > 0
    s = torch.tensor([2.0, 1.0, 0.5, 0.0])
    P = _winning_combo_prob(s, 0, 1, None, "馬連")
    got = combo_nll_loss(s.unsqueeze(0), pos, mask, bet_type="馬連")
    assert abs(got.item() - (-math.log(P.item()))) < 1e-5


def test_combo_nll_skips_no_winner():
    loss = combo_nll_loss(
        torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 1.0]]),
        torch.tensor([[True, True]]), bet_type="馬連",
    )
    assert math.isnan(loss.item())


def test_pl_combo_prob_matches_monte_carlo():
    torch.manual_seed(0)
    s = torch.randn(8)
    a = _pl_exacta(s, 0, 1) + _pl_exacta(s, 1, 0)  # 馬連 {0,1}
    rng = np.random.default_rng(1)
    g = rng.gumbel(size=(150_000, 8))
    order = np.argsort(-(s.numpy()[None, :] + g), axis=1)[:, :2]
    mc = np.mean([frozenset(o) == frozenset((0, 1)) for o in order])
    assert abs(a.item() - mc) < 0.01


# ---------------------------------------------------------------------------
# multi_objective_loss (production all-markets: log_growth + combo_nll)
# ---------------------------------------------------------------------------


def test_multi_objective_is_weighted_sum():
    s = torch.tensor([[2.0, 1.0, 0.5, 0.0]], requires_grad=True)
    pos = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    odds = torch.tensor([[4.0, 6.0, 10.0, 20.0]])
    mask = torch.tensor([[True, True, True, True]])
    lg = log_growth_loss(s, pos, odds, mask)
    cn = combo_nll_loss(s, pos, mask, bet_type="馬連")
    m = multi_objective_loss(s, pos, odds, mask, combo_weight=0.01, combo_bet_type="馬連")
    assert abs(m.item() - (lg.item() + 0.01 * cn.item())) < 1e-5
    m.backward()
    assert s.grad.norm().item() > 0
