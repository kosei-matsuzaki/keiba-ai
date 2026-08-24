"""Tests for ai.model.loss (plackett_luce, log_growth, flat_ev, combo_nll, multi)."""

from __future__ import annotations

import math

import numpy as np
import torch

from ai.model.loss import (
    _pl_exacta,
    _winning_combo_prob,
    combo_nll_loss,
    flat_ev_loss,
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
    loss = log_growth_loss(s, pos, odds, mask, cash_fraction=0.25)
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


# ---------------------------------------------------------------------------
# kelly_deploy_loss (L1: deployment-matched selective Kelly portfolio)
# ---------------------------------------------------------------------------


def test_kelly_deploy_matches_manual_and_differentiable():
    """W = (1 - sum f_i) + f_winner * o_winner with f_i = kf*relu(p_i*o_i-1)/(o_i-1)."""
    from ai.model.loss import kelly_deploy_loss

    s = torch.tensor([[2.0, 1.0, 0.5, 0.0]], requires_grad=True)
    pos = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    odds = torch.tensor([[4.0, 6.0, 10.0, 20.0]])
    mask = torch.tensor([[True, True, True, True]])
    kf = 0.25
    loss = kelly_deploy_loss(s, pos, odds, mask, kelly_fraction=kf)

    import math as _m
    es = [_m.exp(x) for x in (2.0, 1.0, 0.5, 0.0)]
    z = sum(es)
    p = [e / z for e in es]
    o = [4.0, 6.0, 10.0, 20.0]
    f = [kf * max(p[i] * o[i] - 1.0, 0.0) / (o[i] - 1.0) for i in range(4)]
    total = sum(f)
    scale = min(0.95 / total, 1.0) if total > 0 else 1.0
    f = [x * scale for x in f]
    W = (1.0 - sum(f)) + f[0] * o[0]  # winner = idx 0
    expected = -_m.log(W)
    assert abs(loss.item() - expected) < 1e-5
    loss.backward()
    assert s.grad.norm().item() > 0


def test_kelly_deploy_abstains_on_negative_edge():
    """A horse whose p*o <= 1 gets zero stake (relu abstention)."""
    from ai.model.loss import kelly_deploy_loss

    # Two horses, winner has odds so low that p*o < 1 → no +EV bet → W=1, loss=0.
    s = torch.tensor([[0.0, 0.0]])  # p = 0.5 each
    pos = torch.tensor([[1.0, 2.0]])
    odds = torch.tensor([[1.5, 1.5]])  # p*o = 0.75 < 1 for both → abstain
    mask = torch.tensor([[True, True]])
    loss = kelly_deploy_loss(s, pos, odds, mask)
    assert abs(loss.item()) < 1e-6  # log(1) = 0


def test_kelly_deploy_positive_edge_winner_gives_gain():
    """A clearly +EV winner yields W>1 → negative loss (gain)."""
    from ai.model.loss import kelly_deploy_loss

    s = torch.tensor([[3.0, 0.0, 0.0]])  # winner heavily favoured by model
    pos = torch.tensor([[1.0, 2.0, 3.0]])
    odds = torch.tensor([[2.0, 5.0, 5.0]])  # winner p*o >> 1
    mask = torch.tensor([[True, True, True]])
    loss = kelly_deploy_loss(s, pos, odds, mask)
    assert loss.item() < 0  # W > 1


def test_kelly_deploy_nan_winner_odds_skips():
    from ai.model.loss import kelly_deploy_loss

    loss = kelly_deploy_loss(
        torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0, 2.0]]),
        torch.tensor([[float("nan"), 2.0]]), torch.tensor([[True, True]]),
    )
    assert math.isnan(loss.item())


# ---------------------------------------------------------------------------
# flat_ev_loss (deployment-matched flat-stake objective)
# ---------------------------------------------------------------------------


class TestFlatEvLoss:
    """定額配分 (assign_flat_stakes) と同じ決定を微分可能化した損失。"""

    def test_backing_the_winner_at_long_odds_beats_backing_the_loser(self):
        # 勝ち馬 = index 0 (odds 5.0)。0 に確率を寄せた方が損失が小さい。
        positions = torch.tensor([[1.0, 2.0, 3.0]])
        odds = torch.tensor([[5.0, 3.0, 8.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        on_winner = torch.tensor([[4.0, 0.0, 0.0]])
        on_loser = torch.tensor([[0.0, 4.0, 0.0]])
        assert (
            flat_ev_loss(on_winner, positions, odds, mask).item()
            < flat_ev_loss(on_loser, positions, odds, mask).item()
        )

    def test_no_bet_is_the_zero_loss_floor(self):
        """全馬の EV が閾値未満なら gate が閉じ、損益 0 = 損失 0 に漸近する。

        sigmoid gate は厳密に 0 にはならないので完全な 0 ではなく「ほぼ 0」。
        「買わない」が損失 0 の床であること (= -EV 相場では棄権が最適) を示す。
        """
        positions = torch.tensor([[1.0, 2.0, 3.0]])
        # 一様スコア → p=1/3、odds 1.2 → EV=0.4 で閾値 1.1 を大きく下回る
        odds = torch.tensor([[1.2, 1.2, 1.2]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        scores = torch.zeros(1, 3)
        assert abs(flat_ev_loss(scores, positions, odds, mask).item()) < 1e-4

    def test_gradient_flows(self):
        positions = torch.tensor([[1.0, 2.0, 3.0]])
        odds = torch.tensor([[5.0, 3.0, 8.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        scores = torch.tensor([[1.0, 0.5, 0.2]], requires_grad=True)
        flat_ev_loss(scores, positions, odds, mask).backward()
        assert scores.grad is not None
        assert torch.isfinite(scores.grad).all()

    def test_unknown_odds_are_never_staked(self):
        """odds NaN の馬は gate 0 — 損失に寄与しない。"""
        positions = torch.tensor([[2.0, 1.0, 3.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        scores = torch.tensor([[4.0, 0.0, 0.0]])
        with_nan = torch.tensor([[float("nan"), 3.0, 8.0]])
        # NaN 馬に全確率を寄せても賭けられないので損益 0
        assert abs(flat_ev_loss(scores, positions, with_nan, mask).item()) < 1e-6

    def test_max_bets_caps_total_stake(self):
        """max_bets を絞ると賭ける総量が減り、負けレースの損失も縮む。"""
        # 勝ち馬なし側に賭けが集中するケース: 全馬 +EV だが勝つのは 1 頭
        positions = torch.tensor([[3.0, 2.0, 1.0]])
        odds = torch.tensor([[20.0, 20.0, 20.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        scores = torch.zeros(1, 3)
        uncapped = flat_ev_loss(scores, positions, odds, mask, max_bets=0.0).item()
        capped = flat_ev_loss(scores, positions, odds, mask, max_bets=1.0).item()
        assert capped != uncapped
        assert abs(capped) < abs(uncapped)

    def test_returns_nan_when_no_race_has_a_winner(self):
        positions = torch.tensor([[2.0, 3.0, 4.0]])
        odds = torch.tensor([[5.0, 3.0, 8.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        scores = torch.zeros(1, 3)
        assert math.isnan(flat_ev_loss(scores, positions, odds, mask).item())


def test_log_growth_collapses_to_cross_entropy_without_cash():
    """cash_fraction=1 だと odds が勾配から消え、勝ち馬 CE と一致する。

    この性質が「cash_fraction は賭け金の Kelly ではなく、odds を勾配に残すための
    装置」であることの根拠。うっかり 1.0 にすると ROI 志向損失ではなくなる。
    """
    torch.manual_seed(0)
    scores = torch.randn(3, 6, requires_grad=True)
    pos = torch.zeros(3, 6)
    for b in range(3):
        pos[b, 0] = 1.0
        pos[b, 1:] = torch.arange(2, 7, dtype=torch.float)
    odds = torch.rand(3, 6) * 20 + 1.5
    mask = torch.ones(3, 6, dtype=torch.bool)

    def grad_of(fn):
        if scores.grad is not None:
            scores.grad = None
        fn().backward()
        return scores.grad.clone()

    g_no_cash = grad_of(lambda: log_growth_loss(scores, pos, odds, mask, cash_fraction=1.0))
    g_ce = grad_of(
        lambda: -sum(torch.log_softmax(scores[b], dim=0)[0] for b in range(3)) / 3
    )
    assert torch.allclose(g_no_cash, g_ce, atol=1e-6)

    # 本番設定 (0.25) では odds が効くので CE とは一致しない
    g_prod = grad_of(lambda: log_growth_loss(scores, pos, odds, mask, cash_fraction=0.25))
    assert not torch.allclose(g_prod, g_ce, atol=1e-6)
