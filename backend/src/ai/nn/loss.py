"""Ranking loss functions for horse-race prediction.

All functions operate on batched tensors:
    scores           [B, N]  — model logits / scores per horse
    finish_positions [B, N]  — ground-truth finishing position (1-based, NaN = unknown)
    mask             [B, N]  — bool, True = valid horse, False = padded slot

Losses are reduced to a scalar mean over *valid* races / pairs.
"""

from __future__ import annotations

from itertools import permutations

import torch
import torch.nn.functional as F


def _valid_race_mask(
    finish_positions: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return [B] bool: True when race has >= 2 valid (non-NaN, mask=True) horses."""
    valid = mask & ~torch.isnan(finish_positions)
    return valid.sum(dim=-1) >= 2


def plackett_luce_loss(
    scores: torch.Tensor,
    finish_positions: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Plackett-Luce log-likelihood loss.

    Minimising this is equivalent to maximising the probability of observing
    the ground-truth permutation under a Plackett-Luce model where choice
    probabilities are proportional to exp(score).

    Args:
        scores:           [B, N]
        finish_positions: [B, N]  NaN = exclude
        mask:             [B, N]  bool

    Returns:
        Scalar loss (mean over valid races).
    """
    B, N = scores.shape
    device = scores.device
    total_loss = torch.zeros(1, device=device)
    n_valid = 0

    for b in range(B):
        valid = mask[b] & ~torch.isnan(finish_positions[b])
        if valid.sum() < 2:
            continue

        s = scores[b][valid]          # [K]
        pos = finish_positions[b][valid]  # [K]

        # Sort ascending by finish position (winner first)
        order = torch.argsort(pos)
        s_sorted = s[order]  # [K]

        # log P(permutation) = sum_k [ s_k - log sum_{j>=k} exp(s_j) ]
        # Use logsumexp over remaining horses at each stage
        K = s_sorted.size(0)
        log_prob = torch.zeros(1, device=device)
        for k in range(K - 1):  # last stage has no choice
            log_prob = log_prob + s_sorted[k] - torch.logsumexp(s_sorted[k:], dim=0)

        total_loss = total_loss - log_prob
        n_valid += 1

    if n_valid == 0:
        return torch.tensor(float("nan"), device=device)
    return (total_loss / n_valid).squeeze()


def listmle_loss(
    scores: torch.Tensor,
    finish_positions: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """ListMLE loss (reference implementation; numerically identical to PL here).

    ListMLE maximises log P(ground-truth permutation) under a plackett-luce
    model but is presented separately as a named baseline for ablations.

    Args:
        scores:           [B, N]
        finish_positions: [B, N]  NaN = exclude
        mask:             [B, N]  bool

    Returns:
        Scalar loss (mean over valid races).
    """
    B, N = scores.shape
    device = scores.device
    total_loss = torch.zeros(1, device=device)
    n_valid = 0

    for b in range(B):
        valid = mask[b] & ~torch.isnan(finish_positions[b])
        if valid.sum() < 2:
            continue

        s = scores[b][valid]
        pos = finish_positions[b][valid]

        order = torch.argsort(pos)
        s_sorted = s[order]

        K = s_sorted.size(0)
        log_prob = torch.zeros(1, device=device)
        for k in range(K - 1):
            log_prob = log_prob + s_sorted[k] - torch.logsumexp(s_sorted[k:], dim=0)

        total_loss = total_loss - log_prob
        n_valid += 1

    if n_valid == 0:
        return torch.tensor(float("nan"), device=device)
    return (total_loss / n_valid).squeeze()


def log_growth_loss(
    scores: torch.Tensor,
    finish_positions: torch.Tensor,
    odds_win: torch.Tensor,
    mask: torch.Tensor,
    kelly_fraction: float = 0.25,
) -> torch.Tensor:
    """Fractional-Kelly log-growth (decision-focused) loss for 単勝 betting.

    Each race is treated as a 単勝 portfolio.  A softmax over the model scores
    gives per-horse allocation weights ``p_i``; we stake a fraction
    ``kelly_fraction`` of bankroll spread by ``p_i`` and keep the rest in cash.
    The realised wealth multiple of the race is::

        W = 1 + kelly_fraction * (p_winner * odds_winner - 1)

    and the loss is ``-mean(log W)`` over races.  Maximising this maximises
    expected log-growth of bankroll (the Kelly objective) using **real odds**,
    so the model is rewarded for concentrating mass where ``p * odds > 1`` and
    penalised when it does not — i.e. it optimises betting return directly
    rather than ranking accuracy.

    The cash term (``kelly_fraction < 1``) is what keeps ``odds_winner`` in the
    gradient; with full-Kelly (kf=1, no cash) the odds factor degenerates to a
    constant and the objective collapses to plain winner cross-entropy.

    Args:
        scores:           [B, N]
        finish_positions: [B, N]  NaN = exclude; winner is position == 1
        odds_win:         [B, N]  **raw** 単勝 odds (NaN = unknown)
        mask:             [B, N]  bool
        kelly_fraction:   fraction of bankroll staked per race, in (0, 1).

    Returns:
        Scalar loss (mean over races with a known, odds-carrying winner).
    """
    device = scores.device
    kf = float(kelly_fraction)
    total_loss = torch.zeros(1, device=device)
    n_valid = 0

    for b in range(scores.size(0)):
        valid = mask[b] & ~torch.isnan(finish_positions[b])
        if valid.sum() < 2:
            continue

        s = scores[b][valid]                 # [K]
        pos = finish_positions[b][valid]     # [K]
        o = odds_win[b][valid]               # [K]

        # Winner = finishing position 1.  Skip races with no clean winner or
        # whose winner has no recorded odds (can't price the payoff).
        winner_idx = (pos == 1).nonzero(as_tuple=True)[0]
        if winner_idx.numel() == 0:
            continue
        w = winner_idx[0]
        o_w = o[w]
        if torch.isnan(o_w) or o_w <= 0:
            continue

        p = torch.softmax(s, dim=0)          # [K] allocation weights
        p_w = p[w]

        wealth = 1.0 + kf * (p_w * o_w - 1.0)  # > 1 - kf > 0 for kf < 1
        total_loss = total_loss - torch.log(wealth)
        n_valid += 1

    if n_valid == 0:
        return torch.tensor(float("nan"), device=device)
    return (total_loss / n_valid).squeeze()


def log_growth_place_loss(
    scores: torch.Tensor,
    place_returns: torch.Tensor,
    mask: torch.Tensor,
    kelly_fraction: float = 0.25,
) -> torch.Tensor:
    """Fractional-Kelly log-growth loss for 複勝 (place) betting.

    Mirrors :func:`log_growth_loss` but pays off on **placing** (in-the-money)
    rather than winning.  ``place_returns[i]`` is the realised payoff multiple of
    a 複勝 bet on horse i (= payout_yen / 100 if it placed, else 0), supplied
    pre-computed so the loss stays a pure tensor op.

    Each race stakes ``kelly_fraction`` of bankroll spread by softmax(scores);
    realised wealth = ``1 + kf * (Σ_i p_i * place_returns_i - 1)`` and the loss
    is ``-mean(log wealth)``.  This rewards putting mass on horses that place and
    pay well — directly optimising 複勝 return, the lower-variance market.

    Args:
        scores:        [B, N]
        place_returns: [B, N]  payoff multiple if placed, 0 otherwise (NaN→0)
        mask:          [B, N]  bool
        kelly_fraction: bankroll fraction staked per race, in (0, 1).

    Returns:
        Scalar loss (mean over races with >= 2 valid horses).
    """
    device = scores.device
    kf = float(kelly_fraction)
    total_loss = torch.zeros(1, device=device)
    n_valid = 0

    for b in range(scores.size(0)):
        valid = mask[b]
        if valid.sum() < 2:
            continue

        s = scores[b][valid]
        pr = torch.nan_to_num(place_returns[b][valid], nan=0.0)
        if pr.sum() <= 0:
            continue  # no recorded place payoff in this race → skip

        p = torch.softmax(s, dim=0)
        expected = (p * pr).sum()
        wealth = 1.0 + kf * (expected - 1.0)  # > 1 - kf > 0 for kf < 1
        total_loss = total_loss - torch.log(wealth)
        n_valid += 1

    if n_valid == 0:
        return torch.tensor(float("nan"), device=device)
    return (total_loss / n_valid).squeeze()


_COMBO_BET_TYPES = frozenset(["馬連", "馬単", "三連複", "三連単"])


def _pl_exacta(s: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """Analytic Plackett-Luce probability of the ordered pair i→j (馬単).

    P(i 1st, j 2nd) = softmax(s)_i * softmax(s without i)_j.  Differentiable.
    """
    p = torch.softmax(s, dim=0)
    keep = torch.ones_like(s, dtype=torch.bool)
    keep[i] = False
    p2 = torch.softmax(s[keep], dim=0)[j - 1 if j > i else j]
    return p[i] * p2


def _pl_trifecta(s: torch.Tensor, i: int, j: int, k: int) -> torch.Tensor:
    """Analytic Plackett-Luce probability of the ordered triple i→j→k (三連単)."""
    p = torch.softmax(s, dim=0)
    keep1 = torch.ones_like(s, dtype=torch.bool)
    keep1[i] = False
    p2 = torch.softmax(s[keep1], dim=0)[j - 1 if j > i else j]
    keep2 = keep1.clone()
    keep2[j] = False
    rem = [x for x in range(len(s)) if x not in (i, j)]
    p3 = torch.softmax(s[keep2], dim=0)[rem.index(k)]
    return p[i] * p2 * p3


def _winning_combo_prob(
    s: torch.Tensor, i: int, j: int, k: int | None, bet_type: str
) -> torch.Tensor:
    """Analytic PL probability of the realised winning combo, differentiable.

    i / j / k are the within-race indices of the 1st / 2nd / 3rd finishers.
    """
    if bet_type == "馬単":
        return _pl_exacta(s, i, j)
    if bet_type == "馬連":
        return _pl_exacta(s, i, j) + _pl_exacta(s, j, i)
    if bet_type == "三連単":
        return _pl_trifecta(s, i, j, k)  # type: ignore[arg-type]
    # 三連複: sum over all 6 orderings of the unordered triple
    total = s.new_zeros(())
    for a, b, c in permutations((i, j, k)):
        total = total + _pl_trifecta(s, a, b, c)
    return total


def log_growth_combo_loss(
    scores: torch.Tensor,
    finish_positions: torch.Tensor,
    combo_payoff: torch.Tensor,
    mask: torch.Tensor,
    bet_type: str = "馬連",
    kelly_fraction: float = 0.25,
) -> torch.Tensor:
    """Fractional-Kelly log-growth loss for 連系 (馬連/馬単/三連複/三連単).

    Decision-focused analogue of :func:`log_growth_loss` for combination bets.
    The probability of the **realised winning combo** is computed *analytically*
    from the Plackett-Luce model (differentiable in the scores — no Monte-Carlo
    sampling), and the objective maximises log-growth of betting it at the real
    payoff::

        W = 1 + kelly_fraction * (P_PL(winning_combo) * payoff - 1)

    Because the loss penalises over-stating ``P_PL`` (it lowers W when the combo
    misses), the combo *calibration is learned inside the model* — this replaces
    the external post-hoc isotonic ``combo_calibrators``.

    Args:
        scores:           [B, N]
        finish_positions: [B, N]  1-based finishing position (NaN = exclude).
            The winning combo is the 1st/2nd(/3rd) finishers.
        combo_payoff:     [B, N]  race-broadcast payoff multiple of the winning
            combo for this bet type (= payout_yen / 100; 0/NaN = no payout).
        mask:             [B, N]  bool.
        bet_type:         one of 馬連 / 馬単 / 三連複 / 三連単.
        kelly_fraction:   bankroll fraction staked per race, in (0, 1).

    Returns:
        Scalar loss (mean over races with a clean winning combo + payoff).
    """
    if bet_type not in _COMBO_BET_TYPES:
        raise ValueError(
            f"bet_type {bet_type!r} not in {sorted(_COMBO_BET_TYPES)}"
        )
    triple = bet_type in ("三連複", "三連単")
    device = scores.device
    kf = float(kelly_fraction)
    total_loss = torch.zeros(1, device=device)
    n_valid = 0

    for b in range(scores.size(0)):
        valid = mask[b] & ~torch.isnan(finish_positions[b])
        if valid.sum() < (3 if triple else 2):
            continue

        s = scores[b][valid]
        pos = finish_positions[b][valid]
        pay = torch.nan_to_num(combo_payoff[b][valid], nan=0.0)[0]
        if pay <= 0:
            continue

        w1 = (pos == 1).nonzero(as_tuple=True)[0]
        w2 = (pos == 2).nonzero(as_tuple=True)[0]
        if w1.numel() == 0 or w2.numel() == 0:
            continue
        i, j = int(w1[0]), int(w2[0])

        k: int | None = None
        if triple:
            w3 = (pos == 3).nonzero(as_tuple=True)[0]
            if w3.numel() == 0:
                continue
            k = int(w3[0])

        prob = _winning_combo_prob(s, i, j, k, bet_type)
        wealth = 1.0 + kf * (prob * pay - 1.0)  # > 1 - kf > 0 for kf < 1
        total_loss = total_loss - torch.log(wealth)
        n_valid += 1

    if n_valid == 0:
        return torch.tensor(float("nan"), device=device)
    return (total_loss / n_valid).squeeze()


def combo_nll_loss(
    scores: torch.Tensor,
    finish_positions: torch.Tensor,
    mask: torch.Tensor,
    bet_type: str = "馬連",
) -> torch.Tensor:
    """Negative log-likelihood of the realised winning 連系 combo (calibration).

    A *proper scoring rule*: minimising ``-log P_PL(winning_combo)`` drives the
    analytic Plackett-Luce combo probabilities toward their true frequencies, so
    the **combo calibration is learned inside the NN** — this is the direct
    replacement for the external post-hoc isotonic ``combo_calibrators``.  Unlike
    :func:`log_growth_combo_loss` (which optimises betting *return* and therefore
    suppresses probabilities on the −EV combo markets), this targets calibration.

    No odds / payoff needed.

    Args:
        scores:           [B, N]
        finish_positions: [B, N]  1-based finish (NaN = exclude).
        mask:             [B, N]  bool.
        bet_type:         a single 連系 type, or "all" to sum the NLL over
            馬連 + 馬単 + 三連複 + 三連単 (one model calibrated on every combo).

    Returns:
        Scalar loss (mean over races with a clean winning combo).
    """
    types = sorted(_COMBO_BET_TYPES) if bet_type == "all" else [bet_type]
    for bt in types:
        if bt not in _COMBO_BET_TYPES:
            raise ValueError(f"bet_type {bt!r} not in {sorted(_COMBO_BET_TYPES)} or 'all'")
    needs_triple = any(bt in ("三連複", "三連単") for bt in types)
    device = scores.device
    eps = 1e-12
    total_loss = torch.zeros(1, device=device)
    n_valid = 0

    for b in range(scores.size(0)):
        valid = mask[b] & ~torch.isnan(finish_positions[b])
        if valid.sum() < (3 if needs_triple else 2):
            continue
        s = scores[b][valid]
        pos = finish_positions[b][valid]

        w1 = (pos == 1).nonzero(as_tuple=True)[0]
        w2 = (pos == 2).nonzero(as_tuple=True)[0]
        if w1.numel() == 0 or w2.numel() == 0:
            continue
        i, j = int(w1[0]), int(w2[0])

        k: int | None = None
        if needs_triple:
            w3 = (pos == 3).nonzero(as_tuple=True)[0]
            if w3.numel() == 0:
                continue
            k = int(w3[0])

        race_nll = s.new_zeros(())
        for bt in types:
            prob = _winning_combo_prob(s, i, j, k, bt)
            race_nll = race_nll - torch.log(prob.clamp_min(eps))
        total_loss = total_loss + race_nll
        n_valid += 1

    if n_valid == 0:
        return torch.tensor(float("nan"), device=device)
    return (total_loss / n_valid).squeeze()


def multi_objective_loss(
    scores: torch.Tensor,
    finish_positions: torch.Tensor,
    odds_win: torch.Tensor,
    mask: torch.Tensor,
    combo_weight: float = 0.01,
    kelly_fraction: float = 0.25,
    combo_bet_type: str = "馬連",
) -> torch.Tensor:
    """Production all-markets objective: 単複 betting + 連系 calibration.

    Weighted sum of:
      - :func:`log_growth_loss` — optimises 単勝 betting return (drives 単勝/複勝
        ROI; this is the active model's objective), and
      - ``combo_weight`` × :func:`combo_nll_loss` (``combo_bet_type``) — calibrates
        the 連系 combo probabilities **inside the NN** (replaces the external
        isotonic combo_calibrators).

    The two share the same scores, so this trades a little 単複 ROI for honest,
    self-calibrated 連系 probabilities in a single deployable model.  The
    ``combo_weight`` default is small because the combo NLL is ~10× the magnitude
    of the log-growth term; tune via --combo-weight.

    ``combo_bet_type`` defaults to 馬連 (pairs) for speed: calibrating the pair
    marginals tightens the shared scores and carries over to the triples.
    ``"all"`` calibrates every combo type but the triple-ordering sums make
    full-dataset training ~5-10× slower (the analytic combo prob runs in a
    per-race Python loop — vectorising it is a future optimisation).

    Args:
        scores / finish_positions / mask: [B, N].
        odds_win:     [B, N] raw 単勝 odds (for the log_growth term).
        combo_weight: weight on the combo-calibration NLL term.
        kelly_fraction: for the log_growth term.
        combo_bet_type: 連系 type (or "all") for the calibration term.

    Returns:
        Scalar loss.  NaN only when *both* terms are NaN for the batch.
    """
    lg = log_growth_loss(scores, finish_positions, odds_win, mask, kelly_fraction)
    cn = combo_nll_loss(scores, finish_positions, mask, bet_type=combo_bet_type)

    terms = []
    if not torch.isnan(lg):
        terms.append(lg)
    if not torch.isnan(cn):
        terms.append(combo_weight * cn)
    if not terms:
        return torch.tensor(float("nan"), device=scores.device)
    return torch.stack(terms).sum()


def time_margin_loss(
    scores: torch.Tensor,
    finish_positions: torch.Tensor,
    finish_times: torch.Tensor,
    mask: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Hinge loss weighted by finishing-time margin.

    For every ordered pair (i, j) within a race where finish_position[i] < finish_position[j]
    (i beat j), the loss penalises when the model score does not respect the ordering:

        loss_ij = max(0, margin_ij - (score_i - score_j))

    where margin_ij = scale * (finish_time[j] - finish_time[i]).
    NaN finish_times fall back to margin = 0.

    Args:
        scores:           [B, N]
        finish_positions: [B, N]  NaN = exclude
        finish_times:     [B, N]  NaN = use margin 0
        mask:             [B, N]  bool
        scale:            multiplier applied to time differences

    Returns:
        Scalar loss (mean over valid pairs across all races in batch).
    """
    device = scores.device
    total_loss = torch.zeros(1, device=device)
    n_pairs = 0

    for b in range(scores.size(0)):
        valid = mask[b] & ~torch.isnan(finish_positions[b])
        idx = valid.nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue

        s = scores[b][idx]           # [K]
        pos = finish_positions[b][idx]  # [K]
        t = finish_times[b][idx]     # [K]

        K = s.size(0)
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                # i finished before j (lower position number = better)
                if pos[i] >= pos[j]:
                    continue

                # Time margin: t[j] - t[i] >= 0 when j is slower
                if torch.isnan(t[i]) or torch.isnan(t[j]):
                    margin = torch.zeros(1, device=device)
                else:
                    margin = (t[j] - t[i]) * scale

                hinge = F.relu(margin - (s[i] - s[j]))
                total_loss = total_loss + hinge
                n_pairs += 1

    if n_pairs == 0:
        return torch.tensor(float("nan"), device=device)
    return (total_loss / n_pairs).squeeze()
