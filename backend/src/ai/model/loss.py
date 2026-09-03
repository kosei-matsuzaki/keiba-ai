"""Loss functions for the horse-race NN.

All functions operate on batched tensors:
    scores           [B, N]  — model logits / scores per horse
    finish_positions [B, N]  — ground-truth finishing position (1-based, NaN = unknown)
    mask             [B, N]  — bool, True = valid horse, False = padded slot

Active production objective is `multi_objective_loss` (単複 betting via
log_growth + 連系 calibration via combo_nll); `plackett_luce_loss` is the
two-stage pretrain.  Losses are reduced to a scalar mean over *valid* races.
"""

from __future__ import annotations

from itertools import permutations

import torch


def plackett_luce_loss(
    scores: torch.Tensor,
    finish_positions: torch.Tensor,
    mask: torch.Tensor,
    top_k: int | None = None,
) -> torch.Tensor:
    """Plackett-Luce log-likelihood loss (optionally truncated to the top-k).

    Minimising this is equivalent to maximising the probability of observing
    the ground-truth permutation under a Plackett-Luce model where choice
    probabilities are proportional to exp(score).

    **``top_k`` の意味**: 既定 (None) は全順列を当てにいく。しかし払戻が発生するのは
    3 着までで、**6 着と 7 着のどちらが上かを当てても 1 円も生まない**。平均出走頭数
    14.0 のこのデータでは、上位 3 着だけが重要なら **段の 75.3% が payout と無関係**
    (上位 5 着基準でも 58.9%)。``top_k`` を指定すると先頭 k 段だけを尤度に含め、
    残りの順序を無視する (learning-to-rank の truncated ranking loss)。

    Args:
        scores:           [B, N]
        finish_positions: [B, N]  NaN = exclude
        mask:             [B, N]  bool
        top_k:            先頭何着までを尤度に含めるか。None で全段。

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
        n_stages = K - 1 if top_k is None else min(int(top_k), K - 1)
        for k in range(n_stages):  # last stage has no choice
            log_prob = log_prob + s_sorted[k] - torch.logsumexp(s_sorted[k:], dim=0)

        total_loss = total_loss - log_prob
        n_valid += 1

    if n_valid == 0:
        return torch.tensor(float("nan"), device=device)
    return (total_loss / n_valid).squeeze()


def _races_with_priced_winner(
    scores: torch.Tensor,
    finish_positions: torch.Tensor,
    odds_win: torch.Tensor,
    mask: torch.Tensor,
):
    """賭けリターン系の損失が使えるレースだけを (p, o, w) で順に返す。

    条件は 3 つとも「値が計算できない」で、モデルの良し悪しとは無関係:
      - 有効な馬が 2 頭未満 (softmax が意味を持たない)
      - 1 着が確定していない (finish_positions に 1 が無い)
      - 勝ち馬のオッズが不明 / 非正 (払戻を値付けできない)

    log_growth / kelly_deploy / flat_ev がこの 18 行を丸ごと写していたので
    1 つにした。ここを変えると 3 つの損失が同時に変わる = 同じ理由で変わる。

    Yields:
        p: [K] softmax(scores)、o: [K] 生オッズ、w: 勝ち馬の添字
    """
    for b in range(scores.size(0)):
        valid = mask[b] & ~torch.isnan(finish_positions[b])
        if valid.sum() < 2:
            continue

        s = scores[b][valid]                 # [K]
        pos = finish_positions[b][valid]     # [K]
        o = odds_win[b][valid]               # [K]

        winner_idx = (pos == 1).nonzero(as_tuple=True)[0]
        if winner_idx.numel() == 0:
            continue
        w = winner_idx[0]
        if torch.isnan(o[w]) or o[w] <= 0:
            continue

        yield torch.softmax(s, dim=0), o, w


def log_growth_loss(
    scores: torch.Tensor,
    finish_positions: torch.Tensor,
    odds_win: torch.Tensor,
    mask: torch.Tensor,
    cash_fraction: float = 0.25,
) -> torch.Tensor:
    """Log-growth (decision-focused) loss for 単勝 betting.

    Each race is treated as a 単勝 portfolio.  A softmax over the model scores
    gives per-horse allocation weights ``p_i``; a fraction ``cash_fraction`` of
    the notional is spread by ``p_i`` and the rest is held as cash.  The
    realised wealth multiple of the race is::

        W = 1 + cash_fraction * (p_winner * odds_winner - 1)

    and the loss is ``-mean(log W)`` over races.  Maximising this maximises
    expected log-growth of the staked notional using **real odds**,
    so the model is rewarded for concentrating mass where ``p * odds > 1`` and
    penalised when it does not — i.e. it optimises betting return directly
    rather than ranking accuracy.

    **``cash_fraction`` is a gradient device, not a staking policy.**  It used to
    be called ``kelly_fraction``, but betting Kelly (staking a fraction of
    bankroll) was removed from deployment — ``ai.betting.strategy`` stakes a flat
    amount per 買い目.  What this constant actually does is keep the cash term
    alive: with ``cash_fraction = 1`` (no cash) ``W`` degenerates to
    ``p_winner · odds_winner``, so ``log W = log p_winner + log odds_winner`` and
    the odds term is a constant — **the gradient becomes exactly plain winner
    cross-entropy and the loss stops optimising ROI at all** (verified
    numerically: gradients agree to 1e-8).  Values in (0, 1) are what make this a
    betting-return objective rather than a ranking one.

    Args:
        scores:           [B, N]
        finish_positions: [B, N]  NaN = exclude; winner is position == 1
        odds_win:         [B, N]  **raw** 単勝 odds (NaN = unknown)
        mask:             [B, N]  bool
        cash_fraction:    weight on the staked leg vs cash, in (0, 1).  Must be
                          < 1 or the objective collapses to cross-entropy.

    Returns:
        Scalar loss (mean over races with a known, odds-carrying winner).
    """
    device = scores.device
    kf = float(cash_fraction)
    total_loss = torch.zeros(1, device=device)
    n_valid = 0

    for p, o, w in _races_with_priced_winner(scores, finish_positions, odds_win, mask):
        o_w = o[w]
        p_w = p[w]

        wealth = 1.0 + kf * (p_w * o_w - 1.0)  # > 1 - kf > 0 for kf < 1
        total_loss = total_loss - torch.log(wealth)
        n_valid += 1

    if n_valid == 0:
        return torch.tensor(float("nan"), device=device)
    return (total_loss / n_valid).squeeze()


def place_growth_loss(
    scores: torch.Tensor,
    place_ret: torch.Tensor,
    mask: torch.Tensor,
    cash_fraction: float = 0.25,
    temp: float = 0.5,
) -> torch.Tensor:
    """Log-growth (decision-focused) loss for 複勝 betting.

    `log_growth_loss` の複勝版。**なぜ要るか**: 複勝はこれまで自前の目的関数を持たず、
    単勝向けに学習したスコアの Plackett-Luce 変換で選んでいた。レース内では
    ``place_prob`` は ``score`` の単調変換なので、**複勝で買う馬は単勝と必ず同じ 1 頭**に
    なる。「1 着は苦しいが 3 着内は堅い馬」を選ぶ手段が無い。

    実測でも、自前の目的を持つ単勝は市場ベースラインに対し +0.139 (0.931 vs 0.792)
    なのに、派生の複勝は +0.036 (0.886 vs 0.850) しかない。

    単勝との違いは**払戻が 3 頭に発生する**こと。1 着だけが的中の単勝と違い、
    3 着以内の馬すべてが払戻を持つので、``place_ret`` は「その馬に賭けたときの
    実現倍率」(3 着圏外は 0.0) を全馬について持つ。したがって実現富は

        W = 1 + cash_fraction * (Σ_i p_i * place_ret_i - 1)

    となり、単勝のように勝ち馬 1 頭を取り出す形にはならない。配分 ``p`` を
    払戻の出る馬に寄せるほど ``W`` が大きくなる。

    **``temp`` は実運用との整合のためにある。** 実運用は score 最大の **1 頭**に賭ける
    のに、温度 1.0 の softmax は配分を全馬に散らす (実測で 1 位に乗るのは 0.69)。
    複勝は 3 頭に払戻が出るので、散らしたままだと「掲示板に載りそうな馬に薄く広げる」
    のが有利になり、**argmax を当てにいく圧力がかからない**。実際、温度 1.0 で学習した
    最初の版は 3 着内率が上がり平均オッズが下がった (= 堅い方に広がった) だけで、
    本命 1 頭の回収率は active と差が出なかった。温度を下げると配分が 1 位に集中し
    ``Σ p_i r_i → r_argmax`` に近づく。既定 0.5 はレース内スコアの実測分布から選んだ
    (σ=2.31 / 1-2 位差 1.57 に対し 1 位への配分が 0.94。0.2 以下では 0.9995 で
    飽和して勾配が消える)。1.0 は最初の版の再現用。

    ``cash_fraction`` の役割は `log_growth_loss` と同じ (1.0 にすると odds 項が
    勾配から消える) なので、そちらの docstring を参照。

    Args:
        scores:    [B, N]
        place_ret: [B, N]  **生の**複勝実現倍率 (払戻/100、3 着圏外 0.0、不明は NaN)
        mask:      [B, N]  bool
        cash_fraction: 賭ける側の比重、(0, 1)。

    Returns:
        Scalar loss (払戻の分かるレースについての平均)。
    """
    device = scores.device
    kf = float(cash_fraction)
    total_loss = torch.zeros(1, device=device)
    n_valid = 0

    for b in range(scores.size(0)):
        valid = mask[b] & ~torch.isnan(place_ret[b])
        if valid.sum() < 2:
            continue
        r = place_ret[b][valid]
        # 払戻がまったく無いレース (payout_place を取れていない) は学習に使えない
        if not torch.any(r > 0):
            continue

        p = torch.softmax(scores[b][valid] / temp, dim=0)
        expected = (p * r).sum()
        wealth = 1.0 + kf * (expected - 1.0)   # > 1 - kf > 0 for kf < 1
        total_loss = total_loss - torch.log(wealth)
        n_valid += 1

    if n_valid == 0:
        return torch.tensor(float("nan"), device=device)
    return (total_loss / n_valid).squeeze()


def kelly_deploy_loss(
    scores: torch.Tensor,
    finish_positions: torch.Tensor,
    odds_win: torch.Tensor,
    mask: torch.Tensor,
    kelly_fraction: float = 0.25,
    max_total_stake: float = 0.95,
) -> torch.Tensor:
    """Deployment-matched fractional-Kelly 単勝 portfolio log-growth (L1).

    Unlike :func:`log_growth_loss` — which spreads softmax mass over *every*
    horse with a fixed cash term and lets only the winner's odds enter the
    gradient — this mirrors the **live betting decision** in
    ``ai.betting.strategy``: bet selectively, abstain on −EV, stake proportional
    to edge.  Training therefore optimises the decision that is actually
    deployed rather than a bet-on-everything proxy.

    Per race, with ``p_i = softmax(scores)_i`` (win prob) and raw odds ``o_i``::

        edge_i = p_i * o_i - 1                       # EV per unit on horse i
        f_i    = kelly_fraction * relu(edge_i)/(o_i-1)   # stake; 0 if -EV (abstain)
        (stakes scaled down so sum_i f_i <= max_total_stake)
        W      = (1 - sum_i f_i) + f_winner * o_winner   # realised wealth multiple

    and the loss is ``-mean(log W)``.  The ``relu`` is the abstention: −EV horses
    get zero stake and only push probability mass (via the shared softmax).
    Horses with unknown odds are never staked.  ``max_total_stake < 1`` keeps a
    cash floor so ``W`` stays positive (matches never risking the whole bankroll).

    Args:
        scores:           [B, N]
        finish_positions: [B, N]  NaN = exclude; winner is position == 1
        odds_win:         [B, N]  **raw** 単勝 odds (NaN = unknown → no stake)
        mask:             [B, N]  bool
        kelly_fraction:   Kelly multiplier on edge.  Genuinely Kelly here (unlike
                          :func:`log_growth_loss`'s cash term) — this loss models
                          edge-proportional staking, which deployment no longer
                          does (see :func:`flat_ev_loss` for the current rule).
        max_total_stake:  cap on per-race staked fraction (cash floor = 1 - cap).

    Returns:
        Scalar loss (mean over races with a clean, odds-carrying winner).
    """
    device = scores.device
    kf = float(kelly_fraction)
    cap = float(max_total_stake)
    total_loss = torch.zeros(1, device=device)
    n_valid = 0

    for p, o, w in _races_with_priced_winner(scores, finish_positions, odds_win, mask):
        o_w = o[w]

        # Per-horse edge & stake; unknown / non-positive odds → no stake.
        o_ok = torch.nan_to_num(o, nan=0.0)
        bettable = o_ok > 1.0
        edge = p * o_ok - 1.0
        stake = kf * torch.relu(edge) / torch.clamp(o_ok - 1.0, min=1e-6)
        stake = torch.where(bettable, stake, torch.zeros_like(stake))

        total_stake = stake.sum()
        # Scale down if over-leveraged so the cash floor (and W) stays positive.
        scale = torch.clamp(cap / torch.clamp(total_stake, min=1e-6), max=1.0)
        stake = stake * scale
        total_stake = stake.sum()

        wealth = (1.0 - total_stake) + stake[w] * o_w
        total_loss = total_loss - torch.log(wealth)
        n_valid += 1

    if n_valid == 0:
        return torch.tensor(float("nan"), device=device)
    return (total_loss / n_valid).squeeze()


def flat_ev_loss(
    scores: torch.Tensor,
    finish_positions: torch.Tensor,
    odds_win: torch.Tensor,
    mask: torch.Tensor,
    ev_threshold: float = 1.1,
    gate_temp: float = 0.05,
    max_bets: float = 0.0,
) -> torch.Tensor:
    """Flat-stake expected-profit loss — matches the *deployed* 単勝 betting rule.

    ``ai.betting.strategy.assign_flat_stakes`` bets a **fixed amount per 買い目**
    on every horse whose ``EV = p·o`` clears a threshold, highest EV first, up to
    a per-race budget.  Kelly (staking a *fraction of bankroll*) was removed from
    deployment, so :func:`log_growth_loss` / :func:`kelly_deploy_loss` — both of
    which optimise bankroll log-growth under edge-proportional staking — now
    optimise a decision the app never makes.  This loss closes that gap.

    Per race, with ``p_i = softmax(scores)_i`` and raw odds ``o_i``::

        ev_i     = p_i * o_i
        g_i      = sigmoid((ev_i - ev_threshold) / gate_temp)   # soft 買う/買わない
        (g scaled down so sum_i g_i <= max_bets, when max_bets > 0)
        profit_b = g_winner * o_winner - sum_i g_i              # 1 点 = 1 単位

    and the loss is ``-mean(profit_b)`` over races.  ``profit`` is denominated in
    *stake units*, not bankroll multiples: with flat stakes there is no
    compounding, so expected profit — not log-growth — is the quantity a flat
    bettor maximises.

    The sigmoid is the abstention: horses below the threshold contribute almost
    no stake and almost no loss.  Betting nothing scores exactly 0, which is the
    honest optimum when no +EV bet exists — so this loss will happily collapse to
    "buy nothing" and stops teaching the ranking.  Use it as a **fine-tune stage
    on top of a ``plackett_luce`` pretrain** (``--init-from``), and select
    checkpoints on ``--monitor valid_tansho_roi``.

    Unlike :func:`log_growth_loss` there is no log to tame the payoff, so a single
    100:1 winner contributes ~+100 to one race's profit and dominates the batch
    mean.  That heavy tail is inherent to the quantity being optimised (realised
    flat-stake profit); rely on ``--gradient-clip-val`` (default 1.0) rather than
    removing it, since clipping the odds would optimise a different bet.

    Args:
        scores:           [B, N]
        finish_positions: [B, N]  NaN = exclude; winner is position == 1
        odds_win:         [B, N]  **raw** 単勝 odds (NaN / <=0 → never staked)
        mask:             [B, N]  bool
        ev_threshold:     EV above which a horse is bought (deploy 側 min_ev)
        gate_temp:        sigmoid temperature; smaller = harder 買う/買わない境界
        max_bets:         per-race cap on total staked units (0 = 無制限)。
                          deploy の race_budget / stake_unit に相当。

    Returns:
        Scalar loss (mean over races with a clean, odds-carrying winner).
    """
    device = scores.device
    tau = float(ev_threshold)
    temp = max(float(gate_temp), 1e-6)
    cap = float(max_bets)
    total_loss = torch.zeros(1, device=device)
    n_valid = 0

    for p, o, w in _races_with_priced_winner(scores, finish_positions, odds_win, mask):
        o_w = o[w]

        # オッズ不明 / 非正の馬は買えない → gate 0
        o_ok = torch.nan_to_num(o, nan=0.0)
        bettable = o_ok > 0
        ev = p * o_ok
        gate = torch.sigmoid((ev - tau) / temp)
        gate = torch.where(bettable, gate, torch.zeros_like(gate))

        if cap > 0:
            staked = gate.sum()
            scale = torch.clamp(cap / torch.clamp(staked, min=1e-6), max=1.0)
            gate = gate * scale

        profit = gate[w] * o_w - gate.sum()
        total_loss = total_loss - profit
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
    replacement for the external post-hoc isotonic ``combo_calibrators``.  (A
    decision-focused betting-return objective on combos would instead suppress
    probabilities on the −EV combo markets; this loss targets calibration.)

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
    cash_fraction: float = 0.25,
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
        cash_fraction: for the log_growth term (see :func:`log_growth_loss`).
        combo_bet_type: 連系 type (or "all") for the calibration term.

    Returns:
        Scalar loss.  NaN only when *both* terms are NaN for the batch.
    """
    lg = log_growth_loss(scores, finish_positions, odds_win, mask, cash_fraction)
    cn = combo_nll_loss(scores, finish_positions, mask, bet_type=combo_bet_type)

    terms = []
    if not torch.isnan(lg):
        terms.append(lg)
    if not torch.isnan(cn):
        terms.append(combo_weight * cn)
    if not terms:
        return torch.tensor(float("nan"), device=scores.device)
    return torch.stack(terms).sum()


