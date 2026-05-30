"""Plackett-Luce log-likelihood loss for LightGBM custom objective.

Replaces the two-model pipeline (lambdarank + binary classifier) with a
single model whose scores are directly interpretable as log-utilities in a
Plackett-Luce model.  The softmax of scores within a race gives calibrated
win probabilities, eliminating the systematic bias that arises when a separate
binary head disagrees with the ranking head.

Mathematical background
-----------------------
Given a race with n horses and observed finish order σ = (σ(1), ..., σ(n))
(σ(k) is the index of the horse that finished k-th), the Plackett-Luce
log-likelihood is:

    log P(σ | s) = Σ_{k=1}^{n-1} [s_{σ(k)} - logsumexp(s_{σ(k)}, ..., s_{σ(n)})]

We minimise L = -log P(σ | s).

Gradient for horse i
--------------------
    g_i = -(indicator that i finished before all remaining horses at its stage)
          + Σ_{k: i ∈ stage_k} softmax(s_i in stage_k)

where stage_k = {σ(k), σ(k+1), ..., σ(n)}.

Concretely: for the sorted list σ(1), σ(2), ..., σ(n), horse i = σ(r)
participates in stages k = 1, 2, ..., r.  At each stage k, we add
p_{i,k} = softmax(s_i in {σ(k), ..., σ(n)}) to g_i, and subtract 1 for the
stage where i is the chooser (k = r, i.e., i = σ(k)).

Diagonal Hessian
----------------
    h_i ≈ Σ_{k: i ∈ stage_k} p_{ik} (1 - p_{ik})

This is the Bernoulli variance of the softmax probability at each stage where
horse i is still in the running.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _race_grad_hess(
    s: np.ndarray,
    finish_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PL gradient and diagonal hessian for a single race.

    Args:
        s: Raw model scores for each horse in the race (length n).
        finish_positions: Observed finish positions (1-based integers).  Horses
            with NaN or non-positive positions are excluded from the loss
            computation; their gradient/hessian entries remain 0.

    Returns:
        (grad, hess) arrays of length n.  LightGBM convention: the objective
        minimises L, so grad = ∂L/∂s_i (positive means "push score down").
    """
    n = len(s)
    grad = np.zeros(n, dtype=np.float64)
    hess = np.zeros(n, dtype=np.float64)

    # Identify valid finishers (finish_position is a positive integer)
    valid_mask = np.isfinite(finish_positions) & (finish_positions > 0)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < 2:
        # Need at least two runners to have a ranking loss.
        return grad, hess

    # Sort valid horses by finish position ascending (σ(1), σ(2), ..., σ(m))
    sorted_order = valid_indices[np.argsort(finish_positions[valid_indices])]
    m = len(sorted_order)

    # Compute gradient and hessian by iterating over stages.
    # At stage k (0-indexed), the "remaining set" is sorted_order[k:].
    # We compute softmax of scores within that set.
    # Horse sorted_order[k] is the chooser (finished k+1-th among valid).
    for k in range(m - 1):
        # Remaining horses at this stage (k-th finisher and all after)
        stage_idx = sorted_order[k:]
        s_stage = s[stage_idx]

        # Numerically stable softmax
        s_max = s_stage.max()
        exp_s = np.exp(s_stage - s_max)
        sum_exp = exp_s.sum()
        p_k = exp_s / sum_exp  # softmax probabilities in stage k

        # p_k[j] corresponds to stage_idx[j]
        for j, horse_idx in enumerate(stage_idx):
            grad[horse_idx] += p_k[j]
            hess[horse_idx] += p_k[j] * (1.0 - p_k[j])

        # The k-th finisher "chose" itself: subtract 1 from gradient
        chooser_idx = sorted_order[k]
        grad[chooser_idx] -= 1.0

    return grad, hess


def plackett_luce_objective(group_indices: list[int]) -> Callable:
    """Build a LightGBM custom objective implementing Plackett-Luce loss.

    The returned callable matches the LightGBM objective signature:
        f(preds, train_data) -> (grad, hess)

    Args:
        group_indices: Race group sizes in dataset row order.  This must match
            the ``group`` argument passed to ``lgb.Dataset``.  Passed via
            closure because LightGBM's objective signature does not allow extra
            arguments.

    Returns:
        A callable that LightGBM calls at each boosting round.

    Notes:
        - ``train_data.get_label()`` is expected to return finish_position
          values (1-based integers, NaN/0 for non-finishers).
        - The loss treats the first occurrence of each position within a race
          as the canonical winner for that position (ties are broken by the
          order they appear in the data, which is consistent with the issue
          spec's "simplification OK" note).
        - Horses with NaN or non-positive finish_position are excluded from
          the loss but their gradient remains 0 (they do not contribute to
          the ranking signal).
    """
    cumulative_sizes = np.cumsum([0] + list(group_indices))

    def objective(preds: np.ndarray, train_data) -> tuple[np.ndarray, np.ndarray]:
        labels = train_data.get_label()
        n_total = len(preds)
        grad = np.zeros(n_total, dtype=np.float64)
        hess = np.zeros(n_total, dtype=np.float64)

        for race_idx in range(len(group_indices)):
            start = cumulative_sizes[race_idx]
            end = cumulative_sizes[race_idx + 1]
            s = preds[start:end].astype(np.float64)
            fp = labels[start:end].astype(np.float64)
            g, h = _race_grad_hess(s, fp)
            grad[start:end] = g
            hess[start:end] = h

        # LightGBM requires hessian > 0 for all leaves; clip to small positive.
        hess = np.clip(hess, 1e-6, None)
        return grad, hess

    return objective


def plackett_luce_eval_metric(group_indices: list[int]) -> Callable:
    """Build a LightGBM custom eval metric: mean negative log-likelihood per race.

    Lower is better (LightGBM convention: return ``(name, value, is_higher_better)``).

    Args:
        group_indices: Same race group sizes as ``plackett_luce_objective``.

    Returns:
        A callable matching the LightGBM feval signature:
            f(preds, train_data) -> (name, value, is_higher_better)
    """
    cumulative_sizes = np.cumsum([0] + list(group_indices))

    def eval_metric(preds: np.ndarray, train_data) -> tuple[str, float, bool]:
        labels = train_data.get_label()
        total_nll = 0.0
        n_races = 0

        for race_idx in range(len(group_indices)):
            start = cumulative_sizes[race_idx]
            end = cumulative_sizes[race_idx + 1]
            s = preds[start:end].astype(np.float64)
            fp = labels[start:end].astype(np.float64)

            valid_mask = np.isfinite(fp) & (fp > 0)
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) < 2:
                continue

            sorted_order = valid_indices[np.argsort(fp[valid_indices])]
            m = len(sorted_order)

            nll = 0.0
            for k in range(m - 1):
                stage_idx = sorted_order[k:]
                s_stage = s[stage_idx]
                # log P = s_{σ(k)} - logsumexp(s_{σ(k)}, ..., s_{σ(m)})
                s_max = s_stage.max()
                log_sum_exp = s_max + np.log(np.exp(s_stage - s_max).sum())
                nll += log_sum_exp - s[sorted_order[k]]

            total_nll += nll
            n_races += 1

        mean_nll = total_nll / n_races if n_races > 0 else 0.0
        return "pl_nll", mean_nll, False  # False = lower is better

    return eval_metric
