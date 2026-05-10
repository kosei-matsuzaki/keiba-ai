"""Ranking loss functions for horse-race prediction.

All functions operate on batched tensors:
    scores           [B, N]  — model logits / scores per horse
    finish_positions [B, N]  — ground-truth finishing position (1-based, NaN = unknown)
    mask             [B, N]  — bool, True = valid horse, False = padded slot

Losses are reduced to a scalar mean over *valid* races / pairs.
"""

from __future__ import annotations

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

    for b in range(B := scores.size(0)):
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
