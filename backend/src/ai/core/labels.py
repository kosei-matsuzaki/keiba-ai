"""Relevance label assignment for the NN ranking pretrain / NDCG eval."""

from __future__ import annotations


def assign_relevance(finish_position: int | None) -> int:
    """Map finish_position to a 0-4 relevance label.

    Higher is better (1st place = 4). Non-finishers and positions beyond 5th = 0.
    """
    if finish_position is None:
        return 0
    if finish_position == 1:
        return 4
    if finish_position == 2:
        return 3
    if finish_position == 3:
        return 2
    if finish_position in (4, 5):
        return 1
    return 0


def assign_is_winner(finish_position: int | None) -> int:
    """Map finish_position to 0/1 binary "won the race" label.

    Maps finish_position to a 0/1 "won the race" label (win-rate eval helper).

    Returns:
        1 if finish_position == 1, else 0 (including None / DNF).
    """
    return 1 if finish_position == 1 else 0
