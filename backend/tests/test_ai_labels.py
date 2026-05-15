"""Tests for ai/labels.py — boundary value verification."""

from __future__ import annotations

import pytest

from ai.labels import assign_relevance


@pytest.mark.parametrize(
    ("finish_position", "expected"),
    [
        (None, 0),
        (1, 4),
        (2, 3),
        (3, 2),
        (4, 1),
        (5, 1),
        (6, 0),
        (10, 0),
        (18, 0),
    ],
)
def test_assign_relevance(finish_position, expected):
    assert assign_relevance(finish_position) == expected
