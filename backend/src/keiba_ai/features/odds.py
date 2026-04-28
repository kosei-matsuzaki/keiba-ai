"""Odds and market-implied features."""

from __future__ import annotations

import math

from keiba_ai.db.models.entry import Entry


def extract_odds_features(entry: Entry) -> dict[str, float | int | None]:
    """Extract odds and popularity features for a single entry."""
    odds = entry.odds_win
    log_odds = math.log(odds) if (odds is not None and odds > 0) else math.nan

    return {
        "odds_win": odds,
        "popularity": entry.popularity,
        "log_odds_win": log_odds,
    }
