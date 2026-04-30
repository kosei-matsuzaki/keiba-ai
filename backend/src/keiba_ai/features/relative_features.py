"""Within-race relative features computed across all horses in the same race.

All computations are pure in-memory operations over the provided Entry list —
no DB queries, no date-based leakage risk.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from keiba_ai.db.models.entry import Entry


def _percentile_rank(values: list[float]) -> list[float]:
    """Map each value to its percentile rank in [0.0, 1.0].

    Uses ordinal ranking via np.argsort. Ties receive different ranks based on
    their original order (stable). When only one non-NaN value exists the rank
    is 0.5 (midpoint).

    NaN inputs map to NaN output; the valid subset is ranked then values are
    placed back at original indices.
    """
    n = len(values)
    result = [math.nan] * n

    valid_indices = [i for i, v in enumerate(values) if not math.isnan(v)]
    if not valid_indices:
        return result

    valid_vals = np.array([values[i] for i in valid_indices], dtype=float)
    order = np.argsort(valid_vals, stable=True)
    m = len(valid_vals)

    for rank, orig_pos in enumerate(order):
        pct = rank / (m - 1) if m > 1 else 0.5
        result[valid_indices[orig_pos]] = float(pct)

    return result


def compute_within_race_features(
    entries: list[Entry],
    *,
    jockey_recent_win_rates: dict[str, float] | None = None,
    horse_course_place_rates: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Compute relative features for each horse within a single race.

    Args:
        entries: All Entry objects for a single race.
        jockey_recent_win_rates: Pre-computed {horse_id: win_rate} mapping for
            the jockey assigned to each horse. Passed in from builder to avoid
            re-querying per-entry.
        horse_course_place_rates: Pre-computed {horse_id: place_rate} mapping
            for same-course place rate of each horse.

    Returns:
        {horse_id: {feature_name: value}} where value is float or NaN.

    Feature definitions:
        horse_weight_pct: percentile of horse_weight within the race field [0–1]
        odds_win_rank: ordinal rank of odds_win (1 = lowest odds = favourite)
        weight_carried_pct: percentile of weight_carried within the race field [0–1]
        jockey_recent_win_rate_vs_field: jockey win rate minus race-average win rate
        course_place_rate_vs_field: horse course place rate minus race-average
        odds_win_diff_from_favorite: this horse's odds_win minus minimum odds_win in race
    """
    if not entries:
        return {}

    horse_ids = [e.horse_id for e in entries]
    nan = math.nan

    # horse_weight percentile
    weights = [float(e.horse_weight) if e.horse_weight is not None else nan for e in entries]
    weight_pcts = _percentile_rank(weights)

    # weight_carried percentile
    carried = [float(e.weight_carried) if e.weight_carried is not None else nan for e in entries]
    carried_pcts = _percentile_rank(carried)

    # odds_win rank (1 = favourite = lowest odds)
    odds_values = [float(e.odds_win) if e.odds_win is not None else nan for e in entries]
    valid_odds = [v for v in odds_values if not math.isnan(v)]
    min_odds = min(valid_odds) if valid_odds else nan

    odds_ranks: list[float] = []
    if valid_odds:
        # argsort ascending → lowest odds gets rank 1
        valid_pairs = [(v, i) for i, v in enumerate(odds_values) if not math.isnan(v)]
        sorted_valid = sorted(valid_pairs, key=lambda x: x[0])
        rank_map: dict[int, float] = {}
        for rank_idx, (_, orig_i) in enumerate(sorted_valid):
            rank_map[orig_i] = float(rank_idx + 1)
        for i in range(len(entries)):
            odds_ranks.append(rank_map.get(i, nan))
    else:
        odds_ranks = [nan] * len(entries)

    # odds_win_diff_from_favorite
    odds_diffs = [
        (v - min_odds) if not math.isnan(v) and not math.isnan(min_odds) else nan
        for v in odds_values
    ]

    # jockey_recent_win_rate_vs_field
    if jockey_recent_win_rates is not None:
        jwr_values = [
            jockey_recent_win_rates.get(hid, nan) for hid in horse_ids
        ]
        valid_jwr = [v for v in jwr_values if not math.isnan(v)]
        field_avg_jwr = sum(valid_jwr) / len(valid_jwr) if valid_jwr else nan
        jwr_vs_field = [
            (v - field_avg_jwr) if not math.isnan(v) and not math.isnan(field_avg_jwr) else nan
            for v in jwr_values
        ]
    else:
        jwr_vs_field = [nan] * len(entries)

    # course_place_rate_vs_field
    if horse_course_place_rates is not None:
        cpr_values = [
            horse_course_place_rates.get(hid, nan) for hid in horse_ids
        ]
        valid_cpr = [v for v in cpr_values if not math.isnan(v)]
        field_avg_cpr = sum(valid_cpr) / len(valid_cpr) if valid_cpr else nan
        cpr_vs_field = [
            (v - field_avg_cpr) if not math.isnan(v) and not math.isnan(field_avg_cpr) else nan
            for v in cpr_values
        ]
    else:
        cpr_vs_field = [nan] * len(entries)

    result: dict[str, dict] = {}
    for i, entry in enumerate(entries):
        result[entry.horse_id] = {
            "horse_weight_pct": weight_pcts[i],
            "odds_win_rank": odds_ranks[i],
            "weight_carried_pct": carried_pcts[i],
            "jockey_recent_win_rate_vs_field": jwr_vs_field[i],
            "course_place_rate_vs_field": cpr_vs_field[i],
            "odds_win_diff_from_favorite": odds_diffs[i],
        }

    return result
