"""Course and race-level feature extraction."""

from __future__ import annotations

from db.models.entry import Entry
from db.models.race import Race


def extract_race_features(race: Race, entry: Entry, n_runners: int) -> dict[str, object]:
    """Extract race-level and position features for a single entry."""
    post = entry.post_position
    post_ratio = (post / n_runners) if (post is not None and n_runners and n_runners > 0) else None

    return {
        "distance": race.distance,
        "surface": race.surface,
        "course": race.course,
        "weather": race.weather,
        "track_condition": race.track_condition,
        "race_class": race.race_class,
        "n_runners": n_runners,
        "post_position": post,
        "post_position_ratio": post_ratio,
        "age": entry.age,
        "sex": entry.sex,
        "horse_weight": entry.horse_weight,
        "horse_weight_diff": entry.horse_weight_diff,
    }
