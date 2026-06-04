"""User-facing settings persisted as JSON in the data directory."""

from __future__ import annotations

import json
from pathlib import Path

from core.bet_types import DEFAULT_ENABLED_BET_TYPES
from core.paths import data_dir

_DEFAULTS: dict = {
    "user_agent": (
        "Mozilla/5.0 (compatible; keiba-ai-research/0.1; personal research only; "
        "contact: your-email@example.com)"
    ),
    "rate_min_seconds": 3.0,
    "rate_max_seconds": 6.0,
    "night_min_seconds": 5.0,
    "win_ev_threshold": 1.1,
    "place_ev_threshold": 1.05,
    "scraper_stopped": False,
    # Bankroll / Kelly settings
    "bankroll": 100_000,
    "kelly_fraction": 0.25,
    "max_stake_per_race_pct": 0.05,
    "enabled_bet_types": list(DEFAULT_ENABLED_BET_TYPES),
}


class SettingsStore:
    """Load and persist user-editable settings as JSON."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (data_dir() / "settings.json")

    def load(self) -> dict:
        if not self._path.exists():
            return dict(_DEFAULTS)
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            # Fill any missing keys from defaults (forward-compatibility)
            merged = dict(_DEFAULTS)
            merged.update(data)
            return merged
        except (json.JSONDecodeError, OSError):
            return dict(_DEFAULTS)

    def save(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
