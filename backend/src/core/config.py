"""Application settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

_DEFAULT_UA = (
    "Mozilla/5.0 (compatible; keiba-ai-research/0.1; personal research only; "
    "contact: your-email@example.com)"
)


@dataclass(frozen=True)
class Settings:
    user_agent: str = field(default_factory=lambda: os.getenv("KEIBA_USER_AGENT", _DEFAULT_UA))
    rate_min_seconds: float = field(
        default_factory=lambda: float(os.getenv("KEIBA_RATE_MIN_SECONDS", "3.0"))
    )
    rate_max_seconds: float = field(
        default_factory=lambda: float(os.getenv("KEIBA_RATE_MAX_SECONDS", "6.0"))
    )
    night_min_seconds: float = field(
        default_factory=lambda: float(os.getenv("KEIBA_NIGHT_MIN_SECONDS", "5.0"))
    )
    data_dir: str = field(default_factory=lambda: os.getenv("KEIBA_DATA_DIR", ""))


def load_settings() -> Settings:
    return Settings()
