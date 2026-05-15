"""Global scraper stop switch.

Two stop mechanisms:
  1. Environment variable KEIBA_SCRAPER_STOP=1 (checked at call time)
  2. In-process flag set via set_stopped() (useful for API-triggered stops)

Usage in scraping loops:
    if stop_flag.is_stopped():
        raise ScraperStopped("stop flag is set")
"""

from __future__ import annotations

import os

_internal_stop: bool = False


class ScraperStopped(Exception):
    pass


def is_stopped() -> bool:
    return _internal_stop or os.getenv("KEIBA_SCRAPER_STOP", "").strip() == "1"


def set_stopped() -> None:
    global _internal_stop
    _internal_stop = True


def clear_stopped() -> None:
    global _internal_stop
    _internal_stop = False
