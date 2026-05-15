"""Async rate limiter that enforces a minimum delay between HTTP requests.

A single AsyncRateLimiter instance serialises all callers via an asyncio.Lock,
preventing any concurrent requests regardless of how many coroutines share it.
"""

from __future__ import annotations

import asyncio
import random
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from core.config import Settings

_JST = ZoneInfo("Asia/Tokyo")


def _is_night_jst() -> bool:
    """Return True during 22:00-05:00 JST (inclusive start, exclusive end)."""
    hour = datetime.now(_JST).hour
    return hour >= 22 or hour < 5


class AsyncRateLimiter:
    """Enforces per-request delays with jitter and a night-time floor."""

    def __init__(self, settings: Settings) -> None:
        self._min = settings.rate_min_seconds
        self._max = settings.rate_max_seconds
        self._night_min = settings.night_min_seconds
        self._last_request_time: float = 0.0
        self._lock = asyncio.Lock()

    def _compute_wait(self) -> float:
        base = self._min + random.random() * (self._max - self._min)
        if _is_night_jst():
            base = max(base, self._night_min)
        elapsed = time.monotonic() - self._last_request_time
        return max(0.0, base - elapsed)

    async def acquire(self) -> None:
        """Wait the required interval, then mark the current time."""
        async with self._lock:
            wait = self._compute_wait()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_time = time.monotonic()
