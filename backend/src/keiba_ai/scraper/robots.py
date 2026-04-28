"""robots.txt fetch and compliance check with 24-hour in-memory cache.

Fail behaviour (M2):
  On fetch failure, log a warning and allow the request (fail-open).
  TODO(M3): change to fail-fast (raise RobotsFetchError) once we have
  reliable connectivity monitoring in place.
"""

from __future__ import annotations

import time
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from keiba_ai.core.logging import get_logger

logger = get_logger(__name__)

_CACHE_TTL_SECONDS = 24 * 3600


class RobotsFetchError(Exception):
    pass


class RobotsCache:
    """Per-domain robots.txt cache with 24-hour TTL."""

    def __init__(self, user_agent: str) -> None:
        self._user_agent = user_agent
        # domain -> (RobotFileParser, fetched_at)
        self._store: dict[str, tuple[RobotFileParser, float]] = {}

    def _robots_url(self, url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    def _load(self, robots_url: str) -> RobotFileParser:
        rp = RobotFileParser()
        rp.set_url(robots_url)
        try:
            rp.read()
        except Exception as exc:
            # M2: fail-open with warning.  M3 should raise RobotsFetchError.
            logger.warning("Failed to fetch %s: %s — allowing all requests", robots_url, exc)
        return rp

    def _get_parser(self, url: str) -> RobotFileParser:
        domain = urlparse(url).netloc
        entry = self._store.get(domain)
        if entry is None or time.time() - entry[1] > _CACHE_TTL_SECONDS:
            robots_url = self._robots_url(url)
            rp = self._load(robots_url)
            self._store[domain] = (rp, time.time())
        return self._store[domain][0]

    def is_allowed(self, url: str) -> bool:
        rp = self._get_parser(url)
        return rp.can_fetch(self._user_agent, url)
