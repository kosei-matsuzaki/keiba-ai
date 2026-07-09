"""robots.txt fetch and compliance check with 24-hour in-memory cache.

Fail behaviour:
  On fetch failure, log a warning and deny all requests (fail-closed)。
  失敗結果は短い TTL でキャッシュし、一時的なネットワーク障害から自動回復する。
"""

from __future__ import annotations

import time
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from core.logging import get_logger

logger = get_logger(__name__)

_CACHE_TTL_SECONDS = 24 * 3600
# 取得失敗時の再試行間隔。成功時より短くして一時障害から回復できるようにする。
_FAILURE_TTL_SECONDS = 10 * 60


class RobotsFetchError(Exception):
    pass


class RobotsCache:
    """Per-domain robots.txt cache with 24-hour TTL (fail-closed on fetch error)."""

    def __init__(self, user_agent: str) -> None:
        self._user_agent = user_agent
        # domain -> (RobotFileParser | None, fetched_at)。None = 取得失敗 (拒否扱い)
        self._store: dict[str, tuple[RobotFileParser | None, float]] = {}

    def _robots_url(self, url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    def _load(self, robots_url: str) -> RobotFileParser | None:
        rp = RobotFileParser()
        rp.set_url(robots_url)
        try:
            rp.read()
        except Exception as exc:
            logger.warning(
                "Failed to fetch %s: %s — denying all requests (fail-closed)",
                robots_url, exc,
            )
            return None
        return rp

    def _get_parser(self, url: str) -> RobotFileParser | None:
        domain = urlparse(url).netloc
        entry = self._store.get(domain)
        if entry is not None:
            rp, fetched_at = entry
            ttl = _CACHE_TTL_SECONDS if rp is not None else _FAILURE_TTL_SECONDS
            if time.time() - fetched_at <= ttl:
                return rp
        robots_url = self._robots_url(url)
        rp = self._load(robots_url)
        self._store[domain] = (rp, time.time())
        return rp

    def is_allowed(self, url: str) -> bool:
        rp = self._get_parser(url)
        if rp is None:
            return False
        return rp.can_fetch(self._user_agent, url)
