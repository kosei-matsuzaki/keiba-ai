"""High-level HTTP client for netkeiba with cache, robots, rate-limit, and retry."""

from __future__ import annotations

import asyncio

import httpx

from core.config import Settings
from core.logging import get_logger
from scraper import cache as cache_module
from scraper import stop_flag as stop_flag_module
from scraper.rate_limiter import AsyncRateLimiter
from scraper.robots import RobotsCache
from scraper.stop_flag import ScraperStopped

logger = get_logger(__name__)

# 5xx / Timeout 用の指数バックオフ秒数。リスト長 = 最大リトライ回数
_BACKOFF_DELAYS = (4, 8, 16, 30)

# 429 受信時のペナルティ秒と最大リトライ回数（5xx の retry slot とは独立）
_PENALTY_429_SECONDS = 60
_MAX_429_RETRIES = 3


class NetkeibaClient:
    """Fetches netkeiba pages with caching, rate limiting, and retry logic."""

    def __init__(
        self,
        rate_limiter: AsyncRateLimiter,
        robots_cache: RobotsCache,
        http_client: httpx.AsyncClient,
        settings: Settings,
    ) -> None:
        self._rate = rate_limiter
        self._robots = robots_cache
        self._http = http_client
        self._settings = settings

    async def fetch(
        self,
        url: str,
        *,
        use_cache: bool = True,
        cache_max_age_hours: float = 24 * 30,
    ) -> str:
        """Fetch URL, returning HTML.

        Checks local cache first, then robots.txt, then applies rate limiting
        and performs the actual HTTP request with retry logic.
        """
        if use_cache:
            cached = cache_module.read_cache(url, max_age_hours=cache_max_age_hours)
            if cached is not None:
                logger.debug("Cache hit: %s", url)
                return cached

        if not self._robots.is_allowed(url):
            raise PermissionError(f"robots.txt disallows: {url}")

        if stop_flag_module.is_stopped():
            raise ScraperStopped("stop flag set before fetching")

        html = await self._fetch_with_retry(url)
        cache_module.write_cache(url, html)
        return html

    async def _fetch_with_retry(self, url: str) -> str:
        """Rate-limited GET with exponential backoff for 5xx/Timeout and a separate 429 penalty loop."""
        backoff_idx = 0
        n_429 = 0

        while True:
            if stop_flag_module.is_stopped():
                raise ScraperStopped("stop flag set during retry loop")

            await self._rate.acquire()
            try:
                response = await self._http.get(
                    url,
                    headers={"User-Agent": self._settings.user_agent},
                    follow_redirects=True,
                    timeout=30.0,
                )
            except httpx.TimeoutException:
                if backoff_idx >= len(_BACKOFF_DELAYS):
                    raise
                delay = _BACKOFF_DELAYS[backoff_idx]
                backoff_idx += 1
                logger.warning("Timeout on %s — retrying in %ds", url, delay)
                await asyncio.sleep(delay)
                continue

            status = response.status_code

            if status == 429:
                n_429 += 1
                if n_429 > _MAX_429_RETRIES:
                    response.raise_for_status()
                logger.warning(
                    "429 on %s (attempt %d/%d) — sleeping %ds",
                    url, n_429, _MAX_429_RETRIES, _PENALTY_429_SECONDS,
                )
                await asyncio.sleep(_PENALTY_429_SECONDS)
                continue

            if 500 <= status < 600:
                if backoff_idx >= len(_BACKOFF_DELAYS):
                    response.raise_for_status()
                delay = _BACKOFF_DELAYS[backoff_idx]
                backoff_idx += 1
                logger.warning("HTTP %d on %s — retrying in %ds", status, url, delay)
                await asyncio.sleep(delay)
                continue

            response.raise_for_status()
            # netkeiba は EUC-JP で配信されるが Content-Type に charset が
            # 含まれないことがあり、httpx のデフォルトデコード（utf-8）では
            # 不正バイトが U+FFFD に置換されて壊れる。EUC-JP を明示する。
            response.encoding = "euc-jp"
            return response.text
