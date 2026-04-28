"""High-level HTTP client for netkeiba with cache, robots, rate-limit, and retry."""

from __future__ import annotations

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
)

from keiba_ai.core.config import Settings
from keiba_ai.core.logging import get_logger
from keiba_ai.scraper import cache as cache_module
from keiba_ai.scraper import stop_flag as stop_flag_module
from keiba_ai.scraper.rate_limiter import AsyncRateLimiter
from keiba_ai.scraper.robots import RobotsCache

logger = get_logger(__name__)

_RETRY_ATTEMPTS = 4
_BACKOFF_MULTIPLIER = 2
_PENALTY_429_SECONDS = 60


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


def _is_429(exc: BaseException) -> bool:
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429


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
            from keiba_ai.scraper.stop_flag import ScraperStopped
            raise ScraperStopped("stop flag set before fetching")

        html = await self._fetch_with_retry(url)
        cache_module.write_cache(url, html)
        return html

    async def _fetch_with_retry(self, url: str) -> str:
        """Perform rate-limited HTTP GET with exponential backoff retry."""
        import asyncio

        last_exc: Exception | None = None
        delays = [4, 8, 16, 30]

        for attempt, delay in enumerate(delays + [None], start=1):  # type: ignore[arg-type]
            if stop_flag_module.is_stopped():
                from keiba_ai.scraper.stop_flag import ScraperStopped
                raise ScraperStopped("stop flag set during retry loop")

            await self._rate.acquire()
            try:
                response = await self._http.get(
                    url,
                    headers={"User-Agent": self._settings.user_agent},
                    follow_redirects=True,
                    timeout=30.0,
                )
                if response.status_code == 429:
                    logger.warning("429 on %s — sleeping %ds", url, _PENALTY_429_SECONDS)
                    await asyncio.sleep(_PENALTY_429_SECONDS)
                    last_exc = httpx.HTTPStatusError(
                        "429", request=response.request, response=response
                    )
                    continue
                response.raise_for_status()
                return response.text
            except (httpx.TimeoutException, httpx.HTTPStatusError) as exc:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                    raise  # 4xx (except 429 handled above) — do not retry
                last_exc = exc
                if delay is not None:
                    logger.warning(
                        "Attempt %d failed for %s: %s — retrying in %ds",
                        attempt, url, exc, delay,
                    )
                    await asyncio.sleep(delay)

        raise last_exc or RuntimeError(f"All retries exhausted for {url}")
