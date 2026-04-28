"""Tests for AsyncRateLimiter."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from keiba_ai.core.config import Settings
from keiba_ai.scraper.rate_limiter import AsyncRateLimiter, _is_night_jst


@pytest.mark.asyncio
async def test_acquire_enforces_min_delay():
    settings = Settings(rate_min_seconds=0.05, rate_max_seconds=0.05)
    limiter = AsyncRateLimiter(settings)

    t0 = time.monotonic()
    await limiter.acquire()
    t1 = time.monotonic()
    await limiter.acquire()
    t2 = time.monotonic()

    # Second acquire must wait at least min_seconds after first
    assert (t2 - t1) >= 0.04  # allow small floating-point tolerance


@pytest.mark.asyncio
async def test_first_acquire_does_not_block_long():
    """First request should not wait since last_request_time is 0."""
    settings = Settings(rate_min_seconds=0.1, rate_max_seconds=0.1)
    limiter = AsyncRateLimiter(settings)

    t0 = time.monotonic()
    await limiter.acquire()
    elapsed = time.monotonic() - t0
    # First call: elapsed since epoch is huge, so no wait needed
    assert elapsed < 0.5


@pytest.mark.asyncio
async def test_night_min_seconds_applied():
    """During night hours the wait should use night_min_seconds."""
    settings = Settings(rate_min_seconds=0.01, rate_max_seconds=0.01, night_min_seconds=0.1)
    limiter = AsyncRateLimiter(settings)

    # Simulate night time (hour=23)
    from datetime import datetime
    from zoneinfo import ZoneInfo
    fake_now = datetime(2024, 12, 28, 23, 0, 0, tzinfo=ZoneInfo("Asia/Tokyo"))

    with patch("keiba_ai.scraper.rate_limiter.datetime") as mock_dt:
        mock_dt.now.return_value = fake_now
        # First acquire to record last_request_time
        await limiter.acquire()
        # Force last_request_time to now so wait is full
        limiter._last_request_time = time.monotonic()

        t0 = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - t0

    assert elapsed >= 0.08  # should have waited ~night_min_seconds


def test_is_night_jst_daytime():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    daytime = datetime(2024, 12, 28, 12, 0, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
    with patch("keiba_ai.scraper.rate_limiter.datetime") as mock_dt:
        mock_dt.now.return_value = daytime
        assert _is_night_jst() is False


def test_is_night_jst_night():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    night = datetime(2024, 12, 28, 23, 30, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
    with patch("keiba_ai.scraper.rate_limiter.datetime") as mock_dt:
        mock_dt.now.return_value = night
        assert _is_night_jst() is True


def test_is_night_jst_early_morning():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    early = datetime(2024, 12, 28, 3, 0, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
    with patch("keiba_ai.scraper.rate_limiter.datetime") as mock_dt:
        mock_dt.now.return_value = early
        assert _is_night_jst() is True
