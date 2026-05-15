"""Tests for RobotsCache."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scraper.robots import RobotsCache

_ROBOTS_TXT = """\
User-agent: *
Disallow: /member/
Disallow: /cart/

User-agent: BadBot
Disallow: /
"""


def _make_cache_with_mock_fetch(robots_content: str, user_agent: str = "TestAgent") -> RobotsCache:
    """Return a RobotsCache that loads the given robots.txt text without network access."""
    cache = RobotsCache(user_agent=user_agent)

    def fake_load(robots_url: str):
        from io import StringIO
        from urllib.robotparser import RobotFileParser
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.parse(robots_content.splitlines())
        return rp

    cache._load = fake_load  # type: ignore[method-assign]
    return cache


def test_allowed_path():
    cache = _make_cache_with_mock_fetch(_ROBOTS_TXT)
    assert cache.is_allowed("https://db.netkeiba.com/race/202412280101/") is True


def test_disallowed_path():
    cache = _make_cache_with_mock_fetch(_ROBOTS_TXT)
    assert cache.is_allowed("https://db.netkeiba.com/member/login") is False


def test_allowed_for_different_user_agent():
    """BadBot is fully disallowed; TestAgent should still be allowed for /."""
    cache = _make_cache_with_mock_fetch(_ROBOTS_TXT, user_agent="TestAgent")
    assert cache.is_allowed("https://db.netkeiba.com/") is True


def test_ttl_uses_cached_parser():
    """Second call to the same domain should reuse the cached parser."""
    cache = _make_cache_with_mock_fetch(_ROBOTS_TXT)
    load_count = 0
    original_load = cache._load

    def counting_load(url: str):
        nonlocal load_count
        load_count += 1
        return original_load(url)

    cache._load = counting_load  # type: ignore[method-assign]
    cache.is_allowed("https://db.netkeiba.com/race/1/")
    cache.is_allowed("https://db.netkeiba.com/race/2/")
    assert load_count == 1  # fetched only once
