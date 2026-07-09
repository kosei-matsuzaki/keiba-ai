"""Tests for RobotsCache."""

from __future__ import annotations

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


def test_fetch_failure_denies_requests():
    """robots.txt が取得できない場合は fail-closed で全リクエストを拒否する。"""
    cache = RobotsCache(user_agent="TestAgent")
    cache._load = lambda url: None  # type: ignore[method-assign]
    assert cache.is_allowed("https://db.netkeiba.com/race/1/") is False


def test_fetch_failure_is_retried_after_failure_ttl(monkeypatch):
    """失敗キャッシュは短い TTL で再試行され、成功後は許可される。"""
    from scraper import robots as robots_mod

    cache = _make_cache_with_mock_fetch(_ROBOTS_TXT)
    good_load = cache._load
    cache._load = lambda url: None  # type: ignore[method-assign]
    assert cache.is_allowed("https://db.netkeiba.com/race/1/") is False

    # 失敗 TTL 経過前はキャッシュされた失敗を使う（再取得しない）
    assert cache.is_allowed("https://db.netkeiba.com/race/1/") is False

    # 失敗 TTL を経過させると再取得し、今度は成功して許可される
    real_time = robots_mod.time.time
    monkeypatch.setattr(
        robots_mod.time, "time",
        lambda: real_time() + robots_mod._FAILURE_TTL_SECONDS + 1,
    )
    cache._load = good_load  # type: ignore[method-assign]
    assert cache.is_allowed("https://db.netkeiba.com/race/1/") is True
