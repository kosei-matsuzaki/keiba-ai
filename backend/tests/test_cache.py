"""Tests for HTML cache module."""

from __future__ import annotations

import time

import pytest

from scraper import cache as cache_module

_RACE_URL = "https://db.netkeiba.com/race/202412280101/"
_MISC_URL = "https://race.netkeiba.com/top/race_list.html?kaisai_date=20241228"


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Redirect data_dir() to a temp directory for each test."""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))


def test_write_and_read_race_url():
    html = "<html>race result</html>"
    cache_module.write_cache(_RACE_URL, html)
    result = cache_module.read_cache(_RACE_URL, max_age_hours=None)
    assert result == html


def test_read_returns_none_on_miss():
    result = cache_module.read_cache(_RACE_URL)
    assert result is None


def test_cache_path_is_under_yyyy_mm():
    cache_module.write_cache(_RACE_URL, "data")
    from scraper.cache import _cache_path
    path = _cache_path(_RACE_URL)
    # Expect .../2024/12/202412280101.html
    assert path.parts[-3] == "2024"
    assert path.parts[-2] == "12"
    assert path.name == "202412280101.html"


def test_misc_url_goes_to_misc_dir():
    cache_module.write_cache(_MISC_URL, "calendar html")
    from scraper.cache import _cache_path
    path = _cache_path(_MISC_URL)
    assert "misc" in path.parts


def test_stale_cache_returns_none():
    cache_module.write_cache(_RACE_URL, "stale data")
    from scraper.cache import _cache_path
    path = _cache_path(_RACE_URL)
    # Backdate the file mtime by 25 hours
    old_time = time.time() - 25 * 3600
    import os
    os.utime(path, (old_time, old_time))
    result = cache_module.read_cache(_RACE_URL, max_age_hours=24)
    assert result is None


def test_no_age_check_returns_stale():
    cache_module.write_cache(_RACE_URL, "stale data")
    from scraper.cache import _cache_path
    path = _cache_path(_RACE_URL)
    old_time = time.time() - 100 * 3600
    import os
    os.utime(path, (old_time, old_time))
    result = cache_module.read_cache(_RACE_URL, max_age_hours=None)
    assert result == "stale data"


def test_content_hash_consistency():
    html = "<html>test</html>"
    h1 = cache_module.content_hash(html)
    h2 = cache_module.content_hash(html)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_content_hash_differs_for_different_content():
    assert cache_module.content_hash("a") != cache_module.content_hash("b")


# ── clear_misc_cache ─────────────────────────────────────────────────────────


def test_clear_misc_cache_removes_misc_files():
    """Files written to misc/ via horse/ped URLs are deleted; race/ tree is untouched."""
    horse_url = "https://db.netkeiba.com/horse/2019105293/"
    ped_url = "https://db.netkeiba.com/horse/ped/2019105293/"
    calendar_url = "https://db.netkeiba.com/race/list/20241228/"

    cache_module.write_cache(horse_url, "horse html")
    cache_module.write_cache(ped_url, "ped html")
    cache_module.write_cache(calendar_url, "calendar html")
    cache_module.write_cache(_RACE_URL, "race html")  # → 2024/12/, NOT misc

    removed = cache_module.clear_misc_cache()

    assert removed == 3
    assert cache_module.read_cache(horse_url, max_age_hours=None) is None
    assert cache_module.read_cache(ped_url, max_age_hours=None) is None
    assert cache_module.read_cache(calendar_url, max_age_hours=None) is None
    # Race result must remain
    assert cache_module.read_cache(_RACE_URL, max_age_hours=None) == "race html"


def test_clear_misc_cache_returns_zero_when_dir_missing():
    """No-op + returns 0 when misc/ has never been created."""
    removed = cache_module.clear_misc_cache()
    assert removed == 0


def test_clear_misc_cache_idempotent():
    """Calling twice in a row is safe and the second call removes 0 files."""
    cache_module.write_cache("https://example.com/foo", "x")
    first = cache_module.clear_misc_cache()
    second = cache_module.clear_misc_cache()
    assert first == 1
    assert second == 0
