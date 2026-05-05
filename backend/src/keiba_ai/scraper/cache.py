"""HTML cache backed by the local filesystem.

Cache layout:
  data/raw/<yyyy>/<mm>/<race_id>.html           — race result pages
  data/raw/<yyyy>/<mm>/shutuba_<race_id>.html   — shutuba (出馬表) pages
  data/raw/misc/<sha256(url)[:16]>.html         — all other URLs
"""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path

from keiba_ai.core.paths import raw_dir


_RACE_ID_RE = re.compile(r"/race/(\d{12})")
_SHUTUBA_RACE_ID_RE = re.compile(r"race_id=(\d{12})")
_MISC_DIR_NAME = "misc"


def _cache_path(url: str) -> Path:
    # shutuba page: race.netkeiba.com/race/shutuba.html?race_id=<id>
    m_shutuba = _SHUTUBA_RACE_ID_RE.search(url)
    if m_shutuba and "shutuba" in url:
        race_id = m_shutuba.group(1)
        yyyy, mm = race_id[:4], race_id[4:6]
        base = raw_dir() / yyyy / mm
        base.mkdir(parents=True, exist_ok=True)
        return base / f"shutuba_{race_id}.html"

    m = _RACE_ID_RE.search(url)
    if m:
        race_id = m.group(1)
        yyyy, mm = race_id[:4], race_id[4:6]
        base = raw_dir() / yyyy / mm
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{race_id}.html"

    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    misc = raw_dir() / _MISC_DIR_NAME
    misc.mkdir(parents=True, exist_ok=True)
    return misc / f"{url_hash}.html"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def read_cache(url: str, max_age_hours: float | None = 24) -> str | None:
    """Return cached HTML if it exists and is fresh enough.

    Args:
        url: The request URL used as cache key.
        max_age_hours: Maximum cache age in hours.  None means no age check.

    Returns:
        The cached HTML string, or None on miss / stale.
    """
    path = _cache_path(url)
    if not path.exists():
        return None
    if max_age_hours is not None:
        age_seconds = time.time() - path.stat().st_mtime
        if age_seconds > max_age_hours * 3600:
            return None
    return path.read_text(encoding="utf-8")


def write_cache(url: str, html: str) -> Path:
    """Write HTML to the cache and return the file path."""
    path = _cache_path(url)
    path.write_text(html, encoding="utf-8")
    return path


def content_hash(html: str) -> str:
    return _sha256(html)


def clear_misc_cache() -> int:
    """Remove all cached files under data/raw/misc/.

    `misc/` holds horse_detail / horse_pedigree / calendar HTML — content that
    is one-time-use during ingest and offers no value once parsed into the DB.
    Long-running range ingests can grow this to many GB of unique horse pages,
    so we clear it after each successful day to keep disk usage bounded.

    race_result HTML lives under `data/raw/<YYYY>/<MM>/` and is intentionally
    untouched so parser fixes can re-run against cached pages.

    Returns:
        Number of files removed (0 if the directory does not exist).
    """
    misc = raw_dir() / _MISC_DIR_NAME
    if not misc.exists():
        return 0
    count = 0
    for f in misc.iterdir():
        if f.is_file():
            f.unlink()
            count += 1
    return count
