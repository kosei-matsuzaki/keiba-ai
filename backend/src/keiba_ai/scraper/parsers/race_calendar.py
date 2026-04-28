"""Parse netkeiba race calendar page → list of race IDs.

Target URL:
  https://race.netkeiba.com/top/race_list.html?kaisai_date=YYYYMMDD

Assumed HTML structure (to be verified against real pages in M2 manual QA):
  <a class="RaceList_ItemTitle" href="/race/result.html?race_id=202412280101">...</a>
  or any <a href> containing /race/ followed by a 12-digit race_id.

If zero race IDs are extracted, raise ParseError so the caller can log and
decide whether to continue or abort.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from keiba_ai.core.logging import get_logger

logger = get_logger(__name__)

_RACE_ID_RE = re.compile(r"race_id=(\d{12})")
_RACE_PATH_RE = re.compile(r"/race/(\d{12})")


class ParseError(Exception):
    pass


def parse_race_ids_from_calendar(html: str) -> list[str]:
    """Extract race IDs from a kaisai_date calendar page.

    Searches all <a> tags whose href contains a 12-digit race_id in either of:
      - ?race_id=<id> query parameter
      - /race/<id>/ path segment

    Raises:
        ParseError: If no race IDs could be found (likely a selector change).
    """
    soup = BeautifulSoup(html, "lxml")
    race_ids: list[str] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        m = _RACE_ID_RE.search(href) or _RACE_PATH_RE.search(href)
        if m:
            race_id = m.group(1)
            if race_id not in seen:
                seen.add(race_id)
                race_ids.append(race_id)

    if not race_ids:
        logger.error(
            "No race IDs found in calendar HTML — netkeiba page structure may have changed"
        )
        raise ParseError("No race IDs found in calendar HTML")

    logger.info("Extracted %d race IDs from calendar", len(race_ids))
    return race_ids
