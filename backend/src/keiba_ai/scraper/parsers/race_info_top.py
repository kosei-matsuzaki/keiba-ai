"""Parse the netkeiba race-info-top JSON API response.

Target URL:
  https://race.netkeiba.com/api/api_get_race_info_top.html?kaisai_date=YYYYMMDD

Response shape (simplified):
  {
    "status": "OK",
    "data": {
      "info": [
        {"RaceId": "202506050101", ...},
        ...
      ]
    }
  }

Each info entry contains a ``RaceId`` field (12-digit string).  We extract
the unique, sorted list of those IDs.

Error cases:
  - status != "OK"              → raise ParseError
  - data / info key missing     → raise ParseError
  - RaceId missing for an entry → the entry is silently skipped
"""

from __future__ import annotations

import re
from typing import Any

_RACE_ID_RE = re.compile(r"^\d{12}$")


class ParseError(Exception):
    """Raised when the API response cannot be parsed."""


def parse_race_ids(payload: dict[str, Any]) -> list[str]:
    """Extract unique race_ids from an api_get_race_info_top JSON payload.

    Args:
        payload: Decoded JSON dict from the netkeiba API.

    Returns:
        Sorted list of unique 12-digit race_id strings.
        Empty list when the kaisai is valid but contains no races.

    Raises:
        ParseError: When the status is not OK or the expected structure is absent.
    """
    status = payload.get("status")
    if status != "OK":
        raise ParseError(f"API returned status={status!r} (expected 'OK')")

    data = payload.get("data")
    if not isinstance(data, dict):
        raise ParseError("Missing or invalid 'data' field in API response")

    info = data.get("info")
    if not isinstance(info, list):
        raise ParseError("Missing or invalid 'data.info' field in API response")

    seen: set[str] = set()
    for entry in info:
        race_id = entry.get("RaceId") if isinstance(entry, dict) else None
        if isinstance(race_id, str) and _RACE_ID_RE.match(race_id):
            seen.add(race_id)

    return sorted(seen)
