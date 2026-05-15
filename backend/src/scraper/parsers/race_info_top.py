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

# JRA 開催場コード (race_id の 5-6 桁目): 01=札幌 … 10=小倉
# NAR (地方) は 11〜 なので '01'〜'10' のみが JRA
_JRA_VENUE_CODES = frozenset(f"{i:02d}" for i in range(1, 11))


class ParseError(Exception):
    """Raised when the API response cannot be parsed."""


def _extract_info_list(payload: dict[str, Any]) -> list[Any]:
    """Validate payload structure and return data.info list.

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

    return info


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
    info = _extract_info_list(payload)

    seen: set[str] = set()
    for entry in info:
        race_id = entry.get("RaceId") if isinstance(entry, dict) else None
        if isinstance(race_id, str) and _RACE_ID_RE.match(race_id):
            seen.add(race_id)

    return sorted(seen)


def extract_jra_race_ids_with_kaisai_groups(
    payload: dict[str, Any],
) -> tuple[list[str], dict[str, list[str]]]:
    """Extract JRA-only race_ids grouped by kaisai-day key.

    JRA venues are race_id[4:6] in '01'..'10'; NAR codes are '11'+.

    Args:
        payload: Decoded JSON dict from the netkeiba API.

    Returns:
        A 2-tuple of:
          - jra_race_ids: Sorted list of all JRA race_ids found.
          - groups: Dict mapping kaisai_day_key → list[race_id].
            kaisai_day_key = race_id[:8] + race_id[8:10]  (YYYYMMDDNN where NN
            is the 2-digit kaisai-day ordinal from the race_id).
            Each list is sorted ascending.

    Raises:
        ParseError: When the status is not OK or the expected structure is absent.
    """
    info = _extract_info_list(payload)

    # netkeiba の race_info_top API は同じ race_id を複数回 (賭式ごと等) 返すため、
    # set で重複を排除してから list 化する。
    groups_set: dict[str, set[str]] = {}
    for entry in info:
        race_id = entry.get("RaceId") if isinstance(entry, dict) else None
        if not isinstance(race_id, str) or not _RACE_ID_RE.match(race_id):
            continue
        venue_code = race_id[4:6]
        if venue_code not in _JRA_VENUE_CODES:
            continue
        # kaisai_day_key = first 10 chars: YYYY MM DD NN
        # where NN = race_id[8:10] is the kaisai-day ordinal (01=1st racing day…)
        key = race_id[:10]
        groups_set.setdefault(key, set()).add(race_id)

    groups: dict[str, list[str]] = {key: sorted(ids) for key, ids in groups_set.items()}
    jra_race_ids = sorted(rid for ids in groups.values() for rid in ids)
    return jra_race_ids, groups
