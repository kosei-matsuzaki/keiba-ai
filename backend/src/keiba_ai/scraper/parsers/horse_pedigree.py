"""Parse netkeiba horse pedigree page (https://db.netkeiba.com/horse/ped/<id>/)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

from keiba_ai.core.logging import get_logger

logger = get_logger(__name__)

_HORSE_HREF_RE = re.compile(r"/horse/(\d{10})/?")


@dataclass
class ParsedPedigree:
    horse_id: str
    sire_id: str | None = None
    sire_name: str | None = None
    dam_id: str | None = None
    dam_name: str | None = None


def parse_horse_pedigree(html: str, horse_id: str) -> ParsedPedigree:
    """Parse pedigree page. Identifies sire and dam from blood_table rowspan structure.

    The blood_table uses rowspan to lay out generations. The largest rowspan TDs
    are the parents (sire and dam). For a 3-generation table parent rowspan=8;
    for 5-generation, 16. We pick the two TDs with the maximum rowspan as parents.
    First = sire (父), second = dam (母).
    """
    soup = BeautifulSoup(html, "lxml")
    result = ParsedPedigree(horse_id=horse_id)

    table = soup.find("table", class_=re.compile(r"blood_table"))
    if table is None:
        logger.warning("blood_table not found for horse %s", horse_id)
        return result

    # Collect all td rowspan values and pick the maximum.
    parents: list = []
    max_rs = 0
    for td in table.find_all("td"):
        rs_str = td.get("rowspan", "1") or "1"
        try:
            rs = int(rs_str)
        except ValueError:
            continue
        if rs > max_rs:
            max_rs = rs
            parents = [td]
        elif rs == max_rs:
            parents.append(td)

    if len(parents) < 2:
        logger.warning(
            "Could not identify sire/dam in blood_table for horse %s (max rowspan=%d, found %d parent TDs)",
            horse_id,
            max_rs,
            len(parents),
        )
        return result

    if len(parents) > 2:
        # 3 代血統表では rowspan 最大の親 TD は 2 個（父・母）の想定。
        # 3 つ以上ある場合は HTML 構造が想定外なので silent drop せず警告。
        logger.warning(
            "Found %d parent TDs with max rowspan=%d for horse %s; using first 2 (sire, dam)",
            len(parents),
            max_rs,
            horse_id,
        )

    # First parent = sire, second = dam
    sire_link = parents[0].find("a", href=_HORSE_HREF_RE)
    dam_link = parents[1].find("a", href=_HORSE_HREF_RE)

    if sire_link:
        m = _HORSE_HREF_RE.search(sire_link.get("href", ""))
        if m:
            result.sire_id = m.group(1)
            result.sire_name = sire_link.get_text(strip=True) or None

    if dam_link:
        m = _HORSE_HREF_RE.search(dam_link.get("href", ""))
        if m:
            result.dam_id = m.group(1)
            result.dam_name = dam_link.get_text(strip=True) or None

    return result
