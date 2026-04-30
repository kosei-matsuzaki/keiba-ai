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
    """Parse pedigree page. Identifies sire and dam from blood_table tr/rowspan layout.

    blood_table の典型レイアウト:
        行 0:           父(rowspan=N)  | 父父(rowspan=N/2) | 父父父...
        行 1:                                                | 父父母...
        ...
        行 N/2 - 1:                     | 父母              | ...
        行 N/2:         母(rowspan=N)  | 母父              | ...
        ...

    「行 0 の最初の <td> = 父」「その rowspan 番目の <tr> の最初の <td> = 母」で
    3 代血統表（父 rowspan=4）でも 5 代（rowspan=16）でも確実に識別できる。
    """
    soup = BeautifulSoup(html, "lxml")
    result = ParsedPedigree(horse_id=horse_id)

    table = soup.find("table", class_=re.compile(r"blood_table"))
    if table is None:
        logger.warning("blood_table not found for horse %s", horse_id)
        return result

    # blood_table は <tr> + <td rowspan> による世代構造。レイアウトの典型:
    #
    #   行 0:    父(rowspan=N) | 父父(rowspan=N/2) | 父父父...
    #   行 1:                                       | 父父母...
    #   行 N/2-1:                | 父母              | ...
    #   行 N/2:  母(rowspan=N) | 母父              | ...
    #   ...
    #
    # 「最初の <tr> の最初の <td> = 父」「その rowspan 番目の <tr> の最初の
    # <td> = 母」というロジックで、3 代 (父 rowspan=4) でも 5 代 (rowspan=16) でも
    # 確実に父・母を識別できる。
    trs = table.find_all("tr")
    if not trs:
        logger.warning("blood_table has no rows for horse %s", horse_id)
        return result

    # 父 TD: 行 0 の最初の TD
    sire_td = trs[0].find("td")
    if sire_td is None:
        logger.warning("first <tr> has no <td> for horse %s", horse_id)
        return result

    # 父の rowspan を取って母の行 index を決定
    rs_str = sire_td.get("rowspan", "1") or "1"
    try:
        sire_rowspan = int(rs_str)
    except ValueError:
        sire_rowspan = 1

    # 母 TD: 行 sire_rowspan の最初の TD
    if sire_rowspan >= len(trs):
        logger.warning(
            "sire rowspan=%d exceeds total rows=%d for horse %s; cannot locate dam",
            sire_rowspan,
            len(trs),
            horse_id,
        )
        dam_td = None
    else:
        dam_td = trs[sire_rowspan].find("td")

    # 父・母の <a href="/horse/<id>/"> リンクを抽出
    sire_link = sire_td.find("a", href=_HORSE_HREF_RE)
    dam_link = dam_td.find("a", href=_HORSE_HREF_RE) if dam_td is not None else None

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
