"""Parse payout section from netkeiba result page.

M2 scope: tan-sho (単勝, win) and fuku-sho (複勝, place) only.
All other bet types (馬連, 三連単, etc.) are ignored in this milestone.

Assumed HTML structure (to be verified against real pages in M2 manual QA):
  <table class="pay_block">
    <tr>
      <th>単勝</th>
      <td>5</td>         <!-- horse number -->
      <td>1,230 円</td>  <!-- payout -->
    </tr>
    <tr>
      <th>複勝</th>
      <td>5</td>
      <td>320 円</td>
    </tr>
    ...
  </table>

Returns:
  payout_win: int | None
  payout_place: dict[str, int] | None  ({"1": 320, "2": 240, "3": 180})
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from keiba_ai.core.logging import get_logger

logger = get_logger(__name__)

_AMOUNT_RE = re.compile(r"([\d,]+)\s*円")
_DIGITS_RE = re.compile(r"[\d,]+")


def _parse_yen_amount(text: str) -> int | None:
    """Extract a yen amount from text that contains '円'.

    Only matches cells that explicitly contain '円' so that horse-number
    cells (plain integers like '5') are ignored.
    """
    m = _AMOUNT_RE.search(text)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except ValueError:
            return None
    return None


def parse_payout(html: str) -> tuple[int | None, dict[str, int] | None]:
    """Extract win and place payouts.

    Returns:
        (payout_win, payout_place) where payout_place is a dict keyed by
        finish position strings ("1", "2", "3").
    """
    soup = BeautifulSoup(html, "lxml")
    payout_win: int | None = None
    payout_place: dict[str, int] = {}
    place_rank = 1

    pay_table = soup.find("table", class_=re.compile(r"pay_block|Payout"))
    if pay_table is None:
        pay_table = soup.find("div", class_=re.compile(r"pay_block|Payout"))

    if pay_table is None:
        logger.warning("No payout table found — returning None payouts")
        return None, None

    for row in pay_table.find_all("tr"):
        th = row.find("th")
        if th is None:
            continue
        label = th.get_text(strip=True)
        tds = row.find_all("td")

        if "単勝" in label:
            for td in tds:
                amount = _parse_yen_amount(td.get_text())
                if amount is not None:
                    payout_win = amount
                    break
        elif "複勝" in label:
            for td in tds:
                amount = _parse_yen_amount(td.get_text())
                if amount is not None:
                    payout_place[str(place_rank)] = amount
                    place_rank += 1
                    if place_rank > 3:
                        break

    return payout_win, payout_place if payout_place else None
