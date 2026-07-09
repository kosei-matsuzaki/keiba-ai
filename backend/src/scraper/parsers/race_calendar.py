r"""Parse netkeiba race calendar page → list of race IDs.

Target URL:
  https://db.netkeiba.com/race/list/YYYYMMDD/

実 HTML 構造（2026 時点で確認済）:
  <dl class="race_top_data_info fc">
    <dt>1R</dt>
    <dd>
      <a href="/race/202406050901/" title="...">...</a>
      <a href="/race/movie/202406050901">...</a>  ← movie URL は除外
    </dd>
  </dl>

`/race/movie/<id>` は `/race/(\d{12})` パターンに合致しない（`movie/` が間に入るため）
ので自然に除外される。

If zero race IDs are extracted, raise ParseError so the caller can log and
decide whether to continue or abort.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from core.logging import get_logger

logger = get_logger(__name__)

_RACE_ID_RE = re.compile(r"race_id=(\d{12})")
_RACE_PATH_RE = re.compile(r"/race/(\d{12})")

# JRA 中央競馬場のトラックコード (race_id の 5-6 桁目)。
# 11 以上は地方 (NAR/JBA): 30=門別, 35=盛岡, 36=水沢, 42=浦和, 43=船橋, 44=大井,
# 45=川崎, 46=金沢, 47=笠松, 48=名古屋, 50=園田, 51=姫路, 54=高知, 55=佐賀 など。
# 中央予測モデル前提のため calendar 段階で地方を弾く。
# 地方も含めたい場合は環境変数 KEIBA_INCLUDE_NAR=1 を設定する。
_CENTRAL_TRACK_CODES = frozenset({"01", "02", "03", "04", "05", "06", "07", "08", "09", "10"})


class ParseError(Exception):
    pass


def _is_central_race(race_id: str) -> bool:
    """Return True if the race_id encodes a JRA central track (codes 01-10)."""
    return len(race_id) >= 6 and race_id[4:6] in _CENTRAL_TRACK_CODES


def parse_race_ids_from_calendar(
    html: str, *, include_nar: bool = False, raise_if_empty: bool = True
) -> list[str]:
    """Extract race IDs from a kaisai_date calendar page.

    Searches all <a> tags whose href contains a 12-digit race_id in either of:
      - ?race_id=<id> query parameter
      - /race/<id>/ path segment

    By default, only JRA central races are returned (track codes 01-10);
    NAR / 地方 races are filtered out. Pass include_nar=True to keep them.

    Raises:
        ParseError: If no central race IDs could be found (likely a selector change).
    """
    soup = BeautifulSoup(html, "lxml")
    race_ids: list[str] = []
    seen: set[str] = set()
    nar_skipped = 0

    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        m = _RACE_ID_RE.search(href) or _RACE_PATH_RE.search(href)
        if not m:
            continue
        race_id = m.group(1)
        if race_id in seen:
            continue
        seen.add(race_id)
        if not include_nar and not _is_central_race(race_id):
            nar_skipped += 1
            continue
        race_ids.append(race_id)

    if not race_ids:
        # raise_if_empty=False は「開催の無い日 / 直近で db.netkeiba 未アーカイブの日」を
        # 範囲スキャンする想定内ケース用。空でも静かに [] を返す（ERROR を出さない）。
        if not raise_if_empty:
            return []
        logger.error(
            "No central race IDs found in calendar HTML — netkeiba page structure may have changed "
            "(skipped %d NAR ids)",
            nar_skipped,
        )
        raise ParseError("No race IDs found in calendar HTML")

    if nar_skipped:
        logger.info(
            "Extracted %d central race IDs from calendar (skipped %d NAR)",
            len(race_ids),
            nar_skipped,
        )
    else:
        logger.info("Extracted %d race IDs from calendar", len(race_ids))
    return race_ids
