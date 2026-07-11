r"""Parse netkeiba race-list page → list of race IDs for shutuba ingest.

Target URL:
  https://race.netkeiba.com/top/race_list_sub.html?kaisai_date=YYYYMMDD

race_list.html 本体は 2026 時点で Ajax 化されており静的 HTML に race_id を
含まない。race_list_sub.html はその Ajax が読み込む HTML 断片で、構造は
従来の race_list.html と同一 (RaceList_DataItem 内の <a href="...race_id=...">)。

実 HTML 構造（2026 時点で確認済）:
  <dl class="RaceList_DataList">
    <dt>...</dt>
    <dd>
      <ul>
        <li class="RaceList_DataItem">
          <a href="/race/shutuba.html?race_id=202406050901">...</a>
        </li>
      </ul>
    </dd>
  </dl>

race_id は <a href="...?race_id=XXXXXXXXXXXX"> の QueryString から取得する。
race.netkeiba.com の race_list.html は常に ?race_id=... 形式を使用するため、
QS パターンのみを実装する。異なる URL 形式が必要になった場合は別途対応すること。

既存の race_calendar.py (db.netkeiba 結果ページ用) と用途が異なるため別ファイル。

中央のみフィルタ (track codes 01-10) は既存 race_calendar.py と同一ロジック。
地方も取り込む場合は KEIBA_INCLUDE_NAR=1 を設定 (parse_race_ids_from_card_calendar
の include_nar 引数でも制御可)。

If zero race IDs are extracted, raise ParseError so the caller can log and
decide whether to continue or abort.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

from core.logging import get_logger

logger = get_logger(__name__)

# race.netkeiba.com の race_list.html / shutuba.html は ?race_id=XXXXXXXXXXXX 形式のみ使用する。
# パスセグメント形式 (/race/<id>/) は race.netkeiba.com では出現しないため実装しない。
# 将来的に別 URL 形式が必要になった場合は別途パターンを追加すること。
_RACE_ID_QS_RE = re.compile(r"race_id=(\d{12})")

# 開催なし日の race_list_sub.html にも存在する RaceList 系コンテナの class。
# allow_empty=True 時に「開催なし」と「ページ構造変化」を区別するために使う。
_RACELIST_CONTAINER_RE = re.compile(r"^RaceList_")

# JRA 中央競馬場トラックコード (race_id の 5-6 桁目)
_CENTRAL_TRACK_CODES = frozenset({"01", "02", "03", "04", "05", "06", "07", "08", "09", "10"})


class ParseError(Exception):
    pass


def _is_central_race(race_id: str) -> bool:
    """Return True if race_id encodes a JRA central track (codes 01-10)."""
    return len(race_id) >= 6 and race_id[4:6] in _CENTRAL_TRACK_CODES


def parse_race_ids_from_card_calendar(
    html: str, *, include_nar: bool = False, allow_empty: bool = False
) -> list[str]:
    """Extract race IDs from a race.netkeiba.com race_list(_sub) page.

    Searches all <a> tags whose href contains a 12-digit race_id as a
    ?race_id=<id> query parameter.

    By default, only JRA central races (track codes 01-10) are returned.
    Pass include_nar=True to keep NAR / 地方 races as well.

    allow_empty=True の場合、race_id が 0 件でも RaceList 系コンテナが存在する
    (= ページ構造は正常で単に開催がない) なら空リストを返す。開催なし日でも
    race_list_sub.html は空の <div class="RaceList_Box"> を返すため、これで
    「開催なし」と「構造変化」を区別する。

    Raises:
        ParseError: If no race IDs could be found (and the page does not look
            like a valid no-kaisai page when allow_empty=True).
    """
    soup = BeautifulSoup(html, "lxml")
    race_ids: list[str] = []
    seen: set[str] = set()
    nar_skipped = 0

    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        m = _RACE_ID_QS_RE.search(href)
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

    # nar_skipped > 0 なら構造自体は読めている。0 件でも RaceList 系
    # コンテナがあれば「開催なし」ページとみなす。
    if (
        not race_ids
        and allow_empty
        and (nar_skipped or soup.find(class_=_RACELIST_CONTAINER_RE) is not None)
    ):
        logger.info("No central race IDs in race-list HTML — no kaisai on this date")
        return []

    if not race_ids:
        logger.error(
            "No central race IDs found in race-card calendar HTML "
            "— netkeiba page structure may have changed (skipped %d NAR ids)",
            nar_skipped,
        )
        raise ParseError("No race IDs found in race-card calendar HTML")

    if nar_skipped:
        logger.info(
            "Extracted %d central race IDs from race-card calendar (skipped %d NAR)",
            len(race_ids),
            nar_skipped,
        )
    else:
        logger.info("Extracted %d race IDs from race-card calendar", len(race_ids))
    return race_ids
