"""Parse netkeiba shutuba (出馬表) page into structured dataclasses.

Target URL:
  https://race.netkeiba.com/race/shutuba.html?race_id=<race_id>

実 HTML 構造（2026 時点で確認済）:
  - 出走馬テーブル: <table class="Shutuba_Table"> または <table class="RaceTable_01">
  - <thead> に列ヘッダ <th>
  - 馬名・騎手・調教師は <a href="..."> リンクから ID を取得
  - 馬体重は「計不」の場合や未公表の場合は None

レースヘッダ（コース・距離・天候）は以下から取得:
  - 距離・馬場: <div class="RaceData01"> のテキスト
  - 天候: 開催前のため公表されていない場合は None

race_result.py と同様に ParsedEntry / 新規 ShutubaEntry dataclass を使うが、
出走前なので finish_position / agari_3f / passing / finish_time / margin は NULL。
戻り値型 ParsedShutuba を定義して統一感を持たせる。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag

from keiba_ai.core.logging import get_logger

logger = get_logger(__name__)

_WEIGHT_RE = re.compile(r"(\d+)\(([+-]?\d+)\)")
_SURFACE_DIST_RE = re.compile(r"(芝|ダ|障)(?:\s*[右左])?(?:\s*[内外])?\s*(\d{3,4})\s*m")
_WEATHER_RE = re.compile(r"天候\s*[:：]\s*([^\s/]+)")

# JRA トラックコード（race_id の 5-6 桁目）→ コース名
_COURSE_CODE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}


class ParseError(Exception):
    pass


@dataclass
class ShutubaEntry:
    """One row in the shutuba (出馬表) table — pre-race snapshot."""

    race_id: str
    horse_id: str
    horse_name: str | None = None
    post_position: int | None = None
    sex: str | None = None
    age: int | None = None
    weight_carried: float | None = None
    jockey_id: str | None = None
    jockey_name: str | None = None
    trainer_id: str | None = None
    trainer_name: str | None = None
    horse_weight: int | None = None
    horse_weight_diff: int | None = None
    odds_win: float | None = None
    popularity: int | None = None
    # 以下はレース前なので常に None
    finish_position: int | None = None
    finish_time: float | None = None
    margin: str | None = None
    agari_3f: float | None = None
    passing: str | None = None


@dataclass
class ParsedShutuba:
    """Parsed shutuba page — race metadata + pre-race entry list."""

    race_id: str
    date: str | None = None
    course: str | None = None
    surface: str | None = None
    distance: int | None = None
    n_runners: int | None = None
    weather: str | None = None  # 開催前は公表されていない場合 None
    entries: list[ShutubaEntry] = field(default_factory=list)


def _extract_id_from_href(href: str, kind: str) -> str | None:
    """Extract entity ID from a netkeiba path.

    Supports:
      - /horse/<id>/                       (馬)
      - /jockey/result/recent/<id>/        (騎手)
      - /trainer/result/recent/<id>/       (調教師)
    """
    m = re.search(rf"/{kind}/(?:[a-z_]+/)*([0-9a-zA-Z]+)", href)
    return m.group(1) if m else None


def _parse_header(soup: BeautifulSoup, result: ParsedShutuba, race_id: str) -> None:
    """Extract race metadata from race_id and page text."""
    if len(race_id) >= 6:
        result.course = _COURSE_CODE_MAP.get(race_id[4:6])

    page_text = soup.get_text(" ", strip=True)

    sd_m = _SURFACE_DIST_RE.search(page_text)
    if sd_m:
        result.surface = sd_m.group(1)
        result.distance = int(sd_m.group(2))

    # 天候は開催前のため公表されていない場合がある（None のまま）
    weather_m = _WEATHER_RE.search(page_text)
    if weather_m:
        result.weather = weather_m.group(1)


def _parse_entries(soup: BeautifulSoup, race_id: str) -> list[ShutubaEntry]:
    # shutuba ページのテーブルクラス名は netkeiba の旧/新ページで異なる
    table = (
        soup.find("table", class_=re.compile(r"Shutuba_Table|RaceTable_01|ShutubaTable"))
        or soup.find("table", class_=re.compile(r"race_table"))
    )
    if table is None:
        logger.error("Shutuba table not found — netkeiba page structure may have changed")
        raise ParseError("Shutuba table not found")

    # ヘッダ行から列番号辞書を構築
    headers: list[str] = []
    thead = table.find("thead")
    if thead:
        headers = [th.get_text(strip=True) for th in thead.find_all("th")]
    if not headers:
        first_tr = table.find("tr")
        if first_tr:
            headers = [th.get_text(strip=True) for th in first_tr.find_all("th")]

    col: dict[str, int] = {name: idx for idx, name in enumerate(headers)}
    if not col:
        logger.warning("No shutuba table headers found; falling back to fixed column positions")

    entries: list[ShutubaEntry] = []
    for tr in table.find_all("tr"):
        if tr.find("th"):
            continue
        tds = tr.find_all("td")
        if len(tds) < 5:
            continue
        try:
            entry = _parse_entry_row(tds, race_id, col)
        except Exception as exc:
            logger.warning("Failed to parse shutuba entry row: %s", exc)
            continue
        if entry is not None:
            entries.append(entry)
    return entries


def _parse_entry_row(
    tds: list[Tag],
    race_id: str,
    col: dict[str, int],
) -> ShutubaEntry | None:
    def text_for(name: str, fallback_idx: int | None = None) -> str:
        idx = col.get(name, fallback_idx)
        if idx is None or idx >= len(tds):
            return ""
        return tds[idx].get_text(strip=True)

    def link_for(name: str, fallback_idx: int | None = None) -> Tag | None:
        idx = col.get(name, fallback_idx)
        if idx is None or idx >= len(tds):
            return None
        return tds[idx].find("a", href=True)

    def to_int(text: str) -> int | None:
        try:
            return int(text)
        except (ValueError, TypeError):
            return None

    def to_float(text: str) -> float | None:
        try:
            return float(text)
        except (ValueError, TypeError):
            return None

    def name_from_link(tag: Tag | None) -> str | None:
        if tag is None:
            return None
        raw = tag.get("title") or tag.get_text(strip=True)
        if not raw:
            return None
        cleaned = raw.strip().replace("　", "").replace(" ", "")
        return cleaned or None

    horse_link = link_for("馬名", 3)
    if horse_link is None:
        return None
    horse_id = _extract_id_from_href(horse_link["href"], "horse")
    if horse_id is None:
        return None

    entry = ShutubaEntry(race_id=race_id, horse_id=horse_id)
    entry.horse_name = name_from_link(horse_link)

    # 馬番: "馬番" 列、なければ fallback_idx=2
    entry.post_position = to_int(text_for("馬番", 2))

    sex_age = text_for("性齢", 4)
    if sex_age:
        entry.sex = sex_age[0] if sex_age[0] in ("牡", "牝", "セ") else None
        try:
            entry.age = int(sex_age[1:])
        except (ValueError, IndexError):
            pass

    entry.weight_carried = to_float(text_for("斤量", 5))

    jockey_link = link_for("騎手", 6)
    if jockey_link:
        entry.jockey_id = _extract_id_from_href(jockey_link["href"], "jockey")
        entry.jockey_name = name_from_link(jockey_link)

    trainer_link = link_for("調教師")
    if trainer_link:
        entry.trainer_id = _extract_id_from_href(trainer_link["href"], "trainer")
        entry.trainer_name = name_from_link(trainer_link)

    # 馬体重: "計不" や未公表の場合は None
    hw_text = text_for("馬体重")
    m = _WEIGHT_RE.search(hw_text)
    if m:
        entry.horse_weight = int(m.group(1))
        entry.horse_weight_diff = int(m.group(2))
    # "計不" や空欄は None のまま（新馬戦など体重未計測）

    entry.odds_win = to_float(text_for("単勝"))
    entry.popularity = to_int(text_for("人気"))

    return entry


def parse_shutuba(html: str, race_id: str) -> ParsedShutuba:
    """Parse a shutuba page into ParsedShutuba.

    Raises:
        ParseError: If the shutuba table is missing entirely.
    """
    soup = BeautifulSoup(html, "lxml")
    result = ParsedShutuba(race_id=race_id)

    _parse_header(soup, result, race_id)
    result.entries = _parse_entries(soup, race_id)
    result.n_runners = len(result.entries)

    return result
