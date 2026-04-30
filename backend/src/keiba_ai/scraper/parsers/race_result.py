"""Parse netkeiba race result page into structured dataclasses.

Target URL:
  https://db.netkeiba.com/race/<race_id>/

実 HTML 構造（2026 時点で確認済）:
  - 出走馬テーブル: <table class="race_table_01 nk_tb_common">
  - <thead> は無く、最初の <tr> の <th> が列ヘッダ
  - premium 列（タイム指数 / 調教タイム / 厩舎コメント / 備考）が DOM に存在
    （display:none 含む）。固定インデックスではなくヘッダ名 → 列番号の辞書で参照
  - <diary_snap_cut> という非標準タグが <td> を囲む箇所があるが BS4 は
    descendants 検索なので透過に拾える
  - jockey/trainer のリンク URL は /jockey/result/recent/<id>/ 形式
  - course は race_id の 5-6 桁目（JRA トラックコード）から導出可能

レースヘッダ（コース・距離・天候・馬場）は class 名が変動しやすいので、
ページ全体のテキストから正規表現で抽出する。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag

from keiba_ai.core.logging import get_logger
from keiba_ai.scraper.parsers.payout import parse_payout

logger = get_logger(__name__)

_TIME_RE = re.compile(r"(\d+):(\d+)\.(\d+)")
_WEIGHT_RE = re.compile(r"(\d+)\(([+-]?\d+)\)")

# 実 HTML ヘッダ形式（2026 時点）:
#   "ダ右1200m / 天候:晴 / 馬場:良"
#   "芝右 外1600m / 天候:晴 / 馬場:良"
#   "障芝1200m" 等の障害競走
# 旧フィクスチャ形式:
#   "芝1600m / 天候:晴 / 馬場:良"
# surface の直後にコース方向（右/左）、内外（内/外）、距離 が続く
_SURFACE_DIST_RE = re.compile(r"(芝|ダ|障)(?:\s*[右左])?(?:\s*[内外])?\s*(\d{3,4})\s*m")
_WEATHER_RE = re.compile(r"天候\s*[:：]\s*([^\s/]+)")
# 馬場状態は「馬場 : 良」(旧フィクスチャ形式) または surface に続く形式（新）のどちらか
# 新形式は芝→「芝 : 良」、ダート→「ダート : 良」、障害→「障 : 良」のように
# surface 表記が異なる（"ダ" 1 文字ではなく "ダート" のフルスペル）。
# 順序重要: 長い "ダート" を "ダ" より先に書かないと "ダ" だけマッチして失敗する
_TRACK_OLD_RE = re.compile(r"馬場\s*[:：]\s*([^\s/]+)")
_TRACK_NEW_RE = re.compile(r"(?:ダート|芝|ダ|障)\s*[:：]\s*([^\s/]+)")

# JRA トラックコード（race_id の 5-6 桁目）→ コース名
_COURSE_CODE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}


class ParseError(Exception):
    pass


@dataclass
class ParsedEntry:
    race_id: str
    horse_id: str
    post_position: int | None = None
    jockey_id: str | None = None
    trainer_id: str | None = None
    weight_carried: float | None = None
    age: int | None = None
    sex: str | None = None
    horse_weight: int | None = None
    horse_weight_diff: int | None = None
    odds_win: float | None = None
    popularity: int | None = None
    finish_position: int | None = None
    finish_time: float | None = None
    margin: str | None = None


@dataclass
class ParsedRaceResult:
    race_id: str
    date: str | None = None
    course: str | None = None
    surface: str | None = None
    distance: int | None = None
    weather: str | None = None
    track_condition: str | None = None
    race_class: str | None = None
    n_runners: int | None = None
    payout_win: int | None = None
    payout_place: str | None = None  # JSON string
    entries: list[ParsedEntry] = field(default_factory=list)


def _extract_id_from_href(href: str, kind: str) -> str | None:
    """Extract entity ID from a netkeiba path.

    Supports both:
      - /horse/<id>/                          (馬は直接)
      - /jockey/result/recent/<id>/           (騎手・調教師は result/recent 配下)
      - /trainer/result/recent/<id>/

    `/<kind>/` の後にある最初の連続数字を ID として返す。
    """
    m = re.search(rf"/{kind}/(?:[a-z_]+/)*(\d+)", href)
    return m.group(1) if m else None


def _parse_time_to_seconds(text: str) -> float | None:
    m = _TIME_RE.search(text)
    if not m:
        return None
    minutes, secs, tenths = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return minutes * 60 + secs + tenths / 10


def _parse_header(soup: BeautifulSoup, result: ParsedRaceResult, race_id: str) -> None:
    """Extract race metadata from race_id and page-wide text scan."""
    # Course は race_id の 5-6 桁目（JRA トラックコード）から導出
    if len(race_id) >= 6:
        result.course = _COURSE_CODE_MAP.get(race_id[4:6])

    page_text = soup.get_text(" ", strip=True)

    sd_m = _SURFACE_DIST_RE.search(page_text)
    if sd_m:
        result.surface = sd_m.group(1)
        result.distance = int(sd_m.group(2))

    weather_m = _WEATHER_RE.search(page_text)
    if weather_m:
        result.weather = weather_m.group(1)

    # 旧形式（馬場:良）優先、無ければ新形式（芝 : 良）
    track_m = _TRACK_OLD_RE.search(page_text) or _TRACK_NEW_RE.search(page_text)
    if track_m:
        result.track_condition = track_m.group(1)


def _parse_entries(soup: BeautifulSoup, race_id: str) -> list[ParsedEntry]:
    table = soup.find("table", class_=re.compile(r"race_table_01|RaceTable"))
    if table is None:
        logger.error("Race result table not found — netkeiba page structure may have changed")
        raise ParseError("Race result table not found")

    # ヘッダ行を取得（<thead> が無く最初の <tr> の <th> が列名というケースが多い）
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
        logger.warning("No table headers found; falling back to fixed column positions")

    entries: list[ParsedEntry] = []
    for tr in table.find_all("tr"):
        if tr.find("th"):  # skip header row
            continue
        tds = tr.find_all("td")
        if len(tds) < 5:
            continue
        try:
            entry = _parse_entry_row(tds, race_id, col)
        except Exception as exc:
            logger.warning("Failed to parse entry row: %s", exc)
            continue
        if entry is not None:
            entries.append(entry)
    return entries


def _parse_entry_row(
    tds: list[Tag],
    race_id: str,
    col: dict[str, int],
) -> ParsedEntry | None:
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

    horse_link = link_for("馬名", 3)
    if horse_link is None:
        return None
    horse_id = _extract_id_from_href(horse_link["href"], "horse")
    if horse_id is None:
        return None

    entry = ParsedEntry(race_id=race_id, horse_id=horse_id)
    entry.finish_position = to_int(text_for("着順", 0))
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

    entry.finish_time = _parse_time_to_seconds(text_for("タイム", 7))
    entry.margin = text_for("着差", 8) or None

    hw_text = text_for("馬体重")
    m = _WEIGHT_RE.search(hw_text)
    if m:
        entry.horse_weight = int(m.group(1))
        entry.horse_weight_diff = int(m.group(2))

    entry.odds_win = to_float(text_for("単勝"))
    entry.popularity = to_int(text_for("人気"))

    trainer_link = link_for("調教師")
    if trainer_link:
        entry.trainer_id = _extract_id_from_href(trainer_link["href"], "trainer")

    return entry


def parse_race_result(html: str, race_id: str) -> ParsedRaceResult:
    """Parse a race result page into ParsedRaceResult.

    Raises:
        ParseError: If the results table is missing entirely.
    """
    soup = BeautifulSoup(html, "lxml")
    result = ParsedRaceResult(race_id=race_id)

    _parse_header(soup, result, race_id)
    result.entries = _parse_entries(soup, race_id)
    result.n_runners = len(result.entries)

    payout_win, payout_place = parse_payout(html)
    result.payout_win = payout_win
    result.payout_place = json.dumps(payout_place, ensure_ascii=False) if payout_place else None

    return result
