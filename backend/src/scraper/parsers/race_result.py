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

import contextlib
import json
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag

from core.logging import get_logger
from scraper.parsers.common import (
    COURSE_CODE_MAP,
    SURFACE_DIST_RE,
    WEATHER_RE,
    WEIGHT_RE,
    extract_id_from_href,
    normalize_race_class,
)
from scraper.parsers.payout import parse_payout

logger = get_logger(__name__)

_TIME_RE = re.compile(r"(\d+):(\d+)\.(\d+)")

# 馬場状態は「馬場 : 良」(旧フィクスチャ形式) または surface に続く形式（新）のどちらか
# 新形式は芝→「芝 : 良」、ダート→「ダート : 良」、障害→「障 : 良」のように
# surface 表記が異なる（"ダ" 1 文字ではなく "ダート" のフルスペル）。
# 順序重要: 長い "ダート" を "ダ" より先に書かないと "ダ" だけマッチして失敗する
_TRACK_OLD_RE = re.compile(r"馬場\s*[:：]\s*([^\s/]+)")
_TRACK_NEW_RE = re.compile(r"(?:ダート|芝|ダ|障)\s*[:：]\s*([^\s/]+)")

# レースクラス検出 — class="RaceData02" (旧形式) 用。
# word boundary で "TOP" の "OP" 部分への誤マッチを防ぐ。
# `L` は単独文字なので word boundary を付けて "1600L" 等の偶発マッチを防ぐ。
# Roman numeral 変種 GIII/GII/GI を長い順に並べて、貪欲に長いものから捕る。
_GRADE_RE_LEGACY = re.compile(
    r"(GⅢ|GIII|G3|GⅡ|GII|G2|GⅠ|GI(?![IV])|G1|Listed|\bL\b|\bOP\b|重賞)"
)

class ParseError(Exception):
    pass


@dataclass
class ParsedEntry:
    race_id: str
    horse_id: str
    horse_name: str | None = None
    post_position: int | None = None
    jockey_id: str | None = None
    jockey_name: str | None = None
    trainer_id: str | None = None
    trainer_name: str | None = None
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
    agari_3f: float | None = None
    passing: str | None = None


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
    name: str | None = None
    n_runners: int | None = None
    payout_win: int | None = None
    payout_place: str | None = None  # JSON string
    entries: list[ParsedEntry] = field(default_factory=list)


def _parse_time_to_seconds(text: str) -> float | None:
    m = _TIME_RE.search(text)
    if not m:
        return None
    minutes, secs, tenths = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return minutes * 60 + secs + tenths / 10


def _parse_header(soup: BeautifulSoup, result: ParsedRaceResult, race_id: str) -> None:
    """Extract race metadata from race_id and page structure.

    優先的に class="data_intro" (db.netkeiba 現行形式) を参照し、
    存在しなければ旧 class="RaceData02" / "RaceName" 形式にフォールバックする。
    ページ全体テキストへの race_class フォールバックは廃止（"TOP" 誤マッチ防止）。
    """
    # Course は race_id の 5-6 桁目（JRA トラックコード）から導出
    if len(race_id) >= 6:
        result.course = COURSE_CODE_MAP.get(race_id[4:6])

    page_text = soup.get_text(" ", strip=True)

    sd_m = SURFACE_DIST_RE.search(page_text)
    if sd_m:
        result.surface = sd_m.group(1)
        result.distance = int(sd_m.group(2))

    weather_m = WEATHER_RE.search(page_text)
    if weather_m:
        result.weather = weather_m.group(1)

    # 旧形式（馬場:良）優先、無ければ新形式（芝 : 良）
    track_m = _TRACK_OLD_RE.search(page_text) or _TRACK_NEW_RE.search(page_text)
    if track_m:
        result.track_condition = track_m.group(1)

    # ── レース名 & race_class ────────────────────────────────────────────────
    # 現行 db.netkeiba 形式: class="data_intro" 内の <h1> と class="smalltxt" を使う。
    # 旧形式 (フィクスチャ等): class="RaceData02" / "RaceName" を使う。
    data_intro = soup.find(class_="data_intro")
    if data_intro:
        # レース名: <h1> の text（HTMLコメントは get_text で自動除外）
        h1 = data_intro.find("h1")
        if h1:
            result.name = h1.get_text(strip=True) or None

        # race_class: h1 テキスト → smalltxt の順で検索
        race_class: str | None = None

        if h1:
            h1_text = h1.get_text(strip=True)
            race_class = normalize_race_class(h1_text)

        if race_class is None:
            smalltxt = data_intro.find(class_="smalltxt")
            if smalltxt:
                st_text = smalltxt.get_text(strip=True)
                race_class = normalize_race_class(st_text)

        result.race_class = race_class
    else:
        # 旧形式フォールバック: RaceData02 の span 群 → RaceName
        race_data_02 = soup.find(class_="RaceData02")
        candidates: list[str] = []
        if race_data_02:
            candidates = [s.get_text(strip=True) for s in race_data_02.find_all("span")]
        race_name_el = soup.find(class_="RaceName")
        if race_name_el:
            candidates.append(race_name_el.get_text(" ", strip=True))
            if result.name is None:
                result.name = race_name_el.get_text(strip=True) or None

        for text in candidates:
            m = _GRADE_RE_LEGACY.search(text)
            if m:
                result.race_class = normalize_race_class(m.group(1))
                break


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

    def name_from_link(tag: Tag | None) -> str | None:
        """Extract display name from an <a> tag via title attr or inner text."""
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
    horse_id = extract_id_from_href(horse_link["href"], "horse")
    if horse_id is None:
        return None

    entry = ParsedEntry(race_id=race_id, horse_id=horse_id)
    entry.horse_name = name_from_link(horse_link)

    entry.finish_position = to_int(text_for("着順", 0))
    entry.post_position = to_int(text_for("馬番", 2))

    sex_age = text_for("性齢", 4)
    if sex_age:
        entry.sex = sex_age[0] if sex_age[0] in ("牡", "牝", "セ") else None
        with contextlib.suppress(ValueError, IndexError):
            entry.age = int(sex_age[1:])

    entry.weight_carried = to_float(text_for("斤量", 5))

    jockey_link = link_for("騎手", 6)
    if jockey_link:
        entry.jockey_id = extract_id_from_href(jockey_link["href"], "jockey")
        entry.jockey_name = name_from_link(jockey_link)

    entry.finish_time = _parse_time_to_seconds(text_for("タイム", 7))
    entry.margin = text_for("着差", 8) or None

    hw_text = text_for("馬体重")
    m = WEIGHT_RE.search(hw_text)
    if m:
        entry.horse_weight = int(m.group(1))
        entry.horse_weight_diff = int(m.group(2))

    entry.odds_win = to_float(text_for("単勝"))
    entry.popularity = to_int(text_for("人気"))

    trainer_link = link_for("調教師")
    if trainer_link:
        entry.trainer_id = extract_id_from_href(trainer_link["href"], "trainer")
        entry.trainer_name = name_from_link(trainer_link)

    # 上り3F（<span>38.5</span> 等、innerTextをfloat変換）
    agari_raw = text_for("上り")
    entry.agari_3f = to_float(agari_raw)

    # 通過（"2-2" 等の生文字列）
    passing_raw = text_for("通過").strip()
    entry.passing = passing_raw or None

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
