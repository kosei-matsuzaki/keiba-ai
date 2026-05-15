"""Parse netkeiba shutuba (出馬表) page into structured dataclasses.

Target URL:
  https://race.netkeiba.com/race/shutuba.html?race_id=<race_id>

実 HTML 構造（2024-2026 年実 race.netkeiba.com ページで確認済）:

テーブル:
  <table class="Shutuba_Table RaceTable01 ShutubaTable">
    <thead>
      <tr>
        <th class="Waku">枠</th>
        <th class="Umaban">馬番</th>
        <th class="CheckMark">印</th>         ← 合成 fixture には存在しない
        <th class="HorseInfo">馬名</th>
        <th class="Barei">性齢</th>
        <th class="Dredging">斤量</th>
        <th class="Jockey">騎手</th>
        <th class="Trainer">厩舎</th>          ← 合成 fixture では "調教師"
        <th class="Weight">馬体重(増減)</th>   ← 合成 fixture では "馬体重"
        <th class="Popular">更新</th>          ← 合成 fixture では "単勝"（オッズ列）
        <th class="Popular Popular_Ninki">人気</th>
        ...（お気に入り／メモ系は無視）
      </tr>
    </thead>
    <tbody>
      <tr>  <!-- 1頭分 -->
        <td class="Waku1">1</td>
        <td class="Umaban1">1</td>
        <td class="CheckMark">...</td>
        <td class="HorseInfo"><a href="https://db.netkeiba.com/horse/XXXXXXXXXX">馬名</a></td>
        <td class="Barei">牡2</td>
        <td class="">56.0</td>
        <td class="Jockey"><a href=".../jockey/result/recent/XXXXX/" title="騎手名">...</a></td>
        <td class="Trainer">
          <span class="Label1">美浦</span>
          <a href=".../trainer/result/recent/XXXXX/" title="調教師名">...</a>
        </td>
        <td class="Weight">484 (0)</td>        ← 括弧前後にスペースあり
        <td class="Txt_R Popular">---.-</td>   ← オッズ未公開時は "---.-"
        <td class="Popular Popular_Ninki">**</td>  ← 人気未公開時は "**"
      </tr>
    </tbody>
  </table>

レースヘッダ:
  - <title>: "レース名(G1) 出馬表 | 2024年12月28日 会場11R ..." — 日付・グレード取得元
  - <h1 class="RaceName">: 短縮レース名（スパン内のアイコンは get_text で除去）
  - <div class="RaceData01">: "HH:MM発走 / 芝2000m (右 A) / 天候:晴 / 馬場:良"
  - <div class="RaceData02">: "5回 中山 9日目 サラ系２歳 オープン (国際) 牡・牝(指) 馬齢 18頭 ..."

互換 fallback（合成フィクスチャ形式）:
  合成フィクスチャは以下の列名・構造を持つ（クラス属性なし）:
    [枠番, 馬番, 馬名, 性齢, 斤量, 騎手, 調教師, 馬体重, 単勝, 人気]
  _build_col_aliases() で実 HTML の列名を合成 fixture の列名に正規化するため、
  _parse_entry_row() は列名ベースルックアップのみで両形式に対応する。

race_result.py と同様に ParsedEntry / 新規 ShutubaEntry dataclass を使うが、
出走前なので finish_position / agari_3f / passing / finish_time / margin は NULL。
戻り値型 ParsedShutuba を定義して統一感を持たせる。
"""

from __future__ import annotations

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

logger = get_logger(__name__)

_DATE_RE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")

# race.netkeiba.com の出馬表ページの race_class/name 判定用。
# db.netkeiba とは異なり RaceData01 / RaceName が使われることが多い。
_GRADE_RE = re.compile(r"(GⅠ|GⅡ|GⅢ|G[1-3]|Listed|\(L\)|重賞|未勝利|新馬|[1-3]勝クラス|オープン|\bOP\b)")

# 実 HTML のヘッダ列名 → 合成 fixture と共通の正規化列名へのエイリアス。
# 実 HTML では "厩舎"/"馬体重(増減)"/"更新" という名前が使われるが、
# 合成 fixture では "調教師"/"馬体重"/"単勝" という名前が使われる。
# _build_col_aliases() でこれらを正規化して _parse_entry_row() が両形式に対応できるようにする。
_COL_ALIASES: dict[str, str] = {
    "厩舎": "調教師",          # 実 HTML: 厩舎 = 合成: 調教師
    "馬体重(増減)": "馬体重",  # 実 HTML: 馬体重(増減) = 合成: 馬体重
    "更新": "単勝",            # 実 HTML: 更新(オッズ列) = 合成: 単勝
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
    race_class: str | None = None
    name: str | None = None
    entries: list[ShutubaEntry] = field(default_factory=list)


def _build_col_aliases(col: dict[str, int]) -> dict[str, int]:
    """実 HTML のヘッダ列名を合成 fixture と共通の列名に正規化して返す。

    元の col dict を変更せず、エイリアスを追加した新しい dict を返す。
    エイリアス先が既に存在する場合は追加しない（合成 fixture では不要なため）。
    """
    result = dict(col)
    for real_name, canonical in _COL_ALIASES.items():
        if real_name in result and canonical not in result:
            result[canonical] = result[real_name]
    return result


def _parse_header(soup: BeautifulSoup, result: ParsedShutuba, race_id: str) -> None:
    """Extract race metadata from race_id and page elements.

    実 HTML (race.netkeiba.com) と合成フィクスチャの両方に対応する。
    フィールド取得の優先順位:

    date:
      <title> タグの "YYYY年MM月DD日" パターン（実 HTML に存在）。
      合成フィクスチャの title には日付がないため None のまま。
      ingest_shutuba.py は --date CLI 引数で上書きするため、実運用への影響はない。

    race_class:
      1. <title> タグから normalize_race_class()（実 HTML の G1/G2/G3 に有効）
      2. <div class="RaceData02"> テキスト（合成 fixture の "G1" など）
      3. <div class="RaceData01"> テキスト
      4. <span class="smalltxt"> テキスト（db.netkeiba 互換）
      5. レース名から推測

    surface / distance / weather:
      ページ全体テキストからの正規表現マッチ（両形式で動作）。

    course:
      race_id の 5-6 桁目から JRA 開催場コードで解決。

    name (レース名):
      <h1 class="RaceName"> → <span class="RaceTitle"> → <span class="race_name"> → data_intro > h1。
    """
    if len(race_id) >= 6:
        result.course = COURSE_CODE_MAP.get(race_id[4:6])

    # --- 日付 ---
    title_el = soup.find("title")
    if title_el:
        title_text = title_el.get_text(strip=True)
        m = _DATE_RE.search(title_text)
        if m:
            y, mo, d = m.groups()
            result.date = f"{y}-{int(mo):02d}-{int(d):02d}"

    # --- surface / distance / weather ---
    # RaceData01 を優先して取得（ページ全体への fallback より誤マッチが少ない）。
    racedata01 = soup.find(class_="RaceData01")
    search_area = racedata01.get_text(" ", strip=True) if racedata01 else soup.get_text(" ", strip=True)

    sd_m = SURFACE_DIST_RE.search(search_area)
    if sd_m:
        result.surface = sd_m.group(1)
        result.distance = int(sd_m.group(2))

    # 天候は開催前のため公表されていない場合がある（None のまま）
    weather_m = WEATHER_RE.search(search_area)
    if weather_m:
        result.weather = weather_m.group(1)

    # --- レース名 ---
    for cls in ("RaceName", "RaceTitle", "race_name"):
        el = soup.find(class_=cls)
        if el:
            result.name = el.get_text(strip=True) or None
            break
    if result.name is None:
        data_intro = soup.find(class_="data_intro")
        if data_intro:
            h1 = data_intro.find("h1")
            if h1:
                result.name = h1.get_text(strip=True) or None

    # --- race_class ---
    # 実 HTML では <title> に "(G1)" 等が入るため title-first が最も信頼性が高い。
    # 合成フィクスチャは title に grade を含まないため、後続の element-based lookup に fallback する。
    race_class: str | None = None

    if title_el:
        race_class = normalize_race_class(title_el.get_text(strip=True))

    if race_class is None:
        for cls in ("RaceData02", "RaceData01"):
            el = soup.find(class_=cls)
            if el:
                text = el.get_text(strip=True)
                race_class = normalize_race_class(text)
                if race_class is not None:
                    break

    if race_class is None:
        smalltxt = soup.find(class_="smalltxt")
        if smalltxt:
            race_class = normalize_race_class(smalltxt.get_text(strip=True))

    if race_class is None and result.name:
        race_class = normalize_race_class(result.name)

    result.race_class = race_class


def _parse_entries(soup: BeautifulSoup, race_id: str) -> list[ShutubaEntry]:
    # shutuba ページのテーブルクラス名は netkeiba の旧/新ページで異なる
    table = (
        soup.find("table", class_=re.compile(r"Shutuba_Table|RaceTable_01|ShutubaTable"))
        or soup.find("table", class_=re.compile(r"race_table"))
    )
    if table is None:
        logger.error("Shutuba table not found — netkeiba page structure may have changed")
        raise ParseError("Shutuba table not found")

    # ヘッダ行から列番号辞書を構築（テキストベース）
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

    # 実 HTML の列名（厩舎/馬体重(増減)/更新）を合成 fixture 共通の列名に正規化する
    col = _build_col_aliases(col)

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
    horse_id = extract_id_from_href(horse_link["href"], "horse")
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
        entry.jockey_id = extract_id_from_href(jockey_link["href"], "jockey")
        entry.jockey_name = name_from_link(jockey_link)

    # "調教師" は合成 fixture の列名。実 HTML では "厩舎" が使われるが、
    # _build_col_aliases() で "調教師" にエイリアスされている。
    trainer_link = link_for("調教師")
    if trainer_link:
        entry.trainer_id = extract_id_from_href(trainer_link["href"], "trainer")
        entry.trainer_name = name_from_link(trainer_link)

    # 馬体重: "計不" や未公表の場合は None。
    # "馬体重" は合成 fixture の列名。実 HTML では "馬体重(増減)" が使われるが、
    # _build_col_aliases() で "馬体重" にエイリアスされている。
    # 実 HTML では "484 (0)" のようにスペースあり -> WEIGHT_RE の \s* で吸収。
    hw_text = text_for("馬体重")
    m = WEIGHT_RE.search(hw_text)
    if m:
        entry.horse_weight = int(m.group(1))
        entry.horse_weight_diff = int(m.group(2))
    # "計不" や空欄は None のまま（新馬戦など体重未計測）

    # 単勝オッズ: 実 HTML では発走前に "---.-" と表示され to_float が None を返す（正常）。
    # "単勝" は合成 fixture の列名。実 HTML では "更新" が使われるが、
    # _build_col_aliases() で "単勝" にエイリアスされている。
    entry.odds_win = to_float(text_for("単勝"))
    # 人気: 実 HTML では発走前に "**" と表示され to_int が None を返す（正常）。
    entry.popularity = to_int(text_for("人気"))

    return entry


def extract_race_date_from_shutuba_html(html: str) -> str | None:
    """Shutuba HTML から開催日 (YYYY-MM-DD) のみを抽出。

    <title> タグの "YYYY年MM月DD日" パターンを使う。
    タイトルに日付が含まれない合成フィクスチャや予期しない形式の場合は None を返す。
    """
    soup = BeautifulSoup(html, "html.parser")
    title_el = soup.find("title")
    if not title_el:
        return None
    m = _DATE_RE.search(title_el.get_text(strip=True))
    if not m:
        return None
    y, mo, d = m.groups()
    return f"{y}-{int(mo):02d}-{int(d):02d}"


def parse_shutuba(html: str, race_id: str) -> ParsedShutuba:
    """Parse a shutuba page into ParsedShutuba.

    実 race.netkeiba.com HTML と合成フィクスチャ HTML の両方に対応する。

    実 HTML 対応のポイント:
      - 列名エイリアス: 厩舎→調教師, 馬体重(増減)→馬体重, 更新→単勝
      - 馬体重の "484 (0)" 形式（スペースあり括弧）を WEIGHT_RE で処理
      - <title> タグから日付 (YYYY-MM-DD) と race_class (G1/G2/G3 等) を抽出
      - 単勝オッズ "---.-" と人気 "**" は None として扱う（発走前未公開）

    互換 fallback:
      合成フィクスチャ（列名が "調教師" / "馬体重" / "単勝"）はエイリアス不要のため、
      既存の列名ベースルックアップがそのまま機能する。

    Raises:
        ParseError: If the shutuba table is missing entirely.
    """
    soup = BeautifulSoup(html, "html.parser")
    result = ParsedShutuba(race_id=race_id)

    _parse_header(soup, result, race_id)
    result.entries = _parse_entries(soup, race_id)
    result.n_runners = len(result.entries)

    return result
