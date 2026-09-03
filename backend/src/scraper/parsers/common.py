"""parsers モジュールで共有する正規表現・定数・ヘルパ関数。

race_result.py と shutuba.py が同じパターンを別実装しないよう、ここに集約する。
"""

from __future__ import annotations

import contextlib
import re

from bs4 import Tag

from core.logging import get_logger

logger = get_logger(__name__)

# ── 共通正規表現 ──────────────────────────────────────────────────────────────

# 馬体重 "484 (0)" / "478 (+2)" / "494(-2)" — race.netkeiba は括弧前にスペース、
# db.netkeiba はスペース無しなので \s* で両対応する。
WEIGHT_RE = re.compile(r"(\d+)\s*\(([+-]?\d+)\)")

# レースヘッダ "ダ右1200m / 天候:晴 / 馬場:良" — surface 直後にコース形状が
# 続く。コース略号は db: "芝/ダ/障", race: 同等。
#
# **コース形状は 1 文字とは限らない。** 実データにあるもの:
#   "ダ右1200m"        ふつう
#   "芝直線1000m"      新潟の直線 (「直線」は 2 文字)
#   "障芝 外-内2850m"  障害の襷コース (外回り → 内回り)
#   "障芝 ダート3000m" 障害で芝とダートの両方を走るコース
#   "芝右 内2周3600m"  中山のステイヤーズS (コースを 2 周する)
# 以前は `[右左]?[内外]?` の 2 文字までしか許さず、この 2 つが**丸ごと解析に
# 失敗して surface='' / distance=0 のまま保存されていた** (964 レース)。
# 形状に出てくる文字を並べ、間の空白・区切りごと読み飛ばす。
#
# 障害は "障芝" のように surface が後ろに付くので、`障` では距離まで届かず、
# 続く `芝` / `ダ` から改めてマッチする (= 障害でも芝 / ダートが記録される)。
# 芝とダートの両方を走る障害コース ("障芝 ダート3000m") は **先に出る方 (芝)** を
# 採る。DB に「芝とダート」の区分が無く、他の障害レースも 障芝 → 芝 で入っているため。
# `障` の後ろに 芝 / ダ が続くときは、そちらを surface として拾う (負の先読み)。
# 「障芝 ダート3000m」を 障 で拾ってしまうと、既存の障害レース (障芝 → 芝) と
# 区分が食い違い、特徴量の surface に見たことのない水準が増えてしまう。
SURFACE_DIST_RE = re.compile(
    r"(芝|ダ|障(?!\s*[芝ダ]))"
    r"(?:\s*[右左直線内外襷回りコース・\-−ー]|\s*ダート|\s*芝|\s*\d+周)*"
    r"\s*(\d{3,4})\s*m"
)

# 天候表記 "天候:晴" / "天候 ： 雨"
WEATHER_RE = re.compile(r"天候\s*[:：]\s*([^\s/]+)")

# 馬場状態 "馬場:良" / "馬場 ： 稍重" — 開催当日に公表される（開催前は無いことも）。
# 長い表記 (稍重/不良) を短い表記 (良/重) より先に並べて誤マッチを防ぐ。
TRACK_CONDITION_RE = re.compile(r"馬場\s*[:：]\s*(稍重|不良|良|重)")

# JRA トラックコード（race_id の 5-6 桁目）→ コース名
COURSE_CODE_MAP: dict[str, str] = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}

# ── レースクラス正規化 ────────────────────────────────────────────────────────

# Roman numeral / Unicode 全角ローマ数字 / 半角数字すべてに対応する。
# 順序重要: GⅢ/GIII/G3 → GⅡ/GII/G2 → GⅠ/G1 で、長い prefix から評価する。
# GI(?![IV]) は "GII"/"GIII" の prefix として誤マッチしないための negative lookahead。
_CLASS_NORM_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"GⅢ|GIII|G3"), "G3"),
    (re.compile(r"GⅡ|GII|G2"), "G2"),
    (re.compile(r"GⅠ|G1|GI(?![IV])"), "G1"),
    (re.compile(r"Listed|\(L\)"), "Listed"),
    (re.compile(r"重賞"), "重賞"),
    (re.compile(r"未勝利"), "未勝利"),
    (re.compile(r"新馬"), "新馬"),
    (re.compile(r"1勝クラス"), "1勝クラス"),
    (re.compile(r"2勝クラス"), "2勝クラス"),
    (re.compile(r"3勝クラス"), "3勝クラス"),
    (re.compile(r"オープン|\bOP\b"), "OP"),
]


def normalize_race_class(raw: str) -> str | None:
    """raw テキストからレースクラスの canonical ラベルを返す。

    優先順位の高い rule から evaluate するので、たとえば "G1" は "重賞" より優先する。
    どの rule にもマッチしなければ None を返す。
    """
    for pattern, label in _CLASS_NORM_RULES:
        if pattern.search(raw):
            return label
    return None


# ── netkeiba 内部リンクからの ID 抽出 ──────────────────────────────────────────

def extract_id_from_href(href: str, kind: str) -> str | None:
    """netkeiba 内部リンク URL から entity ID を取り出す。

    対応形式:
      - /horse/<id>/                     (馬)
      - /jockey/result/recent/<id>/      (騎手)
      - /trainer/result/recent/<id>/     (調教師)

    `/<kind>/` 直後（必要なら中継パス result/recent/ 等を 0 回以上スキップして）の
    最初の英数字 ID を返す。マッチしなければ None。
    """
    m = re.search(rf"/{kind}/(?:[a-z_]+/)*([0-9a-zA-Z]+)", href)
    return m.group(1) if m else None


# ── 出走表の 1 行を読む ────────────────────────────────────────────────────────
#
# 結果ページ (race_result.py) と出馬表 (shutuba.py) は、テーブルの列名も行の形も
# ほぼ同じで、以前は同じヘルパを 2 つの関数の中に丸ごと写していた (34 行そのまま
# 一致)。netkeiba の表構造が変われば両方が同時に変わる = 同じ理由で変わるので、
# ここに 1 つだけ置く。

def build_column_index(table: Tag, *, what: str) -> dict[str, int]:
    """ヘッダ行から {列名: 列番号} を作る。

    <thead> が無く最初の <tr> の <th> が列名、というページが多いので両方見る。
    列名が 1 つも取れなければ空 dict を返す。呼び出し側は位置 (fallback_idx) で
    引くことになるので、警告だけ出して続ける。
    """
    headers: list[str] = []
    thead = table.find("thead")
    if thead:
        headers = [th.get_text(strip=True) for th in thead.find_all("th")]
    if not headers:
        first_tr = table.find("tr")
        if first_tr:
            headers = [th.get_text(strip=True) for th in first_tr.find_all("th")]

    col = {name: idx for idx, name in enumerate(headers)}
    if not col:
        logger.warning("No %s table headers found; falling back to fixed column positions", what)
    return col


class RowCells:
    """1 行の <td> を「列名、無ければ位置」で引く。

    netkeiba はページによって列名が違い、合成フィクスチャでは列名が無いことも
    ある。そのため列名と位置の二段構えが要る。
    """

    def __init__(self, tds: list[Tag], col: dict[str, int]) -> None:
        self._tds = tds
        self._col = col

    def _index(self, name: str, fallback_idx: int | None) -> int | None:
        idx = self._col.get(name, fallback_idx)
        if idx is None or idx >= len(self._tds):
            return None
        return idx

    def text(self, name: str, fallback_idx: int | None = None) -> str:
        idx = self._index(name, fallback_idx)
        return "" if idx is None else self._tds[idx].get_text(strip=True)

    def link(self, name: str, fallback_idx: int | None = None) -> Tag | None:
        idx = self._index(name, fallback_idx)
        return None if idx is None else self._tds[idx].find("a", href=True)


def to_int(text: str) -> int | None:
    """数字なら int。空欄・"**" (発走前の人気) などは None。"""
    try:
        return int(text)
    except (ValueError, TypeError):
        return None


def to_float(text: str) -> float | None:
    """数字なら float。空欄・"---.-" (発走前のオッズ) などは None。"""
    try:
        return float(text)
    except (ValueError, TypeError):
        return None


def name_from_link(tag: Tag | None) -> str | None:
    """<a> の title 属性か表示文字列から名前を取る。全角/半角スペースは詰める。"""
    if tag is None:
        return None
    raw = tag.get("title") or tag.get_text(strip=True)
    if not raw:
        return None
    cleaned = raw.strip().replace("　", "").replace(" ", "")
    return cleaned or None


def parse_sex_age(text: str) -> tuple[str | None, int | None]:
    """"牡3" のような性齢を (性, 齢) に分ける。読めない部分は None。"""
    if not text:
        return None, None
    sex = text[0] if text[0] in ("牡", "牝", "セ") else None
    age: int | None = None
    with contextlib.suppress(ValueError, IndexError):
        age = int(text[1:])
    return sex, age


def parse_entry_rows(table: Tag, col: dict[str, int], parse_row, *, what: str) -> list:
    """テーブルのデータ行を 1 行ずつ `parse_row(tds, col)` に渡して集める。

    ヘッダ行 (<th> を含む) と、<td> が 5 個未満の飾り行は飛ばす。1 行が壊れていても
    そのレース全体を捨てないよう、例外は握って警告に落とす — netkeiba は取消馬や
    注記でしばしば形の違う行を混ぜてくる。

    結果ページと出馬表で同じ走査をしていたので 1 つにした。行の読み方 (parse_row)
    だけが違う。
    """
    entries = []
    for tr in table.find_all("tr"):
        if tr.find("th"):  # ヘッダ行
            continue
        tds = tr.find_all("td")
        if len(tds) < 5:
            continue
        try:
            entry = parse_row(tds, col)
        except Exception as exc:  # noqa: BLE001 — 1 行の失敗でレースを捨てない
            logger.warning("Failed to parse %s entry row: %s", what, exc)
            continue
        if entry is not None:
            entries.append(entry)
    return entries
