"""parsers モジュールで共有する正規表現・定数・ヘルパ関数。

race_result.py と shutuba.py が同じパターンを別実装しないよう、ここに集約する。
"""

from __future__ import annotations

import re

# ── 共通正規表現 ──────────────────────────────────────────────────────────────

# 馬体重 "484 (0)" / "478 (+2)" / "494(-2)" — race.netkeiba は括弧前にスペース、
# db.netkeiba はスペース無しなので \s* で両対応する。
WEIGHT_RE = re.compile(r"(\d+)\s*\(([+-]?\d+)\)")

# レースヘッダ "ダ右1200m / 天候:晴 / 馬場:良" — surface 直後に方向(右/左)・
# 内外(内/外)・距離 が続く。コース略号は db: "芝/ダ/障", race: 同等。
SURFACE_DIST_RE = re.compile(r"(芝|ダ|障)(?:\s*[右左])?(?:\s*[内外])?\s*(\d{3,4})\s*m")

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
