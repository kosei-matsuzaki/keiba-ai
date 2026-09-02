"""parsers モジュールで共有する正規表現・定数・ヘルパ関数。

race_result.py と shutuba.py が同じパターンを別実装しないよう、ここに集約する。
"""

from __future__ import annotations

import re

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
    r"(?:\s*[右左直線内外襷回りコース・\-−ー]|\s*ダート|\s*芝)*"
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
