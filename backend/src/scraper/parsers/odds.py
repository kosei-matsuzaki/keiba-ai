"""Parse live odds from netkeiba odds pages.

各券種の URL:
  b1: 単勝・複勝  — race.netkeiba.com/odds/index.html?race_id=<id>&type=b1
  b3: 枠連        — &type=b3
  b4: 馬連        — &type=b4
  b5: ワイド      — &type=b5
  b6: 馬単        — &type=b6
  b7: 三連複      — &type=b7
  b8: 三連単      — &type=b8

HTML 構造のまとめ:
  b1: <table class="RaceOdds_HorseList_Table"> × 2
      parent id="odds_tan_block" → 単勝
      parent id="odds_fuku_block" → 複勝
      各行: 枠セル(WakuN) | 馬番(WN) | ... | オッズ(Odds Popular)
      オッズセル: 単勝="12.3", 複勝="1.5~3.0"（min~max）

  b4/b5: <table class="Odds_Table"> × N (axis horse ごとに 1 テーブル)
      各テーブル先頭行 = axis 馬番 (クラス=WakuN)
      以降の行: Waku_Normal(相手馬番) | Odds Popular(オッズ)
      b4 馬連: オッズ = "25.4"
      b5 ワイド: オッズ = "2.5~5.0"（min~max）
      axis < 相手 の組合せのみ存在するため、重複なく全 C(n,2) を網羅

  b6: 同構造、axis horse = 1着馬 (n テーブル)、相手 = 2着馬
      axios と相手が同一でも「1→2」「2→1」は別オッズなので全 P(n,2) 収録

  b7: <table class="Odds_Table"> × (n-2) テーブル（axis=1 implied）
      各テーブル先頭行 = second 馬番
      残行: third 馬番 | オッズ
      → 1-second-third の組合せ群。この 1 ページでは axis=1 のみ。
      注: 実際には複数 axis ページを fetch して全組合せを得る必要がある。
          ここでは HTML に含まれる組合せのみをパースする。

  b8: 同構造（axis=1 implied, ordered）
      各テーブル先頭行 = second 馬番
      残行: third 馬番 | オッズ
      → 1→second→third の組合せ群

combo 文字列形式は payouts.combo と一致させる:
  単勝: "3"
  複勝: "3"
  馬連: "3-7"  昇順
  ワイド: "3-7" 昇順
  馬単: "3→5"  順序つき
  三連複: "3-5-9" 昇順
  三連単: "3→5→9" 順序つき

オッズ未確定（"---.-" 等）の行は LiveOddsRow を返さない（skip）。
確定オッズが取れた combo のみ live_odds テーブルに書き込まれる。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

from core.logging import get_logger

logger = get_logger(__name__)

# オッズ未確定を示す文字列パターン
_UNDECIDED_RE = re.compile(r"^-+\.?-*$")


@dataclass
class LiveOddsRow:
    """パーサが返す 1 オッズレコード。"""
    bet_type: str
    combo: str
    odds: float | None       # 確定オッズ（複勝/ワイドは min）
    odds_max: float | None   # 複勝/ワイドの max オッズ。それ以外は None
    popularity: int | None   # 人気順位。HTML に含まれない場合は None


# ---------------------------------------------------------------------------
# 内部ユーティリティ
# ---------------------------------------------------------------------------

def _parse_odds_text(text: str) -> float | None:
    """オッズテキストを float に変換。未確定 ("---.-" 等) は None。"""
    t = text.strip().replace(",", "")
    if not t or _UNDECIDED_RE.match(t):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _parse_range_odds(text: str) -> tuple[float | None, float | None]:
    """複勝/ワイドの "min~max" または "min" テキストをパースする。

    Returns:
        (odds_min, odds_max): 範囲なしの場合 odds_max=None
    """
    t = text.strip()
    if "~" in t:
        parts = t.split("~", 1)
        return _parse_odds_text(parts[0]), _parse_odds_text(parts[1])
    v = _parse_odds_text(t)
    return v, None


def _to_int(text: str) -> int | None:
    cleaned = text.strip().replace(",", "")
    if not cleaned:
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


def _combo_ascending(*horse_nos: int) -> str:
    """昇順ソートした馬番をハイフン区切りで返す。"""
    return "-".join(str(h) for h in sorted(horse_nos))


def _combo_ordered(*horse_nos: int) -> str:
    """順序をそのままで矢印区切りの文字列を返す。"""
    return "→".join(str(h) for h in horse_nos)


# ---------------------------------------------------------------------------
# 単勝・複勝パーサ (b1)
# ---------------------------------------------------------------------------

def parse_tan_fuku_odds(html: str) -> list[LiveOddsRow]:
    """単勝 (b1) と複勝 (b1) のオッズをパースする。

    HTML: <table class="RaceOdds_HorseList_Table"> × 2
    parent id="odds_tan_block" → 単勝
    parent id="odds_fuku_block" → 複勝

    各データ行のセル構成:
      0: 枠番 (WakuN)
      1: 馬番 (WN 等)
      2: 印
      3: 選択
      4: 馬名
      5: オッズ (Odds Popular) — 単勝: "12.3" / 複勝: "1.5~3.0"
    """
    soup = BeautifulSoup(html, "lxml")
    rows: list[LiveOddsRow] = []

    for block_id, bet_type, is_range in [
        ("odds_tan_block", "単勝", False),
        ("odds_fuku_block", "複勝", True),
    ]:
        block = soup.find(id=block_id)
        if block is None:
            logger.warning("Block not found: %s", block_id)
            continue

        table = block.find("table", class_="RaceOdds_HorseList_Table")
        if table is None:
            logger.warning("RaceOdds_HorseList_Table not found in %s", block_id)
            continue

        tr_list = table.find_all("tr")
        for tr in tr_list:
            cells = tr.find_all(["td", "th"])
            # ヘッダ行はスキップ（最初の th セルが枠 or 印 等）
            if tr.find("th"):
                continue
            if len(cells) < 6:
                continue

            # 馬番セル (index 1: class WN / W31 等)
            bango_cell = cells[1]
            horse_no = _to_int(bango_cell.get_text(strip=True))
            if horse_no is None:
                continue

            # オッズセル (index 5: class Odds Popular)
            odds_cell = cells[5]
            odds_text = odds_cell.get_text(strip=True)

            if is_range:
                odds_min, odds_max = _parse_range_odds(odds_text)
                # オッズ未確定の行は live_odds に書き込まない
                if odds_min is None:
                    continue
                rows.append(LiveOddsRow(
                    bet_type=bet_type,
                    combo=str(horse_no),
                    odds=odds_min,
                    odds_max=odds_max,
                    popularity=None,
                ))
            else:
                odds_val = _parse_odds_text(odds_text)
                # オッズ未確定の行は live_odds に書き込まない
                if odds_val is None:
                    continue
                rows.append(LiveOddsRow(
                    bet_type=bet_type,
                    combo=str(horse_no),
                    odds=odds_val,
                    odds_max=None,
                    popularity=None,
                ))

    return rows


# ---------------------------------------------------------------------------
# 枠連パーサ (b3)
# ---------------------------------------------------------------------------

def parse_wakuren_odds(html: str) -> list[LiveOddsRow]:
    """枠連 (b3) のオッズをパースする。

    構造は馬連 (b4) と同様。Odds_Table × N、axis 枠番 < 相手枠番。
    """
    return _parse_pair_odds_tables(html, bet_type="枠連", is_range=False)


# ---------------------------------------------------------------------------
# 馬連パーサ (b4)
# ---------------------------------------------------------------------------

def parse_umaren_odds(html: str) -> list[LiveOddsRow]:
    """馬連 (b4) のオッズをパースする。

    HTML: <table class="Odds_Table"> × (n-1) テーブル
    各テーブル:
      先頭行: axis 馬番 (class=WakuN、セル 1 個)
      以降: Waku_Normal (相手馬番) | Odds Popular (オッズ)
    axios < 相手 の組合せのみ → combo は昇順 "axis-相手"
    """
    return _parse_pair_odds_tables(html, bet_type="馬連", is_range=False)


# ---------------------------------------------------------------------------
# ワイドパーサ (b5)
# ---------------------------------------------------------------------------

def parse_wide_odds(html: str) -> list[LiveOddsRow]:
    """ワイド (b5) のオッズをパースする。

    構造は馬連と同様だが、オッズセルが "min~max" 形式。
    """
    return _parse_pair_odds_tables(html, bet_type="ワイド", is_range=True)


def _parse_pair_odds_tables(
    html: str,
    bet_type: str,
    is_range: bool,
) -> list[LiveOddsRow]:
    """馬連・ワイド・枠連共通: Odds_Table テーブル群をパースする。

    各テーブルの先頭行が axis 馬番（単独セル、class=WakuN）。
    以降の行が相手馬番 (Waku_Normal) とオッズ (Odds Popular)。
    axis < 相手 の前提なので combo は axis-相手 の昇順形式で一致する。
    """
    soup = BeautifulSoup(html, "lxml")
    rows: list[LiveOddsRow] = []

    for table in soup.find_all("table", class_="Odds_Table"):
        tr_list = table.find_all("tr")
        if not tr_list:
            continue

        # 先頭行から axis 馬番を取得
        header_cells = tr_list[0].find_all(["td", "th"])
        if not header_cells:
            continue
        axis_no = _to_int(header_cells[0].get_text(strip=True))
        if axis_no is None:
            continue

        for tr in tr_list[1:]:
            cells = tr.find_all(["td", "th"])
            if len(cells) < 2:
                continue

            partner_no = _to_int(cells[0].get_text(strip=True))
            if partner_no is None:
                continue

            odds_text = cells[1].get_text(strip=True)
            combo = _combo_ascending(axis_no, partner_no)

            if is_range:
                odds_min, odds_max = _parse_range_odds(odds_text)
                # オッズ未確定の行は live_odds に書き込まない
                if odds_min is None:
                    continue
                rows.append(LiveOddsRow(
                    bet_type=bet_type,
                    combo=combo,
                    odds=odds_min,
                    odds_max=odds_max,
                    popularity=None,
                ))
            else:
                odds_val = _parse_odds_text(odds_text)
                # オッズ未確定の行は live_odds に書き込まない
                if odds_val is None:
                    continue
                rows.append(LiveOddsRow(
                    bet_type=bet_type,
                    combo=combo,
                    odds=odds_val,
                    odds_max=None,
                    popularity=None,
                ))

    return rows


# ---------------------------------------------------------------------------
# 馬単パーサ (b6)
# ---------------------------------------------------------------------------

def parse_umatan_odds(html: str) -> list[LiveOddsRow]:
    """馬単 (b6) のオッズをパースする。

    構造は馬連と同様だが、axis が 1 着馬なので combo は順序つき "axis→相手"。
    axis=1 の HTML では「1→2」「1→3」… が収録される。
    """
    soup = BeautifulSoup(html, "lxml")
    rows: list[LiveOddsRow] = []

    for table in soup.find_all("table", class_="Odds_Table"):
        tr_list = table.find_all("tr")
        if not tr_list:
            continue

        header_cells = tr_list[0].find_all(["td", "th"])
        if not header_cells:
            continue
        axis_no = _to_int(header_cells[0].get_text(strip=True))
        if axis_no is None:
            continue

        for tr in tr_list[1:]:
            cells = tr.find_all(["td", "th"])
            if len(cells) < 2:
                continue

            partner_no = _to_int(cells[0].get_text(strip=True))
            if partner_no is None:
                continue

            odds_text = cells[1].get_text(strip=True)
            odds_val = _parse_odds_text(odds_text)
            # オッズ未確定の行は live_odds に書き込まない
            if odds_val is None:
                continue
            combo = _combo_ordered(axis_no, partner_no)

            rows.append(LiveOddsRow(
                bet_type="馬単",
                combo=combo,
                odds=odds_val,
                odds_max=None,
                popularity=None,
            ))

    return rows


# ---------------------------------------------------------------------------
# 三連複パーサ (b7)
# ---------------------------------------------------------------------------

def parse_sanrenpuku_odds(html: str) -> list[LiveOddsRow]:
    """三連複 (b7) のオッズをパースする。

    この HTML は axis horse = 1 の組合せ群を保持する。
    (他の axis については別途 fetch が必要だが、ここでは HTML 内のデータのみを返す)

    構造:
      1 つの GraphOdds div 内に Odds_Table × (n-2) テーブル。
      各テーブル:
        先頭行: second 馬番 (WakuN セル)
        残行: Waku_Normal (third 馬番) | Odds Popular (オッズ)
      axios (=1) は HTML 外のコンテキストから定まるが、先行する div のテキストで確認可能。
      ここでは先頭テーブルの先頭行の値 (= 2) から axis=1 を推定する。

    combo: "1-second-third" 昇順
    """
    soup = BeautifulSoup(html, "lxml")
    rows: list[LiveOddsRow] = []

    tables = soup.find_all("table", class_="Odds_Table")
    if not tables:
        return rows

    # axis を推定: 先頭テーブルの second 馬番から最小の axis を決定
    # 先頭テーブルヘッダが "2" なら axis=1
    first_header_cells = tables[0].find_all("tr")[0].find_all(["td", "th"])
    if not first_header_cells:
        return rows
    first_second = _to_int(first_header_cells[0].get_text(strip=True))
    if first_second is None:
        return rows
    # axis = second - 1（先頭テーブルが最小の second の場合）
    # ただし second=2 なら axis=1 の前提
    # 一般化: axis は先頭テーブルの second より 1 小さい（連番前提）
    axis_no = first_second - 1

    for table in tables:
        tr_list = table.find_all("tr")
        if not tr_list:
            continue

        header_cells = tr_list[0].find_all(["td", "th"])
        if not header_cells:
            continue
        second_no = _to_int(header_cells[0].get_text(strip=True))
        if second_no is None:
            continue

        for tr in tr_list[1:]:
            cells = tr.find_all(["td", "th"])
            if len(cells) < 2:
                continue

            third_no = _to_int(cells[0].get_text(strip=True))
            if third_no is None:
                continue

            odds_text = cells[1].get_text(strip=True)
            odds_val = _parse_odds_text(odds_text)
            # オッズ未確定の行は live_odds に書き込まない
            if odds_val is None:
                continue
            combo = _combo_ascending(axis_no, second_no, third_no)

            rows.append(LiveOddsRow(
                bet_type="三連複",
                combo=combo,
                odds=odds_val,
                odds_max=None,
                popularity=None,
            ))

    return rows


# ---------------------------------------------------------------------------
# 三連単パーサ (b8)
# ---------------------------------------------------------------------------

def parse_sanrentan_odds(html: str) -> list[LiveOddsRow]:
    """三連単 (b8) のオッズをパースする。

    この HTML は axis horse = 1 (1着) の ordered 組合せ群を保持する。

    構造:
      Odds_Table × (n-1) テーブル（axis=1 implied）。
      各テーブル:
        先頭行: second 馬番 (2着: WakuN セル)
        残行: Waku_Normal (third 馬番: 3着) | Odds Popular (オッズ)

    combo: "1→second→third" 順序つき
    """
    soup = BeautifulSoup(html, "lxml")
    rows: list[LiveOddsRow] = []

    tables = soup.find_all("table", class_="Odds_Table")
    if not tables:
        return rows

    # axis 推定: 先頭テーブルの second の最小値から axis を決定
    first_header_cells = tables[0].find_all("tr")[0].find_all(["td", "th"])
    if not first_header_cells:
        return rows
    first_second = _to_int(first_header_cells[0].get_text(strip=True))
    if first_second is None:
        return rows
    # 三連単は全 second horse が対象（axis を除く全馬）
    # second の最小値より 1 小さいのが axis horse
    axis_no = first_second - 1

    for table in tables:
        tr_list = table.find_all("tr")
        if not tr_list:
            continue

        header_cells = tr_list[0].find_all(["td", "th"])
        if not header_cells:
            continue
        second_no = _to_int(header_cells[0].get_text(strip=True))
        if second_no is None:
            continue

        for tr in tr_list[1:]:
            cells = tr.find_all(["td", "th"])
            if len(cells) < 2:
                continue

            third_no = _to_int(cells[0].get_text(strip=True))
            if third_no is None:
                continue

            odds_text = cells[1].get_text(strip=True)
            odds_val = _parse_odds_text(odds_text)
            # オッズ未確定の行は live_odds に書き込まない
            if odds_val is None:
                continue
            combo = _combo_ordered(axis_no, second_no, third_no)

            rows.append(LiveOddsRow(
                bet_type="三連単",
                combo=combo,
                odds=odds_val,
                odds_max=None,
                popularity=None,
            ))

    return rows
