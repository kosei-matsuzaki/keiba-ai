"""Parse payout section from netkeiba race result page.

スコープ: 単勝・複勝のみ（馬連 / 三連単 等は無視）。

実 HTML 構造（2026 時点で確認済）:
  <table class="pay_table_01">
    <tr>
      <th class="tan">単勝</th>
      <td>8</td>                  <!-- 馬番 -->
      <td class="txt_r">170</td>  <!-- 払戻金（"円" 無しの素数字、3桁以上はカンマ区切り） -->
      <td class="txt_r">1</td>    <!-- 人気 -->
    </tr>
    <tr>
      <th class="fuku">複勝</th>
      <td>8<br/>10<br/>7</td>           <!-- 1〜3 着の馬番 -->
      <td class="txt_r">110<br/>160<br/>150</td>  <!-- 各払戻 -->
      <td class="txt_r">1<br/>4<br/>2</td>        <!-- 人気 -->
    </tr>
    ...
  </table>

Returns:
  payout_win: int | None         (例: 170)
  payout_place: dict[str,int]    (例: {"1": 110, "2": 160, "3": 150})
"""

from __future__ import annotations

from bs4 import BeautifulSoup

from keiba_ai.core.logging import get_logger

logger = get_logger(__name__)


def _to_int(text: str) -> int | None:
    """カンマ・空白を除去して int に変換。"""
    cleaned = text.replace(",", "").replace("円", "").strip()
    if not cleaned:
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


def _split_br(td) -> list[str]:
    """<td>110<br/>160<br/>150</td> を ['110','160','150'] に分割。"""
    return [s.strip() for s in td.get_text("\n").split("\n") if s.strip()]


def parse_payout(html: str) -> tuple[int | None, dict[str, int] | None]:
    """単勝・複勝の払戻金を抽出。

    Returns:
        (payout_win, payout_place):
            - payout_win は int 円。取得失敗時は None
            - payout_place は finish_position 文字列キー (str(int))
              例: {"1": 110, "2": 160, "3": 150}
              取得失敗時は None
    """
    soup = BeautifulSoup(html, "lxml")
    payout_win: int | None = None
    payout_place: dict[str, int] = {}

    # netkeiba 結果ページは <table class="pay_table_01"> が複数並ぶ
    # （単勝〜馬連のテーブルとワイド〜三連単のテーブル）
    tables = soup.find_all("table", class_="pay_table_01")
    if not tables:
        # 後方互換: 旧フィクスチャの class="pay_block" にもフォールバック
        tables = soup.find_all("table", class_="pay_block")
    if not tables:
        logger.warning("No payout table found — returning None payouts")
        return None, None

    for table in tables:
        for row in table.find_all("tr"):
            th = row.find("th")
            if th is None:
                continue
            label = th.get_text(strip=True)
            # 払戻金 td は最初の class="txt_r"
            amount_td = row.find("td", class_="txt_r")
            if amount_td is None:
                continue
            amounts = _split_br(amount_td)

            if "単勝" in label:
                if amounts:
                    payout_win = _to_int(amounts[0])
            elif "複勝" in label:
                for i, txt in enumerate(amounts[:3], start=1):
                    val = _to_int(txt)
                    if val is not None:
                        payout_place[str(i)] = val

    return payout_win, (payout_place if payout_place else None)
