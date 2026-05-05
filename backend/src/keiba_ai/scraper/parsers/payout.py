"""Parse payout section from netkeiba race result page.

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
    <!-- 枠連・馬連・ワイド・馬単・三連複・三連単 も同構造 -->
    ...
  </table>
  <!-- 連系券種は 2 枚目テーブルに並ぶことが多い -->
  <table class="pay_table_01"> ... </table>

Returns (parse_payout):
  payout_win: int | None         (例: 170)
  payout_place: dict[str,int]    (例: {"1": 110, "2": 160, "3": 150})

Returns (parse_payouts):
  list[PayoutRow] — 全 bet_type × combo の行リスト
"""

from __future__ import annotations

from dataclasses import dataclass

from bs4 import BeautifulSoup

from keiba_ai.core.logging import get_logger

logger = get_logger(__name__)

# th class 属性 → bet_type 文字列 の対応表
# netkeiba の class 名は略称。複数パターン（表記揺れ）を first-match で処理する。
_TH_CLASS_TO_BET_TYPE: dict[str, str] = {
    "tan": "単勝",
    "fuku": "複勝",
    "waku": "枠連",
    "uren": "馬連",
    "wide": "ワイド",
    "utan": "馬単",
    "sanrenpuku": "三連複",
    "sanrentan": "三連単",
}


@dataclass
class PayoutRow:
    """1 レース分の 1 bet_type × 1 combo に対応する払戻レコード。"""
    bet_type: str
    combo: str
    amount: int
    popularity: int | None


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


def _bet_type_from_th(th) -> str | None:
    """<th> タグの class 属性から bet_type 文字列を解決する。

    class="tan" → "単勝" など。
    テキストラベルにも照合してフォールバックを提供する。
    """
    classes = th.get("class") or []
    for cls in classes:
        if cls in _TH_CLASS_TO_BET_TYPE:
            return _TH_CLASS_TO_BET_TYPE[cls]

    # class 属性が無い / 不明な場合はテキストで照合
    label = th.get_text(strip=True)
    for bet_type in _TH_CLASS_TO_BET_TYPE.values():
        if bet_type in label:
            return bet_type
    return None


def _parse_combo_td(td) -> list[str]:
    """馬番 td (<td>5-8</td> or <td>5<br/>8</td>) をパースして馬番文字列リストに変換。

    netkeiba は馬連/三連単等のコンボを "5-8" や "5→8" という文字列で 1 セルに入れることがある。
    複勝やワイドの複数コンボは <br/> 区切りで複数行になる。

    【前提】netkeiba HTML は馬連・三連複・ワイドの馬番を常に昇順（小さい番号が先）で提供する。
    例: 馬連 "3-5"（3 < 5）、ワイド "3-5" / "3-9" / "5-9"、三連複 "3-5-9"。
    この関数は HTML テキストをそのまま返すため、昇順ソートは HTML 側で保証される前提に依存している。
    馬単・三連単は着順情報そのものなので昇順ではなく HTML 順（例: "3→5"、"3→5→9"）。
    """
    # HTML のテキストをそのまま返す（昇順ソートは netkeiba HTML 側が保証）
    raw = _split_br(td)
    return raw


def _find_payout_tables(soup: BeautifulSoup) -> list:
    tables = soup.find_all("table", class_="pay_table_01")
    if not tables:
        tables = soup.find_all("table", class_="pay_block")
    return tables


def parse_payout(html: str) -> tuple[int | None, dict[str, int] | None]:
    """単勝・複勝の払戻金を抽出。後方互換 API（変更禁止）。

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
    tables = _find_payout_tables(soup)
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


def parse_payouts(html: str) -> list[PayoutRow]:
    """全 bet_type の払戻情報を PayoutRow リストとして返す。

    netkeiba の pay_table_01 テーブル内の全行を走査し、
    8 種の bet_type（単勝・複勝・枠連・馬連・ワイド・馬単・三連複・三連単）を
    combo 単位で 1 PayoutRow として収集する。

    combo 文字列の形式（netkeiba HTML の正規化前提に依存）:
      単勝/複勝:  "8"（馬番のみ）
      馬連:       "3-5"（ハイフン区切り、馬番昇順）
      ワイド:     "3-5" / "3-9" / "5-9"（ハイフン区切り、馬番昇順）
      三連複:     "3-5-9"（ハイフン区切り、馬番昇順）
      枠連:       "2-3"（ハイフン区切り、枠番昇順）
      馬単:       "3→5"（矢印区切り、着順そのまま — 順序が情報）
      三連単:     "3→5→9"（矢印区切り、着順そのまま — 順序が情報）

    昇順ソートは netkeiba HTML 側で保証される前提であり、
    このパーサ内でのソート処理は行わない。

    Returns:
        PayoutRow のリスト。払戻テーブルが無い場合は空リスト。
        combo 数と amount 数が不一致の malformed HTML を検知した場合は、
        該当 bet_type の rows をスキップし warning ログを出力する。
    """
    soup = BeautifulSoup(html, "lxml")
    rows: list[PayoutRow] = []

    tables = _find_payout_tables(soup)
    if not tables:
        logger.warning("No payout table found — parse_payouts returns empty list")
        return rows

    for table in tables:
        for tr in table.find_all("tr"):
            th = tr.find("th")
            if th is None:
                continue

            bet_type = _bet_type_from_th(th)
            if bet_type is None:
                continue

            # td 一覧を取得（th を除く）
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue

            combo_td = tds[0]
            amount_td = next((td for td in tds if "txt_r" in (td.get("class") or [])), None)
            if amount_td is None:
                continue

            combos_raw = _parse_combo_td(combo_td)
            amounts = _split_br(amount_td)

            # 人気 td: class="txt_r" が複数あれば 2 番目
            txt_r_tds = [td for td in tds if "txt_r" in (td.get("class") or [])]
            popularity_td = txt_r_tds[1] if len(txt_r_tds) >= 2 else None
            popularities = _split_br(popularity_td) if popularity_td else []

            if bet_type in ("単勝", "複勝"):
                # 複勝は combo が <br/> 区切りで複数行（馬番ごとに 1 PayoutRow）
                if len(combos_raw) != len(amounts):
                    logger.warning(
                        "Mismatch combos vs amounts for %s: %d vs %d",
                        bet_type, len(combos_raw), len(amounts),
                    )
                    continue
                for idx, (combo_str, amount_str) in enumerate(zip(combos_raw, amounts)):
                    amount = _to_int(amount_str)
                    if amount is None:
                        continue
                    # popularities 列の長さが異なる場合は popularity=None で続行
                    pop = _to_int(popularities[idx]) if idx < len(popularities) else None
                    rows.append(PayoutRow(
                        bet_type=bet_type,
                        combo=combo_str,
                        amount=amount,
                        popularity=pop,
                    ))
            elif bet_type in ("ワイド",):
                # ワイド: 複数コンボが <br/> 区切りで並ぶ
                if len(combos_raw) != len(amounts):
                    logger.warning(
                        "Mismatch combos vs amounts for %s: %d vs %d",
                        bet_type, len(combos_raw), len(amounts),
                    )
                    continue
                for idx, (combo_str, amount_str) in enumerate(zip(combos_raw, amounts)):
                    amount = _to_int(amount_str)
                    if amount is None:
                        continue
                    # popularities 列の長さが異なる場合は popularity=None で続行
                    pop = _to_int(popularities[idx]) if idx < len(popularities) else None
                    rows.append(PayoutRow(
                        bet_type=bet_type,
                        combo=combo_str,
                        amount=amount,
                        popularity=pop,
                    ))
            else:
                # 枠連・馬連・馬単・三連複・三連単: 1 コンボ 1 行
                if not combos_raw or not amounts:
                    continue
                combo_str = combos_raw[0]
                amount = _to_int(amounts[0])
                if amount is None:
                    continue
                pop = _to_int(popularities[0]) if popularities else None
                rows.append(PayoutRow(
                    bet_type=bet_type,
                    combo=combo_str,
                    amount=amount,
                    popularity=pop,
                ))

    return rows
