"""Tests for scraper/parsers/payout.py.

既存の parse_payout()（単数形）テストと、新規追加の parse_payouts()（複数形）テストを含む。
既存テストは変更しない方針で、新規テスト関数を追加する。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from keiba_ai.scraper.parsers.payout import PayoutRow, parse_payout, parse_payouts

FIXTURES = Path(__file__).parent / "fixtures"
RESULT_HTML = (FIXTURES / "race_result_202406010101.html").read_text(encoding="utf-8")
ALL_PAYOUT_HTML = (FIXTURES / "race_result_all_payout_types.html").read_text(encoding="utf-8")


# ── 既存 parse_payout() テスト（後方互換ロック） ─────────────────────────────

def test_parse_payout_win():
    """単勝払戻が正しく抽出される。"""
    win, _ = parse_payout(RESULT_HTML)
    assert win == 310


def test_parse_payout_place():
    """複勝払戻が finish_position キーで正しく抽出される。"""
    _, place = parse_payout(RESULT_HTML)
    assert place == {"1": 140, "2": 110, "3": 170}


def test_parse_payout_no_table():
    """払戻テーブルが無い場合は (None, None) を返す。"""
    win, place = parse_payout("<html><body></body></html>")
    assert win is None
    assert place is None


# ── parse_payouts() テスト ──────────────────────────────────────────────────

def test_parse_payouts_returns_list():
    """parse_payouts() は PayoutRow のリストを返す。"""
    rows = parse_payouts(RESULT_HTML)
    assert isinstance(rows, list)


def test_parse_payouts_basic_fixture_has_tan_fuku():
    """既存フィクスチャ（単勝・複勝のみ）で 4 行返る（複勝 3 行 + 単勝 1 行）。"""
    rows = parse_payouts(RESULT_HTML)
    bet_types = [r.bet_type for r in rows]
    assert "単勝" in bet_types
    assert "複勝" in bet_types


def test_parse_payouts_tan_row():
    """単勝は combo='5', amount=310, popularity=2。"""
    rows = parse_payouts(RESULT_HTML)
    tan = next(r for r in rows if r.bet_type == "単勝")
    assert tan.combo == "5"
    assert tan.amount == 310
    assert tan.popularity == 2


def test_parse_payouts_fuku_rows():
    """複勝は 3 行、各 combo は馬番文字列。"""
    rows = parse_payouts(RESULT_HTML)
    fuku = [r for r in rows if r.bet_type == "複勝"]
    assert len(fuku) == 3
    combos = [r.combo for r in fuku]
    assert "5" in combos
    assert "8" in combos
    assert "3" in combos


def test_parse_payouts_no_table_returns_empty():
    """払戻テーブルが無い場合は空リストを返す。"""
    rows = parse_payouts("<html><body></body></html>")
    assert rows == []


# ── 全 bet_type フィクスチャを使ったテスト ──────────────────────────────────

def _rows_by_type(html: str) -> dict[str, list[PayoutRow]]:
    rows = parse_payouts(html)
    result: dict[str, list[PayoutRow]] = {}
    for r in rows:
        result.setdefault(r.bet_type, []).append(r)
    return result


def test_parse_payouts_all_8_bet_types():
    """全 8 bet_type が抽出される。"""
    by_type = _rows_by_type(ALL_PAYOUT_HTML)
    expected_types = {"単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "三連複", "三連単"}
    assert expected_types == set(by_type.keys())


def test_parse_payouts_tan_all_fixture():
    """全種フィクスチャ: 単勝 combo='3', amount=520, popularity=1。"""
    by_type = _rows_by_type(ALL_PAYOUT_HTML)
    tan = by_type["単勝"]
    assert len(tan) == 1
    assert tan[0].combo == "3"
    assert tan[0].amount == 520
    assert tan[0].popularity == 1


def test_parse_payouts_fuku_all_fixture():
    """全種フィクスチャ: 複勝 3 行、金額・人気が正しい。"""
    by_type = _rows_by_type(ALL_PAYOUT_HTML)
    fuku = by_type["複勝"]
    assert len(fuku) == 3
    amounts = {r.combo: r.amount for r in fuku}
    assert amounts["3"] == 130
    assert amounts["5"] == 180
    assert amounts["9"] == 350
    pops = {r.combo: r.popularity for r in fuku}
    assert pops["3"] == 1
    assert pops["5"] == 2
    assert pops["9"] == 3


def test_parse_payouts_wakuren_all_fixture():
    """全種フィクスチャ: 枠連 combo='2-3', amount=1040, popularity=2。"""
    by_type = _rows_by_type(ALL_PAYOUT_HTML)
    waku = by_type["枠連"]
    assert len(waku) == 1
    assert waku[0].combo == "2-3"
    assert waku[0].amount == 1040
    assert waku[0].popularity == 2


def test_parse_payouts_umaren_all_fixture():
    """全種フィクスチャ: 馬連 combo='3-5', amount=2160, popularity=4。"""
    by_type = _rows_by_type(ALL_PAYOUT_HTML)
    uren = by_type["馬連"]
    assert len(uren) == 1
    assert uren[0].combo == "3-5"
    assert uren[0].amount == 2160
    assert uren[0].popularity == 4


def test_parse_payouts_wide_all_fixture():
    """全種フィクスチャ: ワイド 3 行、各 combo と amount が正しい。"""
    by_type = _rows_by_type(ALL_PAYOUT_HTML)
    wide = by_type["ワイド"]
    assert len(wide) == 3
    amounts = {r.combo: r.amount for r in wide}
    assert amounts["3-5"] == 620
    assert amounts["3-9"] == 1450
    assert amounts["5-9"] == 2380
    pops = {r.combo: r.popularity for r in wide}
    assert pops["3-5"] == 2
    assert pops["3-9"] == 5
    assert pops["5-9"] == 8


def test_parse_payouts_umatan_all_fixture():
    """全種フィクスチャ: 馬単 combo='3→5'（矢印 U+2192）, amount=4320。"""
    by_type = _rows_by_type(ALL_PAYOUT_HTML)
    utan = by_type["馬単"]
    assert len(utan) == 1
    assert utan[0].combo == "3→5"
    assert utan[0].amount == 4320
    assert utan[0].popularity == 7


def test_parse_payouts_sanrenpuku_all_fixture():
    """全種フィクスチャ: 三連複 combo='3-5-9', amount=8750, popularity=12。"""
    by_type = _rows_by_type(ALL_PAYOUT_HTML)
    srp = by_type["三連複"]
    assert len(srp) == 1
    assert srp[0].combo == "3-5-9"
    assert srp[0].amount == 8750
    assert srp[0].popularity == 12


def test_parse_payouts_sanrentan_all_fixture():
    """全種フィクスチャ: 三連単 combo='3→5→9'（矢印 U+2192）, amount=52300。"""
    by_type = _rows_by_type(ALL_PAYOUT_HTML)
    srt = by_type["三連単"]
    assert len(srt) == 1
    assert srt[0].combo == "3→5→9"
    assert srt[0].amount == 52300
    assert srt[0].popularity == 45


def test_parse_payouts_payout_row_is_dataclass():
    """PayoutRow が dataclass で bet_type/combo/amount/popularity フィールドを持つ。"""
    row = PayoutRow(bet_type="単勝", combo="5", amount=310, popularity=2)
    assert row.bet_type == "単勝"
    assert row.combo == "5"
    assert row.amount == 310
    assert row.popularity == 2


def test_parse_payouts_popularity_none_when_missing():
    """人気 td が無い場合は popularity=None になる。"""
    html = """
    <html><body>
    <table class="pay_table_01">
      <tbody>
        <tr>
          <th class="tan">単勝</th>
          <td>3</td>
          <td class="txt_r">500</td>
        </tr>
      </tbody>
    </table>
    </body></html>
    """
    rows = parse_payouts(html)
    assert len(rows) == 1
    assert rows[0].bet_type == "単勝"
    assert rows[0].amount == 500
    assert rows[0].popularity is None


def test_parse_payouts_comma_in_amount():
    """払戻金にカンマが含まれていても正しくパースされる（既存 _to_int の流用確認）。"""
    by_type = _rows_by_type(ALL_PAYOUT_HTML)
    # 三連単は 52,300 円
    srt = by_type["三連単"]
    assert srt[0].amount == 52300


# ── malformed HTML: combo/amount 行数不一致テスト ────────────────────────────

def test_parse_payouts_mismatch_combos_amounts_returns_empty_and_warns(caplog):
    """複勝で combo 3 つ・amount 2 つの malformed HTML を渡したとき、
    該当 bet_type の rows が返らず（空）、warning ログが出る。"""
    # combo td に 3 行、amount td に 2 行を意図的に仕込む
    html = """
    <html><body>
    <table class="pay_table_01">
      <tbody>
        <tr>
          <th class="fuku">複勝</th>
          <td>3<br/>5<br/>9</td>
          <td class="txt_r">130<br/>180</td>
          <td class="txt_r">1<br/>2<br/>3</td>
        </tr>
      </tbody>
    </table>
    </body></html>
    """
    import logging

    with caplog.at_level(logging.WARNING, logger="keiba_ai.scraper.parsers.payout"):
        rows = parse_payouts(html)

    # 不一致の bet_type は skip されるため rows は空
    assert rows == []
    # warning ログが出ていること
    assert any("Mismatch combos vs amounts" in record.message for record in caplog.records)
