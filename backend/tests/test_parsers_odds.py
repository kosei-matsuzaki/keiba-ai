"""Tests for scraper/parsers/odds.py — live odds HTML parsing.

各フィクスチャの実 HTML を使いパーサの動作を検証する。
フィクスチャはすべて「当日レース前」の未確定状態なので odds は "---.-"。
パーサが正しく combo/bet_type を抽出し、odds=None を返すことを確認する。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from keiba_ai.scraper.parsers.odds import (
    LiveOddsRow,
    _parse_odds_text,
    _parse_range_odds,
    parse_sanrenpuku_odds,
    parse_sanrentan_odds,
    parse_tan_fuku_odds,
    parse_umaren_odds,
    parse_umatan_odds,
    parse_wide_odds,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# ユーティリティ関数テスト
# ---------------------------------------------------------------------------

class TestParseOddsText:
    def test_valid_float(self):
        assert _parse_odds_text("12.3") == pytest.approx(12.3)

    def test_undecided_dash(self):
        assert _parse_odds_text("---.-") is None

    def test_undecided_full_dash(self):
        assert _parse_odds_text("---") is None

    def test_empty_string(self):
        assert _parse_odds_text("") is None

    def test_comma_separated(self):
        assert _parse_odds_text("1,234.5") == pytest.approx(1234.5)

    def test_integer_like(self):
        assert _parse_odds_text("10") == pytest.approx(10.0)


class TestParseRangeOdds:
    def test_range_format(self):
        lo, hi = _parse_range_odds("1.5~3.0")
        assert lo == pytest.approx(1.5)
        assert hi == pytest.approx(3.0)

    def test_single_value(self):
        lo, hi = _parse_range_odds("2.5")
        assert lo == pytest.approx(2.5)
        assert hi is None

    def test_undecided_range(self):
        lo, hi = _parse_range_odds("---.-~---.-")
        assert lo is None
        assert hi is None

    def test_undecided_single(self):
        lo, hi = _parse_range_odds("---.-")
        assert lo is None
        assert hi is None


# ---------------------------------------------------------------------------
# 単勝・複勝パーサ (b1)
# ---------------------------------------------------------------------------

@pytest.fixture()
def tan_fuku_html() -> str:
    return (FIXTURES / "odds_real_tan_fuku.html").read_bytes().decode("euc-jp", errors="replace")


class TestParseTanFukuOdds:
    def test_returns_list(self, tan_fuku_html):
        rows = parse_tan_fuku_odds(tan_fuku_html)
        assert isinstance(rows, list)

    def test_has_tan_and_fuku_rows(self, tan_fuku_html):
        rows = parse_tan_fuku_odds(tan_fuku_html)
        bet_types = {r.bet_type for r in rows}
        assert "単勝" in bet_types
        assert "複勝" in bet_types

    def test_tan_rows_count(self, tan_fuku_html):
        rows = parse_tan_fuku_odds(tan_fuku_html)
        tan_rows = [r for r in rows if r.bet_type == "単勝"]
        # ホープフルステークス G1 = 18 頭
        assert len(tan_rows) == 18

    def test_fuku_rows_count(self, tan_fuku_html):
        rows = parse_tan_fuku_odds(tan_fuku_html)
        fuku_rows = [r for r in rows if r.bet_type == "複勝"]
        assert len(fuku_rows) == 18

    def test_tan_combo_format(self, tan_fuku_html):
        rows = parse_tan_fuku_odds(tan_fuku_html)
        tan_rows = [r for r in rows if r.bet_type == "単勝"]
        # combo は馬番の文字列
        combos = {r.combo for r in tan_rows}
        assert "1" in combos
        assert "18" in combos

    def test_tan_odds_undecided(self, tan_fuku_html):
        rows = parse_tan_fuku_odds(tan_fuku_html)
        tan_rows = [r for r in rows if r.bet_type == "単勝"]
        # フィクスチャはレース前なので全てオッズ未確定
        assert all(r.odds is None for r in tan_rows)

    def test_fuku_odds_undecided(self, tan_fuku_html):
        rows = parse_tan_fuku_odds(tan_fuku_html)
        fuku_rows = [r for r in rows if r.bet_type == "複勝"]
        assert all(r.odds is None for r in fuku_rows)

    def test_tan_odds_max_is_none(self, tan_fuku_html):
        rows = parse_tan_fuku_odds(tan_fuku_html)
        tan_rows = [r for r in rows if r.bet_type == "単勝"]
        assert all(r.odds_max is None for r in tan_rows)

    def test_fuku_is_livoddrsrow(self, tan_fuku_html):
        rows = parse_tan_fuku_odds(tan_fuku_html)
        fuku_rows = [r for r in rows if r.bet_type == "複勝"]
        for r in fuku_rows:
            assert isinstance(r, LiveOddsRow)


# ---------------------------------------------------------------------------
# 馬連パーサ (b4)
# ---------------------------------------------------------------------------

@pytest.fixture()
def umaren_html() -> str:
    return (FIXTURES / "odds_real_umaren.html").read_bytes().decode("euc-jp", errors="replace")


class TestParseUmarenOdds:
    def test_returns_list(self, umaren_html):
        rows = parse_umaren_odds(umaren_html)
        assert isinstance(rows, list)

    def test_all_bet_type_umaren(self, umaren_html):
        rows = parse_umaren_odds(umaren_html)
        assert all(r.bet_type == "馬連" for r in rows)

    def test_row_count_18_horses(self, umaren_html):
        rows = parse_umaren_odds(umaren_html)
        # テーブル 0: axis=1, rows=[2..18]=17
        # テーブル 1: axis=2, rows=[3..18]=16
        # ...
        # テーブル 16: axis=17, rows=[18]=1
        # → sum(17,16,...,1) = 153 = C(18,2)
        assert len(rows) == 153

    def test_combo_ascending(self, umaren_html):
        rows = parse_umaren_odds(umaren_html)
        for r in rows:
            parts = r.combo.split("-")
            assert len(parts) == 2
            assert int(parts[0]) < int(parts[1]), f"Not ascending: {r.combo}"

    def test_odds_undecided(self, umaren_html):
        rows = parse_umaren_odds(umaren_html)
        assert all(r.odds is None for r in rows)

    def test_no_odds_max(self, umaren_html):
        rows = parse_umaren_odds(umaren_html)
        assert all(r.odds_max is None for r in rows)


# ---------------------------------------------------------------------------
# ワイドパーサ (b5)
# ---------------------------------------------------------------------------

@pytest.fixture()
def wide_html() -> str:
    return (FIXTURES / "odds_real_wide.html").read_bytes().decode("euc-jp", errors="replace")


class TestParseWideOdds:
    def test_returns_list(self, wide_html):
        rows = parse_wide_odds(wide_html)
        assert isinstance(rows, list)

    def test_all_bet_type_wide(self, wide_html):
        rows = parse_wide_odds(wide_html)
        assert all(r.bet_type == "ワイド" for r in rows)

    def test_row_count(self, wide_html):
        rows = parse_wide_odds(wide_html)
        assert len(rows) == 153

    def test_combo_ascending(self, wide_html):
        rows = parse_wide_odds(wide_html)
        for r in rows:
            parts = r.combo.split("-")
            assert len(parts) == 2
            assert int(parts[0]) < int(parts[1])

    def test_odds_undecided(self, wide_html):
        rows = parse_wide_odds(wide_html)
        assert all(r.odds is None for r in rows)


# ---------------------------------------------------------------------------
# 馬単パーサ (b6)
# ---------------------------------------------------------------------------

@pytest.fixture()
def umatan_html() -> str:
    return (FIXTURES / "odds_real_umatan.html").read_bytes().decode("euc-jp", errors="replace")


class TestParseUmatanOdds:
    def test_returns_list(self, umatan_html):
        rows = parse_umatan_odds(umatan_html)
        assert isinstance(rows, list)

    def test_all_bet_type_umatan(self, umatan_html):
        rows = parse_umatan_odds(umatan_html)
        assert all(r.bet_type == "馬単" for r in rows)

    def test_row_count_axis_1(self, umatan_html):
        rows = parse_umatan_odds(umatan_html)
        # axis=1 の 18 テーブル、各 17 rows → 18*17=306?
        # 実際は 18 テーブル、各 16 rows(1 を除く) → depends on structure
        # fixtures shows 17 tables, each with 16 data rows → 17*16=272
        assert len(rows) == 272

    def test_combo_ordered_format(self, umatan_html):
        rows = parse_umatan_odds(umatan_html)
        for r in rows:
            assert "→" in r.combo, f"Missing arrow in combo: {r.combo}"
            parts = r.combo.split("→")
            assert len(parts) == 2

    def test_odds_undecided(self, umatan_html):
        rows = parse_umatan_odds(umatan_html)
        assert all(r.odds is None for r in rows)


# ---------------------------------------------------------------------------
# 三連複パーサ (b7)
# ---------------------------------------------------------------------------

@pytest.fixture()
def sanrenpuku_html() -> str:
    return (FIXTURES / "odds_real_sanrenpuku.html").read_bytes().decode("euc-jp", errors="replace")


class TestParseSanrenpukuOdds:
    def test_returns_list(self, sanrenpuku_html):
        rows = parse_sanrenpuku_odds(sanrenpuku_html)
        assert isinstance(rows, list)

    def test_all_bet_type_sanrenpuku(self, sanrenpuku_html):
        rows = parse_sanrenpuku_odds(sanrenpuku_html)
        assert all(r.bet_type == "三連複" for r in rows)

    def test_row_count(self, sanrenpuku_html):
        rows = parse_sanrenpuku_odds(sanrenpuku_html)
        # axis=1: sum(16,15,...,1) = 136 = C(17,2)
        assert len(rows) == 136

    def test_combo_ascending_three(self, sanrenpuku_html):
        rows = parse_sanrenpuku_odds(sanrenpuku_html)
        for r in rows:
            parts = r.combo.split("-")
            assert len(parts) == 3
            nums = [int(p) for p in parts]
            assert nums == sorted(nums), f"Not ascending: {r.combo}"

    def test_axis_horse_in_all_combos(self, sanrenpuku_html):
        rows = parse_sanrenpuku_odds(sanrenpuku_html)
        # axis=1 なので全 combo に "1" が含まれるはず
        for r in rows:
            parts = [int(p) for p in r.combo.split("-")]
            assert 1 in parts, f"axis horse not in combo: {r.combo}"

    def test_odds_undecided(self, sanrenpuku_html):
        rows = parse_sanrenpuku_odds(sanrenpuku_html)
        assert all(r.odds is None for r in rows)


# ---------------------------------------------------------------------------
# 三連単パーサ (b8)
# ---------------------------------------------------------------------------

@pytest.fixture()
def sanrentan_html() -> str:
    return (FIXTURES / "odds_real_sanrentan.html").read_bytes().decode("euc-jp", errors="replace")


class TestParseSanrentanOdds:
    def test_returns_list(self, sanrentan_html):
        rows = parse_sanrentan_odds(sanrentan_html)
        assert isinstance(rows, list)

    def test_all_bet_type_sanrentan(self, sanrentan_html):
        rows = parse_sanrentan_odds(sanrentan_html)
        assert all(r.bet_type == "三連単" for r in rows)

    def test_row_count(self, sanrentan_html):
        rows = parse_sanrentan_odds(sanrentan_html)
        # axis=1: 17 tables × 16 rows each = 272
        assert len(rows) == 272

    def test_combo_ordered_three(self, sanrentan_html):
        rows = parse_sanrentan_odds(sanrentan_html)
        for r in rows:
            assert "→" in r.combo, f"Missing arrow: {r.combo}"
            parts = r.combo.split("→")
            assert len(parts) == 3

    def test_axis_horse_is_first(self, sanrentan_html):
        rows = parse_sanrentan_odds(sanrentan_html)
        for r in rows:
            first = int(r.combo.split("→")[0])
            assert first == 1, f"axis should be 1 in combo: {r.combo}"

    def test_odds_undecided(self, sanrentan_html):
        rows = parse_sanrentan_odds(sanrentan_html)
        assert all(r.odds is None for r in rows)


# ---------------------------------------------------------------------------
# synthetic HTML での実オッズ確認
# ---------------------------------------------------------------------------

class TestParseOddsWithSyntheticHtml:
    """実オッズが入った簡易 HTML でパーサ動作を確認する。"""

    def test_tan_with_real_odds(self):
        html = """
        <div id="odds_tan_block">
          <h4>単勝</h4>
          <table class="RaceOdds_HorseList_Table">
            <tr><th>枠</th><th class="Waku">馬番</th><th class="Mark">印</th>
                <th>選択</th><th>馬名</th><th>オッズ</th></tr>
            <tr><td class="Waku1 W31">1</td><td class="W31">1</td>
                <td class="Mark_User"></td><td class="Horse_Select"></td>
                <td class="Horse_Name">テスト馬</td>
                <td class="Odds Popular">5.3</td></tr>
            <tr><td class="Waku1 W31">1</td><td class="W31">2</td>
                <td class="Mark_User"></td><td class="Horse_Select"></td>
                <td class="Horse_Name">テスト馬2</td>
                <td class="Odds Popular">12.5</td></tr>
          </table>
        </div>
        <div id="odds_fuku_block">
          <h4>複勝</h4>
          <table class="RaceOdds_HorseList_Table">
            <tr><th>枠</th><th class="Waku">馬番</th><th class="Mark">印</th>
                <th>選択</th><th>馬名</th><th>オッズ</th></tr>
            <tr><td class="Waku1 W31">1</td><td class="W31">1</td>
                <td class="Mark_User"></td><td class="Horse_Select"></td>
                <td class="Horse_Name">テスト馬</td>
                <td class="Odds Popular">1.5~2.8</td></tr>
          </table>
        </div>
        """
        rows = parse_tan_fuku_odds(html)
        tan_rows = [r for r in rows if r.bet_type == "単勝"]
        fuku_rows = [r for r in rows if r.bet_type == "複勝"]

        assert len(tan_rows) == 2
        assert tan_rows[0].combo == "1"
        assert tan_rows[0].odds == pytest.approx(5.3)
        assert tan_rows[0].odds_max is None
        assert tan_rows[1].odds == pytest.approx(12.5)

        assert len(fuku_rows) == 1
        assert fuku_rows[0].odds == pytest.approx(1.5)
        assert fuku_rows[0].odds_max == pytest.approx(2.8)

    def test_umaren_with_real_odds(self):
        html = """
        <table class="Odds_Table">
          <tr><td class="Waku1">1</td></tr>
          <tr><td class="Waku_Normal">2</td><td class="Odds Popular">25.4</td></tr>
          <tr><td class="Waku_Normal">3</td><td class="Odds Popular">18.2</td></tr>
        </table>
        """
        rows = parse_umaren_odds(html)
        assert len(rows) == 2
        assert rows[0].combo == "1-2"
        assert rows[0].odds == pytest.approx(25.4)
        assert rows[1].combo == "1-3"
        assert rows[1].odds == pytest.approx(18.2)

    def test_wide_with_range_odds(self):
        html = """
        <table class="Odds_Table">
          <tr><td class="Waku1">1</td></tr>
          <tr><td class="Waku_Normal">2</td><td class="Odds Popular">2.5~5.0</td></tr>
        </table>
        """
        rows = parse_wide_odds(html)
        assert len(rows) == 1
        assert rows[0].bet_type == "ワイド"
        assert rows[0].combo == "1-2"
        assert rows[0].odds == pytest.approx(2.5)
        assert rows[0].odds_max == pytest.approx(5.0)

    def test_umatan_with_real_odds(self):
        html = """
        <table class="Odds_Table">
          <tr><td class="Waku1">1</td></tr>
          <tr><td class="Waku_Normal">2</td><td class="Odds Popular">35.6</td></tr>
          <tr><td class="Waku_Normal">3</td><td class="Odds Popular">---.-</td></tr>
        </table>
        """
        rows = parse_umatan_odds(html)
        assert len(rows) == 2
        assert rows[0].combo == "1→2"
        assert rows[0].odds == pytest.approx(35.6)
        assert rows[1].combo == "1→3"
        assert rows[1].odds is None

    def test_sanrenpuku_with_real_odds(self):
        html = """
        <table class="Odds_Table">
          <tr><td class="Waku2">3</td></tr>
          <tr><td class="Waku_Normal">4</td><td class="Odds Popular">120.5</td></tr>
          <tr><td class="Waku_Normal">5</td><td class="Odds Popular">85.0</td></tr>
        </table>
        <table class="Odds_Table">
          <tr><td class="Waku2">4</td></tr>
          <tr><td class="Waku_Normal">5</td><td class="Odds Popular">200.0</td></tr>
        </table>
        """
        rows = parse_sanrenpuku_odds(html)
        # axis=2 (first_second=3 → axis=2)
        assert len(rows) == 3
        combos = {r.combo for r in rows}
        assert "2-3-4" in combos
        assert "2-3-5" in combos
        assert "2-4-5" in combos
        odds_map = {r.combo: r.odds for r in rows}
        assert odds_map["2-3-4"] == pytest.approx(120.5)

    def test_sanrentan_with_real_odds(self):
        html = """
        <table class="Odds_Table">
          <tr><td class="Waku1">2</td></tr>
          <tr><td class="Waku_Normal">3</td><td class="Odds Popular">450.0</td></tr>
        </table>
        <table class="Odds_Table">
          <tr><td class="Waku2">3</td></tr>
          <tr><td class="Waku_Normal">2</td><td class="Odds Popular">320.0</td></tr>
        </table>
        """
        rows = parse_sanrentan_odds(html)
        # axis=1 (first_second=2 → axis=1)
        combos = {r.combo: r.odds for r in rows}
        assert "1→2→3" in combos
        assert combos["1→2→3"] == pytest.approx(450.0)
        assert "1→3→2" in combos
        assert combos["1→3→2"] == pytest.approx(320.0)
