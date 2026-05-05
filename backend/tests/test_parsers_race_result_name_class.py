"""Tests for race_name and race_class extraction from real db.netkeiba HTML structure.

フィクスチャ:
  - race_result_real_db_netkeiba.html  : data_intro + smalltxt 形式 (有馬記念 G1)
  - race_result_maiden_db_netkeiba.html: data_intro + smalltxt 形式 (3歳未勝利)
  - race_result_202406010101.html      : 旧 RaceData02 形式 (G1 フィクスチャ)

修正前の挙動:
  - 全レースで race_class='OP' （"競馬データベースTOP" の "OP" に誤マッチ）

修正後の期待挙動:
  - 有馬記念 → name="有馬記念", race_class="G1"
  - 3歳未勝利 → name="3歳未勝利", race_class="未勝利"
  - 旧フィクスチャ → race_class="G1"（既存テストとの互換維持）
  - "競馬データベースTOP" リンクでは race_class が "OP" にならない
"""

from __future__ import annotations

from pathlib import Path

import pytest

from keiba_ai.scraper.parsers.race_result import parse_race_result

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def arima_html() -> str:
    return (FIXTURES / "race_result_real_db_netkeiba.html").read_text(encoding="utf-8")


@pytest.fixture()
def maiden_html() -> str:
    return (FIXTURES / "race_result_maiden_db_netkeiba.html").read_text(encoding="utf-8")


# ── 有馬記念 (G1) テスト ────────────────────────────────────────────────────────

class TestArimaKinen:
    def test_name(self, arima_html):
        parsed = parse_race_result(arima_html, "202412220601")
        assert parsed.name == "有馬記念"

    def test_race_class_is_g1(self, arima_html):
        """smalltxt に "G1" があるので race_class="G1" になること。"""
        parsed = parse_race_result(arima_html, "202412220601")
        assert parsed.race_class == "G1"

    def test_no_op_false_match(self, arima_html):
        """競馬データベースTOP の "OP" で race_class="OP" にならないこと。"""
        parsed = parse_race_result(arima_html, "202412220601")
        assert parsed.race_class != "OP"

    def test_surface(self, arima_html):
        parsed = parse_race_result(arima_html, "202412220601")
        assert parsed.surface == "芝"

    def test_distance(self, arima_html):
        parsed = parse_race_result(arima_html, "202412220601")
        assert parsed.distance == 2500

    def test_track_condition(self, arima_html):
        parsed = parse_race_result(arima_html, "202412220601")
        assert parsed.track_condition == "良"

    def test_n_runners(self, arima_html):
        parsed = parse_race_result(arima_html, "202412220601")
        assert parsed.n_runners == 2


# ── 3歳未勝利テスト ─────────────────────────────────────────────────────────────

class TestMaiden:
    def test_name(self, maiden_html):
        parsed = parse_race_result(maiden_html, "202402241001")
        assert parsed.name == "3歳未勝利"

    def test_race_class_is_mishori(self, maiden_html):
        """3歳未勝利 → race_class="未勝利" に正規化されること。"""
        parsed = parse_race_result(maiden_html, "202402241001")
        assert parsed.race_class == "未勝利"

    def test_no_op_false_match(self, maiden_html):
        parsed = parse_race_result(maiden_html, "202402241001")
        assert parsed.race_class != "OP"


# ── 旧フィクスチャ (RaceData02 形式) との後方互換テスト ──────────────────────────

@pytest.fixture()
def legacy_html() -> str:
    return (FIXTURES / "race_result_202406010101.html").read_text(encoding="utf-8")


class TestLegacyFixture:
    def test_race_class_g1(self, legacy_html):
        """旧 RaceData02 形式でも race_class="G1" が抽出できること。"""
        parsed = parse_race_result(legacy_html, "202406010101")
        assert parsed.race_class == "G1"

    def test_surface(self, legacy_html):
        parsed = parse_race_result(legacy_html, "202406010101")
        assert parsed.surface == "芝"

    def test_distance(self, legacy_html):
        parsed = parse_race_result(legacy_html, "202406010101")
        assert parsed.distance == 2400


# ── Roman numeral 正規化テスト ───────────────────────────────────────────────

class TestRomanNumeralNormalization:
    """h1 にローマ数字 GIII / GII / GI が含まれる graded race の正規化検証。

    実 db.netkeiba 結果ページではグレード表記が `<h1>第29回プロキオン
    ステークス(GIII)</h1>` のように Roman numeral で書かれており、
    Unicode 全角ローマ数字 (GⅢ / GⅡ / GⅠ) を期待する旧 regex では
    マッチせず "OP" に誤分類されていた回帰防止。
    """

    @staticmethod
    def _build_html(h1_text: str) -> str:
        return f'''<!DOCTYPE html><html><head><title>test</title></head>
<body>
<div class="data_intro">
  <dl class="racedata fc">
    <dt>11 R</dt>
    <dd>
      <h1>{h1_text}</h1>
      <p><span>芝1200m / 天候 : 晴 / 馬場 : 良 / 発走 : 15:25</span></p>
    </dd>
  </dl>
  <p class="smalltxt">2024年07月14日 1回中京テスト</p>
</div>
<a href="https://db.netkeiba.com/" title="競馬データベースTOP">競馬データベースTOP</a>
</body></html>'''

    def test_giii_roman_normalized_to_g3(self):
        html = self._build_html("第29回プロキオンステークス(GIII)")
        parsed = parse_race_result(html, "202410030411")
        assert parsed.race_class == "G3"

    def test_gii_roman_normalized_to_g2(self):
        html = self._build_html("函館記念(GII)")
        parsed = parse_race_result(html, "202401010411")
        assert parsed.race_class == "G2"

    def test_gi_roman_normalized_to_g1(self):
        html = self._build_html("天皇賞・春(GI)")
        parsed = parse_race_result(html, "202403030411")
        assert parsed.race_class == "G1"

    def test_giii_does_not_match_g1(self):
        """GIII 中の "GI" 部分が G1 に誤マッチしないこと。"""
        html = self._build_html("テストGIII記念")
        parsed = parse_race_result(html, "202410030412")
        assert parsed.race_class == "G3"

    def test_gii_does_not_match_g1(self):
        html = self._build_html("テストGII記念")
        parsed = parse_race_result(html, "202410030413")
        assert parsed.race_class == "G2"
