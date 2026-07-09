"""Tests for shutuba (出馬表) parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from scraper.parsers.shutuba import ParsedShutuba, ParseError, parse_shutuba

FIXTURES = Path(__file__).parent / "fixtures"

RACE_ID_16 = "202406010111"
RACE_ID_NOWEATHER = "202406010101"
RACE_ID_MAIDEN = "202406010102"


@pytest.fixture()
def shutuba_html_16() -> str:
    return (FIXTURES / "shutuba_202406010111.html").read_text(encoding="utf-8")


@pytest.fixture()
def shutuba_html_noweather() -> str:
    return (FIXTURES / "shutuba_202406010101_noweather.html").read_text(encoding="utf-8")


@pytest.fixture()
def shutuba_html_maiden() -> str:
    return (FIXTURES / "shutuba_202406010102_maiden.html").read_text(encoding="utf-8")


@pytest.fixture()
def parsed_16(shutuba_html_16: str) -> ParsedShutuba:
    return parse_shutuba(shutuba_html_16, RACE_ID_16)


@pytest.fixture()
def parsed_noweather(shutuba_html_noweather: str) -> ParsedShutuba:
    return parse_shutuba(shutuba_html_noweather, RACE_ID_NOWEATHER)


@pytest.fixture()
def parsed_maiden(shutuba_html_maiden: str) -> ParsedShutuba:
    return parse_shutuba(shutuba_html_maiden, RACE_ID_MAIDEN)


# ── 16頭立て標準ケース ────────────────────────────────────────────────────────

class TestStandard16Runners:
    def test_race_id(self, parsed_16):
        assert parsed_16.race_id == RACE_ID_16

    def test_course_from_race_id(self, parsed_16):
        # race_id[4:6] が JRA 開催場コード（0-indexed）。
        # RACE_ID_16 = "202406010111" → [4:6] = "06" → 中山
        # (コード: 01=札幌, 05=東京, 06=中山, 09=阪神 等)
        assert parsed_16.course == "中山"

    def test_surface(self, parsed_16):
        assert parsed_16.surface == "芝"

    def test_distance(self, parsed_16):
        assert parsed_16.distance == 2000

    def test_weather(self, parsed_16):
        assert parsed_16.weather == "晴"

    def test_track_condition(self, parsed_16):
        # 当日に公表される馬場状態 (馬場:良) を RaceData01 から抽出する。
        assert parsed_16.track_condition == "良"

    def test_n_runners(self, parsed_16):
        assert parsed_16.n_runners == 16

    def test_entries_count(self, parsed_16):
        assert len(parsed_16.entries) == 16

    def test_first_entry_horse_id(self, parsed_16):
        assert parsed_16.entries[0].horse_id == "2019105293"

    def test_first_entry_horse_name(self, parsed_16):
        assert parsed_16.entries[0].horse_name == "ドウデュース"

    def test_first_entry_post_position(self, parsed_16):
        assert parsed_16.entries[0].post_position == 1

    def test_first_entry_sex(self, parsed_16):
        assert parsed_16.entries[0].sex == "牡"

    def test_first_entry_age(self, parsed_16):
        assert parsed_16.entries[0].age == 5

    def test_first_entry_weight_carried(self, parsed_16):
        assert parsed_16.entries[0].weight_carried == 57.0

    def test_first_entry_jockey_id(self, parsed_16):
        assert parsed_16.entries[0].jockey_id == "01167"

    def test_first_entry_jockey_name(self, parsed_16):
        assert parsed_16.entries[0].jockey_name == "武豊"

    def test_first_entry_trainer_id(self, parsed_16):
        assert parsed_16.entries[0].trainer_id == "01096"

    def test_first_entry_trainer_name(self, parsed_16):
        assert parsed_16.entries[0].trainer_name == "友道康夫"

    def test_first_entry_horse_weight(self, parsed_16):
        assert parsed_16.entries[0].horse_weight == 486

    def test_first_entry_horse_weight_diff(self, parsed_16):
        assert parsed_16.entries[0].horse_weight_diff == 2

    def test_first_entry_odds_win(self, parsed_16):
        assert parsed_16.entries[0].odds_win == pytest.approx(3.1)

    def test_first_entry_popularity(self, parsed_16):
        assert parsed_16.entries[0].popularity == 1

    def test_finish_position_is_none_for_all(self, parsed_16):
        """出馬表なので finish_position は常に None。"""
        for e in parsed_16.entries:
            assert e.finish_position is None

    def test_agari_3f_is_none_for_all(self, parsed_16):
        for e in parsed_16.entries:
            assert e.agari_3f is None

    def test_passing_is_none_for_all(self, parsed_16):
        for e in parsed_16.entries:
            assert e.passing is None

    def test_female_entry_sex(self, parsed_16):
        """9番（牝）の sex パース確認。"""
        entry_9 = parsed_16.entries[8]  # 馬番9
        assert entry_9.sex == "牝"

    def test_last_entry_post_position(self, parsed_16):
        assert parsed_16.entries[15].post_position == 16

    def test_no_duplicate_horse_ids(self, parsed_16):
        horse_ids = [e.horse_id for e in parsed_16.entries]
        assert len(horse_ids) == len(set(horse_ids))


# ── 天候未公開ケース ──────────────────────────────────────────────────────────

class TestNoWeather:
    def test_weather_is_none(self, parsed_noweather):
        """天候未公開のページでは weather が None になること。"""
        assert parsed_noweather.weather is None

    def test_track_condition_is_none(self, parsed_noweather):
        """馬場状態未公開のページでは track_condition が None になること。"""
        assert parsed_noweather.track_condition is None

    def test_surface_still_parsed(self, parsed_noweather):
        assert parsed_noweather.surface == "ダ"

    def test_distance_still_parsed(self, parsed_noweather):
        assert parsed_noweather.distance == 1200

    def test_n_runners(self, parsed_noweather):
        assert parsed_noweather.n_runners == 2

    def test_entries_have_horse_weight(self, parsed_noweather):
        for e in parsed_noweather.entries:
            assert e.horse_weight is not None


# ── 新馬戦（馬体重「計不」）ケース ───────────────────────────────────────────

class TestMaidenRaceNoWeight:
    def test_horse_weight_is_none(self, parsed_maiden):
        """新馬戦で「計不」の場合、horse_weight は None になること。"""
        for e in parsed_maiden.entries:
            assert e.horse_weight is None
            assert e.horse_weight_diff is None

    def test_weather_parsed(self, parsed_maiden):
        assert parsed_maiden.weather == "雨"

    def test_track_condition_parsed(self, parsed_maiden):
        # 馬場:重 — 稍重 より先に評価されるが「重」を正しく取れること。
        assert parsed_maiden.track_condition == "重"

    def test_surface_parsed(self, parsed_maiden):
        assert parsed_maiden.surface == "芝"

    def test_n_runners(self, parsed_maiden):
        assert parsed_maiden.n_runners == 3

    def test_castrated_sex_parsed(self, parsed_maiden):
        """「セ」（セン馬）の sex パース確認。"""
        entry_3 = parsed_maiden.entries[2]  # 馬番3
        assert entry_3.sex == "セ"

    def test_entries_all_have_race_id(self, parsed_maiden):
        for e in parsed_maiden.entries:
            assert e.race_id == RACE_ID_MAIDEN


# ── エラーケース ──────────────────────────────────────────────────────────────

def test_raises_parse_error_on_empty_html():
    """テーブルが存在しない HTML は ParseError を発生させること。"""
    with pytest.raises(ParseError):
        parse_shutuba("<html><body><p>no table here</p></body></html>", "202406010111")


# ── race.netkeiba.com 形式の合成フィクスチャ ──────────────────────────────────

# 実ページ capture の再配布を避けるため、実 HTML と同じ DOM 構造・列名
# （厩舎 / 馬体重(増減) / 更新、"---.-" / "**" プレースホルダ、EUC-JP）だけを
# 再現した合成フィクスチャを使う。
#
# fixtures/shutuba_synthetic_arima.html:
#   race_id=202406050911, EUC-JP エンコード
#   ホープフルS 相当（2024-12-28, 中山, 芝2000m, 18頭）
#   発走前の HTML のためオッズ/人気は "---.-" / "**" で未公開
#
# fixtures/shutuba_synthetic_r1.html:
#   race_id=202401010101, EUC-JP エンコード
#   ２歳未勝利 相当（2024-07-20, 札幌, 芝1200m, 5頭）


def _read_euc(path: Path) -> str:
    """EUC-JP エンコードの HTML（実ページと同じ符号化の合成フィクスチャ）をデコードして返す。"""
    return path.read_bytes().decode("euc-jp", errors="replace")


@pytest.fixture()
def synthetic_arima_html() -> str:
    return _read_euc(FIXTURES / "shutuba_synthetic_arima.html")


@pytest.fixture()
def synthetic_r1_html() -> str:
    return _read_euc(FIXTURES / "shutuba_synthetic_r1.html")


RACE_ID_SYNTH_ARIMA = "202406050911"
RACE_ID_SYNTH_R1 = "202401010101"


@pytest.fixture()
def parsed_synthetic_arima(synthetic_arima_html: str) -> ParsedShutuba:
    return parse_shutuba(synthetic_arima_html, RACE_ID_SYNTH_ARIMA)


@pytest.fixture()
def parsed_synthetic_r1(synthetic_r1_html: str) -> ParsedShutuba:
    return parse_shutuba(synthetic_r1_html, RACE_ID_SYNTH_R1)


class TestSyntheticArima:
    """race.netkeiba 形式の合成フィクスチャ (shutuba_synthetic_arima.html) — G1 18頭。

    オッズ・人気は発走前未公開なので None が正しい。
    """

    def test_race_id(self, parsed_synthetic_arima):
        assert parsed_synthetic_arima.race_id == RACE_ID_SYNTH_ARIMA

    def test_date(self, parsed_synthetic_arima):
        # <title> タグから抽出: "2024年12月28日"
        assert parsed_synthetic_arima.date == "2024-12-28"

    def test_course(self, parsed_synthetic_arima):
        # race_id[4:6] = "06" -> 中山
        assert parsed_synthetic_arima.course == "中山"

    def test_race_name(self, parsed_synthetic_arima):
        # <h1 class="RaceName"> から取得
        assert parsed_synthetic_arima.name == "ホープフルS"

    def test_race_class(self, parsed_synthetic_arima):
        # <title> に "(G1)" が含まれる
        assert parsed_synthetic_arima.race_class == "G1"

    def test_surface(self, parsed_synthetic_arima):
        assert parsed_synthetic_arima.surface == "芝"

    def test_distance(self, parsed_synthetic_arima):
        assert parsed_synthetic_arima.distance == 2000

    def test_weather(self, parsed_synthetic_arima):
        assert parsed_synthetic_arima.weather == "晴"

    def test_n_runners(self, parsed_synthetic_arima):
        assert parsed_synthetic_arima.n_runners == 18

    def test_entries_count(self, parsed_synthetic_arima):
        assert len(parsed_synthetic_arima.entries) == 18

    def test_post_positions_sequential(self, parsed_synthetic_arima):
        """馬番が 1〜n_runners の連番であること。"""
        positions = [e.post_position for e in parsed_synthetic_arima.entries]
        assert positions == list(range(1, 19))

    def test_first_entry_horse_id(self, parsed_synthetic_arima):
        assert parsed_synthetic_arima.entries[0].horse_id == "2022103995"

    def test_first_entry_horse_name(self, parsed_synthetic_arima):
        assert parsed_synthetic_arima.entries[0].horse_name == "ジョバンニ"

    def test_first_entry_post_position(self, parsed_synthetic_arima):
        assert parsed_synthetic_arima.entries[0].post_position == 1

    def test_first_entry_sex_age(self, parsed_synthetic_arima):
        entry = parsed_synthetic_arima.entries[0]
        assert entry.sex == "牡"
        assert entry.age == 2

    def test_first_entry_weight_carried(self, parsed_synthetic_arima):
        assert parsed_synthetic_arima.entries[0].weight_carried == pytest.approx(56.0)

    def test_first_entry_jockey_id(self, parsed_synthetic_arima):
        assert parsed_synthetic_arima.entries[0].jockey_id == "01126"

    def test_first_entry_jockey_name(self, parsed_synthetic_arima):
        assert parsed_synthetic_arima.entries[0].jockey_name == "松山"

    def test_first_entry_trainer_id(self, parsed_synthetic_arima):
        assert parsed_synthetic_arima.entries[0].trainer_id == "01157"

    def test_first_entry_trainer_name(self, parsed_synthetic_arima):
        assert parsed_synthetic_arima.entries[0].trainer_name == "杉山晴"

    def test_first_entry_horse_weight(self, parsed_synthetic_arima):
        # "484(0)" -> horse_weight=484, horse_weight_diff=0
        assert parsed_synthetic_arima.entries[0].horse_weight == 484
        assert parsed_synthetic_arima.entries[0].horse_weight_diff == 0

    def test_entry_horse_weight_with_diff(self, parsed_synthetic_arima):
        # 2番: "478 (+2)"
        assert parsed_synthetic_arima.entries[1].horse_weight == 478
        assert parsed_synthetic_arima.entries[1].horse_weight_diff == 2

    def test_odds_win_none_before_race(self, parsed_synthetic_arima):
        """発走前は単勝オッズ "---.-" -> None となること。"""
        for e in parsed_synthetic_arima.entries:
            assert e.odds_win is None

    def test_popularity_none_before_race(self, parsed_synthetic_arima):
        """発走前は人気 "**" -> None となること。"""
        for e in parsed_synthetic_arima.entries:
            assert e.popularity is None

    def test_finish_position_none(self, parsed_synthetic_arima):
        for e in parsed_synthetic_arima.entries:
            assert e.finish_position is None

    def test_no_duplicate_horse_ids(self, parsed_synthetic_arima):
        horse_ids = [e.horse_id for e in parsed_synthetic_arima.entries]
        assert len(horse_ids) == len(set(horse_ids))

    def test_all_entries_have_race_id(self, parsed_synthetic_arima):
        for e in parsed_synthetic_arima.entries:
            assert e.race_id == RACE_ID_SYNTH_ARIMA

    def test_all_entries_have_trainer_id(self, parsed_synthetic_arima):
        """全エントリで trainer_id が取得できること（厩舎列の修正確認）。"""
        for e in parsed_synthetic_arima.entries:
            assert e.trainer_id is not None, f"trainer_id missing for horse {e.horse_name}"

    def test_all_entries_have_horse_weight(self, parsed_synthetic_arima):
        """全エントリで馬体重が取得できること（馬体重(増減)列の修正確認）。"""
        for e in parsed_synthetic_arima.entries:
            assert e.horse_weight is not None, f"horse_weight missing for horse {e.horse_name}"


class TestSyntheticR1:
    """race.netkeiba 形式の合成フィクスチャ (shutuba_synthetic_r1.html) — ２歳未勝利 札幌1R (5頭)。"""

    def test_race_id(self, parsed_synthetic_r1):
        assert parsed_synthetic_r1.race_id == RACE_ID_SYNTH_R1

    def test_date(self, parsed_synthetic_r1):
        assert parsed_synthetic_r1.date == "2024-07-20"

    def test_course(self, parsed_synthetic_r1):
        # race_id[4:6] = "01" -> 札幌
        assert parsed_synthetic_r1.course == "札幌"

    def test_race_name(self, parsed_synthetic_r1):
        assert parsed_synthetic_r1.name == "2歳未勝利"

    def test_race_class(self, parsed_synthetic_r1):
        assert parsed_synthetic_r1.race_class == "未勝利"

    def test_surface(self, parsed_synthetic_r1):
        assert parsed_synthetic_r1.surface == "芝"

    def test_distance(self, parsed_synthetic_r1):
        assert parsed_synthetic_r1.distance == 1200

    def test_weather(self, parsed_synthetic_r1):
        assert parsed_synthetic_r1.weather == "晴"

    def test_n_runners(self, parsed_synthetic_r1):
        assert parsed_synthetic_r1.n_runners == 5

    def test_entries_count(self, parsed_synthetic_r1):
        assert len(parsed_synthetic_r1.entries) == 5

    def test_post_positions_sequential(self, parsed_synthetic_r1):
        positions = [e.post_position for e in parsed_synthetic_r1.entries]
        assert positions == list(range(1, 6))

    def test_first_entry_horse_id(self, parsed_synthetic_r1):
        assert parsed_synthetic_r1.entries[0].horse_id == "2022105762"

    def test_first_entry_horse_name(self, parsed_synthetic_r1):
        assert parsed_synthetic_r1.entries[0].horse_name == "ルージュアマリア"

    def test_first_entry_sex_age(self, parsed_synthetic_r1):
        entry = parsed_synthetic_r1.entries[0]
        assert entry.sex == "牝"
        assert entry.age == 2

    def test_first_entry_weight_carried(self, parsed_synthetic_r1):
        assert parsed_synthetic_r1.entries[0].weight_carried == pytest.approx(55.0)

    def test_first_entry_jockey_id(self, parsed_synthetic_r1):
        assert parsed_synthetic_r1.entries[0].jockey_id == "01188"

    def test_first_entry_trainer_id(self, parsed_synthetic_r1):
        assert parsed_synthetic_r1.entries[0].trainer_id == "01133"

    def test_first_entry_horse_weight(self, parsed_synthetic_r1):
        # "410 (+6)"
        assert parsed_synthetic_r1.entries[0].horse_weight == 410
        assert parsed_synthetic_r1.entries[0].horse_weight_diff == 6

    def test_all_entries_have_trainer_id(self, parsed_synthetic_r1):
        for e in parsed_synthetic_r1.entries:
            assert e.trainer_id is not None

    def test_all_entries_have_horse_weight(self, parsed_synthetic_r1):
        for e in parsed_synthetic_r1.entries:
            assert e.horse_weight is not None

    def test_odds_win_none_before_race(self, parsed_synthetic_r1):
        for e in parsed_synthetic_r1.entries:
            assert e.odds_win is None

    def test_popularity_none_before_race(self, parsed_synthetic_r1):
        for e in parsed_synthetic_r1.entries:
            assert e.popularity is None
