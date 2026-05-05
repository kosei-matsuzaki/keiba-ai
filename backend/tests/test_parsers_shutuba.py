"""Tests for shutuba (出馬表) parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from keiba_ai.scraper.parsers.shutuba import ParseError, ParsedShutuba, parse_shutuba

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
        # race_id 5-6桁目 "01" → "札幌" ... "05" → "東京"
        # RACE_ID_16 = "202406010111" → 5-6桁目 = "01" → 札幌
        # フィクスチャの race_id では COURSE_CODE_MAP で解決する
        # "202406010111"[4:6] = "01" → 札幌
        assert parsed_16.course == "札幌"

    def test_surface(self, parsed_16):
        assert parsed_16.surface == "芝"

    def test_distance(self, parsed_16):
        assert parsed_16.distance == 2000

    def test_weather(self, parsed_16):
        assert parsed_16.weather == "晴"

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
