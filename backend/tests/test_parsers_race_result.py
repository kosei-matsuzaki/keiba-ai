"""Tests for race result parser."""

from __future__ import annotations

import json

import pytest

from keiba_ai.scraper.parsers.race_result import ParsedRaceResult, parse_race_result


RACE_ID = "202406010101"


@pytest.fixture()
def parsed(race_result_html) -> ParsedRaceResult:
    return parse_race_result(race_result_html, RACE_ID)


def test_race_id(parsed):
    assert parsed.race_id == RACE_ID


def test_surface(parsed):
    assert parsed.surface == "芝"


def test_distance(parsed):
    assert parsed.distance == 2400


def test_weather(parsed):
    assert parsed.weather == "晴"


def test_track_condition(parsed):
    assert parsed.track_condition == "良"


def test_race_class(parsed):
    assert parsed.race_class == "G1"


def test_n_runners(parsed):
    assert parsed.n_runners == 3


def test_payout_win(parsed):
    assert parsed.payout_win == 310


def test_payout_place_is_json(parsed):
    assert parsed.payout_place is not None
    place = json.loads(parsed.payout_place)
    assert place["1"] == 140
    assert place["2"] == 110
    assert place["3"] == 170


def test_entries_count(parsed):
    assert len(parsed.entries) == 3


def test_first_entry_horse_id(parsed):
    assert parsed.entries[0].horse_id == "2019105293"


def test_first_entry_finish_position(parsed):
    assert parsed.entries[0].finish_position == 1


def test_first_entry_post_position(parsed):
    assert parsed.entries[0].post_position == 5


def test_first_entry_sex_age(parsed):
    e = parsed.entries[0]
    assert e.sex == "牡"
    assert e.age == 5


def test_first_entry_weight_carried(parsed):
    assert parsed.entries[0].weight_carried == 57.0


def test_first_entry_jockey_id(parsed):
    assert parsed.entries[0].jockey_id == "01167"


def test_first_entry_trainer_id(parsed):
    assert parsed.entries[0].trainer_id == "01096"


def test_first_entry_finish_time(parsed):
    # 2:23.1 = 143.1 seconds
    assert abs(parsed.entries[0].finish_time - 143.1) < 0.01


def test_first_entry_horse_weight(parsed):
    e = parsed.entries[0]
    assert e.horse_weight == 486
    assert e.horse_weight_diff == 2


def test_first_entry_odds_win(parsed):
    assert parsed.entries[0].odds_win == 3.1


def test_first_entry_popularity(parsed):
    assert parsed.entries[0].popularity == 2


def test_first_entry_horse_name(parsed):
    assert parsed.entries[0].horse_name == "ドウデュース"


def test_first_entry_jockey_name(parsed):
    assert parsed.entries[0].jockey_name == "武豊"


def test_first_entry_trainer_name(parsed):
    assert parsed.entries[0].trainer_name == "友道康夫"


def test_first_entry_agari_3f(parsed):
    assert parsed.entries[0].agari_3f == pytest.approx(35.1)


def test_first_entry_passing(parsed):
    assert parsed.entries[0].passing == "2-2"


def test_entries_have_passing(parsed):
    for e in parsed.entries:
        assert e.passing is not None


def test_entries_have_agari_3f(parsed):
    for e in parsed.entries:
        assert e.agari_3f is not None
