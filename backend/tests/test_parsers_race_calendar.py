"""Tests for race calendar parser."""

from __future__ import annotations

import pytest

from keiba_ai.scraper.parsers.race_calendar import ParseError, parse_race_ids_from_calendar


def test_extracts_race_ids(calendar_html):
    race_ids = parse_race_ids_from_calendar(calendar_html)
    assert race_ids == [
        "202412280101",
        "202412280102",
        "202412280103",
        "202412280201",
        "202412280202",
    ]


def test_no_duplicates(calendar_html):
    race_ids = parse_race_ids_from_calendar(calendar_html)
    assert len(race_ids) == len(set(race_ids))


def test_raises_parse_error_on_empty_html():
    with pytest.raises(ParseError):
        parse_race_ids_from_calendar("<html><body><p>no races here</p></body></html>")


def test_race_id_format(calendar_html):
    race_ids = parse_race_ids_from_calendar(calendar_html)
    for rid in race_ids:
        assert len(rid) == 12
        assert rid.isdigit()
