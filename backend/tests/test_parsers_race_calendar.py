"""Tests for race calendar parser."""

from __future__ import annotations

import pytest

from keiba_ai.scraper.parsers.race_calendar import ParseError, parse_race_ids_from_calendar


def test_extracts_race_ids(calendar_html):
    """Fixture is the 2024-12-28 開催日 page (中山6回5日目 + 京都8回7日目).

    netkeiba race_id encodes track/開催回/日目, NOT the date itself, so
    expected IDs reflect the 開催 metadata of the listed races.
    """
    race_ids = parse_race_ids_from_calendar(calendar_html)
    assert race_ids == [
        "202406050901",
        "202406050911",
        "202408070901",
        "202408070911",
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
