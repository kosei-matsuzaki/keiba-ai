"""Tests for scraper/parsers/race_info_top.py."""

from __future__ import annotations

import pytest

from keiba_ai.scraper.parsers.race_info_top import ParseError, parse_race_ids


def _make_payload(*race_ids: str) -> dict:
    """Helper to build a minimal valid API payload."""
    return {
        "status": "OK",
        "data": {
            "info": [{"RaceId": rid} for rid in race_ids],
        },
    }


class TestParseRaceIds:
    def test_extracts_race_ids_from_valid_payload(self) -> None:
        payload = _make_payload("202506050101", "202506050102", "202506050201")
        result = parse_race_ids(payload)
        assert result == ["202506050101", "202506050102", "202506050201"]

    def test_returns_sorted_unique_ids(self) -> None:
        # Duplicate and out-of-order entries should be deduplicated and sorted.
        payload = _make_payload("202506050201", "202506050101", "202506050101")
        result = parse_race_ids(payload)
        assert result == ["202506050101", "202506050201"]

    def test_returns_empty_list_when_info_is_empty(self) -> None:
        payload = {"status": "OK", "data": {"info": []}}
        result = parse_race_ids(payload)
        assert result == []

    def test_skips_entries_without_race_id(self) -> None:
        payload = {
            "status": "OK",
            "data": {
                "info": [
                    {"RaceId": "202506050101"},
                    {"other_field": "value"},  # no RaceId
                    {"RaceId": None},           # None
                ],
            },
        }
        result = parse_race_ids(payload)
        assert result == ["202506050101"]

    def test_skips_non_12_digit_race_ids(self) -> None:
        payload = {
            "status": "OK",
            "data": {
                "info": [
                    {"RaceId": "202506050101"},   # valid
                    {"RaceId": "short"},           # too short
                    {"RaceId": "12345678901234"},  # too long
                ],
            },
        }
        result = parse_race_ids(payload)
        assert result == ["202506050101"]

    def test_raises_parse_error_when_status_not_ok(self) -> None:
        payload = {"status": "NG", "data": {"info": []}}
        with pytest.raises(ParseError, match="status='NG'"):
            parse_race_ids(payload)

    def test_raises_parse_error_when_data_missing(self) -> None:
        payload = {"status": "OK"}
        with pytest.raises(ParseError, match="'data'"):
            parse_race_ids(payload)

    def test_raises_parse_error_when_info_missing(self) -> None:
        payload = {"status": "OK", "data": {}}
        with pytest.raises(ParseError, match="'data.info'"):
            parse_race_ids(payload)

    def test_raises_parse_error_when_info_is_not_list(self) -> None:
        payload = {"status": "OK", "data": {"info": "not-a-list"}}
        with pytest.raises(ParseError):
            parse_race_ids(payload)
