"""Tests for scraper/parsers/race_info_top.py."""

from __future__ import annotations

import pytest

from scraper.parsers.race_info_top import (
    ParseError,
    extract_jra_race_ids_with_kaisai_groups,
    parse_race_ids,
)


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


class TestExtractJraRaceIdsWithKaisaiGroups:
    def test_dedupes_repeated_race_ids_within_a_group(self) -> None:
        """netkeiba race_info_top API は同じ race_id を 4-5 回返すことがあるため、
        groups[key] と jra_race_ids 双方で重複排除されている必要がある。
        """
        # 同じ race_id を 4 回繰り返す（実 API の挙動を再現）
        payload = _make_payload(
            "202605020501", "202605020501", "202605020501", "202605020501",
            "202605020502", "202605020502", "202605020502",
        )
        jra_ids, groups = extract_jra_race_ids_with_kaisai_groups(payload)

        assert jra_ids == ["202605020501", "202605020502"]
        assert "2026050205" in groups
        assert groups["2026050205"] == ["202605020501", "202605020502"]

    def test_groups_by_kaisai_day_key(self) -> None:
        """race_id[:10] で kaisai_day_key を作り、別キーごとにリスト化する。"""
        payload = _make_payload(
            # venue 05 / kaisai 02 / day 05
            "202605020501", "202605020512",
            # venue 06 / kaisai 02 / day 05
            "202606020501",
            # venue 05 / kaisai 02 / day 06 (異なる日)
            "202605020601",
        )
        _, groups = extract_jra_race_ids_with_kaisai_groups(payload)

        assert set(groups.keys()) == {"2026050205", "2026060205", "2026050206"}
        assert groups["2026050205"] == ["202605020501", "202605020512"]
        assert groups["2026060205"] == ["202606020501"]
        assert groups["2026050206"] == ["202605020601"]

    def test_excludes_nar_venues(self) -> None:
        """venue code (race_id[4:6]) 11 以上 (NAR) は除外する。"""
        payload = _make_payload(
            "202605020501",  # JRA (venue 05 at [4:6])
            "202611050201",  # NAR (venue 11 at [4:6])
            "202612050201",  # NAR (venue 12 at [4:6])
        )
        jra_ids, groups = extract_jra_race_ids_with_kaisai_groups(payload)

        assert jra_ids == ["202605020501"]
        assert list(groups.keys()) == ["2026050205"]
