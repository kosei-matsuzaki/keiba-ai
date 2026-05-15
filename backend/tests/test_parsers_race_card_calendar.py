"""Tests for race-card calendar parser (race.netkeiba.com/top/race_list.html)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scraper.parsers.race_card_calendar import (
    ParseError,
    parse_race_ids_from_card_calendar,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def card_calendar_html() -> str:
    return (FIXTURES / "race_card_calendar_20260505.html").read_text(encoding="utf-8")


def test_extracts_central_race_ids(card_calendar_html: str) -> None:
    """フィクスチャの中央レース (東京2本 + 京都2本) が抽出されること。"""
    race_ids = parse_race_ids_from_card_calendar(card_calendar_html)
    assert race_ids == [
        "202605010101",
        "202605010111",
        "202608070901",
        "202608070911",
    ]


def test_no_duplicates(card_calendar_html: str) -> None:
    race_ids = parse_race_ids_from_card_calendar(card_calendar_html)
    assert len(race_ids) == len(set(race_ids))


def test_race_id_format(card_calendar_html: str) -> None:
    race_ids = parse_race_ids_from_card_calendar(card_calendar_html)
    for rid in race_ids:
        assert len(rid) == 12
        assert rid.isdigit()


def test_nar_excluded_by_default(card_calendar_html: str) -> None:
    """地方 (NAR) の大井レース (track code 44) はデフォルトで除外されること。"""
    race_ids = parse_race_ids_from_card_calendar(card_calendar_html)
    assert "202644010101" not in race_ids


def test_include_nar_keeps_all(card_calendar_html: str) -> None:
    """include_nar=True で地方も含めて返すこと。"""
    race_ids = parse_race_ids_from_card_calendar(card_calendar_html, include_nar=True)
    assert "202644010101" in race_ids
    # 中央も含まれる
    assert "202605010101" in race_ids


def test_raises_parse_error_on_empty_html() -> None:
    with pytest.raises(ParseError):
        parse_race_ids_from_card_calendar("<html><body><p>no races</p></body></html>")


def test_raises_parse_error_when_only_nar() -> None:
    """中央が 1 件もない HTML は ParseError。"""
    html = """
    <html><body>
      <a href="/race/shutuba.html?race_id=202644010101">大井 1R</a>
      <a href="/race/shutuba.html?race_id=202655010101">佐賀 1R</a>
    </body></html>
    """
    with pytest.raises(ParseError):
        parse_race_ids_from_card_calendar(html)


def test_central_only_filter_inline() -> None:
    """インライン HTML で JRA 中央のみフィルタが効くこと。"""
    html = """
    <html><body>
      <a href="/race/shutuba.html?race_id=202605010101">東京 1R</a>
      <a href="/race/shutuba.html?race_id=202608070901">京都 1R</a>
      <a href="/race/shutuba.html?race_id=202644010103">大井 3R</a>
    </body></html>
    """
    race_ids = parse_race_ids_from_card_calendar(html)
    assert race_ids == ["202605010101", "202608070901"]
    assert "202644010103" not in race_ids


