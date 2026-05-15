"""Tests for race calendar parser."""

from __future__ import annotations

import pytest

from scraper.parsers.race_calendar import ParseError, parse_race_ids_from_calendar


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


def test_central_only_filter_skips_nar():
    """地方競馬の race_id (track code 30+) は default で skip される。"""
    html = """
    <html><body>
      <a href="/race/202406050901/">中山 1R</a>
      <a href="/race/202408070901/">京都 1R</a>
      <a href="/race/202444010103/">大井 3R</a>
      <a href="/race/202455010101/">佐賀 1R</a>
    </body></html>
    """
    race_ids = parse_race_ids_from_calendar(html)
    assert race_ids == ["202406050901", "202408070901"]
    # 地方 (44=大井, 55=佐賀) は除外


def test_include_nar_keeps_all():
    """include_nar=True で地方も含めて返す。"""
    html = """
    <html><body>
      <a href="/race/202406050901/">中山 1R</a>
      <a href="/race/202444010103/">大井 3R</a>
    </body></html>
    """
    race_ids = parse_race_ids_from_calendar(html, include_nar=True)
    assert race_ids == ["202406050901", "202444010103"]


def test_central_filter_raises_when_only_nar():
    """中央が 1 件もない (地方のみ) HTML は ParseError。"""
    html = """
    <html><body>
      <a href="/race/202444010103/">大井 3R</a>
      <a href="/race/202455010101/">佐賀 1R</a>
    </body></html>
    """
    with pytest.raises(ParseError):
        parse_race_ids_from_calendar(html)
