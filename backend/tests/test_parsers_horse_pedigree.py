"""Tests for horse pedigree page parser."""

from __future__ import annotations

from pathlib import Path

from keiba_ai.scraper.parsers.horse_pedigree import ParsedPedigree, parse_horse_pedigree

FIXTURES = Path(__file__).parent / "fixtures"
HORSE_ID = "2022104732"
_HTML = (FIXTURES / "horse_pedigree_2022104732.html").read_text(encoding="utf-8")


def test_parse_horse_pedigree_extracts_sire():
    result = parse_horse_pedigree(_HTML, HORSE_ID)
    assert result.sire_name == "ロードカナロア"
    assert result.sire_id == "2008103552"


def test_parse_horse_pedigree_extracts_dam():
    result = parse_horse_pedigree(_HTML, HORSE_ID)
    assert result.dam_name == "スターハイネス"
    assert result.dam_id == "1955100309"


def test_parse_horse_pedigree_handles_no_blood_table():
    """No blood_table should return empty ParsedPedigree without raising."""
    result = parse_horse_pedigree("<html><body></body></html>", "9999999999")
    assert isinstance(result, ParsedPedigree)
    assert result.horse_id == "9999999999"
    assert result.sire_name is None
    assert result.dam_name is None
    assert result.sire_id is None
    assert result.dam_id is None


def test_parse_horse_pedigree_handles_insufficient_parents():
    """blood_table with only one large-rowspan TD should not raise.

    Sire is still extracted (the TD is present); only dam stays None because
    the TR count is shorter than sire's rowspan, so the dam row index is
    unreachable. Returning the partial sire is safer than discarding both.
    """
    html = """
    <html><body>
      <table class="blood_table detail">
        <tr><td rowspan="8"><a href="/horse/2008103552/">ロードカナロア</a></td></tr>
      </table>
    </body></html>
    """
    result = parse_horse_pedigree(html, "0000000001")
    assert isinstance(result, ParsedPedigree)
    assert result.sire_name == "ロードカナロア"
    assert result.dam_name is None
    assert result.dam_id is None
