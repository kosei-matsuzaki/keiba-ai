"""Tests for horse detail page parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from keiba_ai.scraper.parsers.horse_detail import ParsedHorseDetail, parse_horse_detail

FIXTURES = Path(__file__).parent / "fixtures"
HORSE_ID = "2022104732"
_HTML = (FIXTURES / "horse_detail_2022104732.html").read_text(encoding="utf-8")


def test_parse_horse_detail_extracts_name():
    result = parse_horse_detail(_HTML, HORSE_ID)
    assert result.name == "アパッシメント"


def test_parse_horse_detail_extracts_sex():
    result = parse_horse_detail(_HTML, HORSE_ID)
    assert result.sex == "セ"


def test_parse_horse_detail_extracts_birth_date_iso_format():
    result = parse_horse_detail(_HTML, HORSE_ID)
    assert result.birth_date == "2022-02-01"


def test_parse_horse_detail_handles_missing_fields_gracefully():
    """Completely empty HTML should not raise; all fields return None."""
    result = parse_horse_detail("<html><body></body></html>", "9999999999")
    assert isinstance(result, ParsedHorseDetail)
    assert result.horse_id == "9999999999"
    assert result.name is None
    assert result.sex is None
    assert result.birth_date is None


@pytest.mark.parametrize(
    ("sex_token", "expected"),
    [
        ("牡4歳", "牡"),
        ("牝3歳", "牝"),
        ("セ5歳", "セ"),
    ],
)
def test_parse_horse_detail_sex_variants(sex_token: str, expected: str):
    html = f"""
    <html><head><title>テスト馬 | netkeiba</title></head>
    <body>
      <div class="horse_title">Info: テスト馬 現役　{sex_token}　鹿毛</div>
    </body></html>
    """
    result = parse_horse_detail(html, "0000000001")
    assert result.sex == expected
