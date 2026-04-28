"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture()
def calendar_html() -> str:
    return (FIXTURES_DIR / "race_calendar_20241228.html").read_text(encoding="utf-8")


@pytest.fixture()
def race_result_html() -> str:
    return (FIXTURES_DIR / "race_result_202406010101.html").read_text(encoding="utf-8")


@pytest.fixture()
def robots_txt() -> str:
    return (FIXTURES_DIR / "robots.txt").read_text(encoding="utf-8")
