"""Parse netkeiba horse detail page (https://db.netkeiba.com/horse/<id>/)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

from core.logging import get_logger

logger = get_logger(__name__)

_DATE_RE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")
_SEX_AGE_RE = re.compile(r"(牡|牝|セ)\d+歳")
# <title> の先頭語（全角スペース・カッコ・半角スペースの前まで）
_NAME_RE = re.compile(r"^([^\s(（）]+)")


@dataclass
class ParsedHorseDetail:
    horse_id: str
    name: str | None = None
    sex: str | None = None
    birth_date: str | None = None  # "YYYY-MM-DD"


def parse_horse_detail(html: str, horse_id: str) -> ParsedHorseDetail:
    """Parse horse detail page. Returns ParsedHorseDetail with extracted fields.

    Best-effort: missing fields are returned as None.
    """
    soup = BeautifulSoup(html, "lxml")
    result = ParsedHorseDetail(horse_id=horse_id)

    # name from <title> — e.g. "アパッシメント (Appassimento) | 競走馬データ - netkeiba"
    title_tag = soup.find("title")
    if title_tag:
        m = _NAME_RE.search(title_tag.get_text(strip=True))
        if m:
            result.name = m.group(1)

    # sex from <div class="horse_title"> — e.g. "Info: アパッシメント Appassimento 現役　セ4歳　鹿毛"
    horse_title = soup.find(class_="horse_title")
    if horse_title:
        text = horse_title.get_text(" ", strip=True)
        m = _SEX_AGE_RE.search(text)
        if m:
            result.sex = m.group(1)

    # birth_date from <table class="db_prof_table"> — th "生年月日" row
    prof = soup.find("table", class_="db_prof_table")
    if prof:
        for tr in prof.find_all("tr"):
            th = tr.find("th")
            td = tr.find("td")
            if th and td and "生年月日" in th.get_text(strip=True):
                m = _DATE_RE.search(td.get_text(strip=True))
                if m:
                    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    result.birth_date = f"{y:04d}-{mo:02d}-{d:02d}"
                break

    return result
