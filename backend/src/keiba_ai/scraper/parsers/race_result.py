"""Parse netkeiba race result page into structured dataclasses.

Target URL:
  https://db.netkeiba.com/race/<race_id>/

Assumed HTML structure (to be verified against real pages in M2 manual QA):
  Race info header:
    <div class="RaceData01">
      芝1600m / 天候:晴 / 馬場:良
    </div>
    <div class="RaceData02">
      <span>東京</span><span>11R</span><span>G1</span>
    </div>

  Results table:
    <table class="race_table_01">
      <thead><tr><th>着順</th><th>馬番</th><th>馬名</th>...></thead>
      <tbody>
        <tr>
          <td>1</td>           <!-- finish_position -->
          <td>5</td>           <!-- post_position -->
          <td><a href="/horse/2019105293/">ホウオウビスケッツ</a></td>
          <td>牡4</td>         <!-- sex + age -->
          <td>57.0</td>        <!-- weight_carried -->
          <td><a href="/jockey/01011/">横山武史</a></td>
          <td>1:33.4</td>      <!-- finish_time -->
          <td>...</td>         <!-- margin -->
          <td>...</td>         <!-- horse_weight / diff -->
          <td>2.8</td>         <!-- odds_win -->
          <td>1</td>           <!-- popularity -->
          <td><a href="/trainer/01096/">田中博康</a></td>
        </tr>
      </tbody>
    </table>

Selectors are intentionally lenient (row count checks, try/except per cell)
so that partial parses still produce useful data.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup, Tag

from keiba_ai.core.logging import get_logger
from keiba_ai.scraper.parsers.payout import parse_payout

logger = get_logger(__name__)

_ID_FROM_HREF = re.compile(r"/(\w+)/(\w+)/?$")
_TIME_RE = re.compile(r"(\d+):(\d+)\.(\d+)")
_WEIGHT_RE = re.compile(r"(\d+)\(([+-]?\d+)\)")
_SURFACE_DIST_RE = re.compile(r"(芝|ダ|障)(\d+)")


class ParseError(Exception):
    pass


@dataclass
class ParsedEntry:
    race_id: str
    horse_id: str
    post_position: int | None = None
    jockey_id: str | None = None
    trainer_id: str | None = None
    weight_carried: float | None = None
    age: int | None = None
    sex: str | None = None
    horse_weight: int | None = None
    horse_weight_diff: int | None = None
    odds_win: float | None = None
    popularity: int | None = None
    finish_position: int | None = None
    finish_time: float | None = None
    margin: str | None = None


@dataclass
class ParsedRaceResult:
    race_id: str
    date: str | None = None
    course: str | None = None
    surface: str | None = None
    distance: int | None = None
    weather: str | None = None
    track_condition: str | None = None
    race_class: str | None = None
    n_runners: int | None = None
    payout_win: int | None = None
    payout_place: str | None = None  # JSON string
    entries: list[ParsedEntry] = field(default_factory=list)


def _extract_id_from_href(href: str, kind: str) -> str | None:
    """Extract entity ID from a netkeiba path like /horse/<id>/ or /jockey/<id>/."""
    m = re.search(rf"/{kind}/(\w+)", href)
    return m.group(1) if m else None


def _parse_time_to_seconds(text: str) -> float | None:
    m = _TIME_RE.search(text)
    if not m:
        return None
    minutes, secs, tenths = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return minutes * 60 + secs + tenths / 10


def _parse_header(soup: BeautifulSoup, result: ParsedRaceResult) -> None:
    """Extract race metadata from the page header area."""
    data01 = soup.find(class_=re.compile(r"RaceData01"))
    if data01:
        text = data01.get_text(" ", strip=True)
        m = _SURFACE_DIST_RE.search(text)
        if m:
            surface_char = m.group(1)
            result.surface = "芝" if surface_char == "芝" else "ダ"
            result.distance = int(m.group(2))
        if "天候:" in text:
            result.weather = text.split("天候:")[1].split()[0].rstrip("/ ")
        if "馬場:" in text:
            result.track_condition = text.split("馬場:")[1].split()[0].rstrip("/ ")

    data02 = soup.find(class_=re.compile(r"RaceData02"))
    if data02:
        spans = data02.find_all("span")
        for span in spans:
            t = span.get_text(strip=True)
            if any(kw in t for kw in ["G1", "G2", "G3", "GI", "GII", "GIII", "条件", "特別", "オープン"]):
                result.race_class = t
            elif not result.course and len(t) in (2, 3) and not t.isdigit():
                result.course = t


def _parse_entries(soup: BeautifulSoup, race_id: str) -> list[ParsedEntry]:
    table = soup.find("table", class_=re.compile(r"race_table_01|RaceTable"))
    if table is None:
        logger.error("Race result table not found — netkeiba page structure may have changed")
        raise ParseError("Race result table not found")

    # Map header column indices
    headers: list[str] = []
    thead = table.find("thead")
    if thead:
        headers = [th.get_text(strip=True) for th in thead.find_all("th")]

    # Column name → index (lenient: use position if header missing)
    COL = {name: idx for idx, name in enumerate(headers)}

    def col(name: str, fallback: int) -> int:
        return COL.get(name, fallback)

    entries: list[ParsedEntry] = []
    for tr in table.find_all("tr")[1:]:  # skip header row
        tds = tr.find_all("td")
        if len(tds) < 5:
            continue
        try:
            entry = _parse_entry_row(tds, race_id, col)
        except Exception as exc:
            logger.warning("Failed to parse entry row: %s", exc)
            continue
        if entry is not None:
            entries.append(entry)
    return entries


def _parse_entry_row(
    tds: list[Tag],
    race_id: str,
    col: "dict[str, int] | function",  # type: ignore[valid-type]
) -> ParsedEntry | None:
    def safe_text(idx: int) -> str:
        return tds[idx].get_text(strip=True) if idx < len(tds) else ""

    def safe_int(idx: int) -> int | None:
        t = safe_text(idx)
        try:
            return int(t)
        except (ValueError, TypeError):
            return None

    def safe_float(idx: int) -> float | None:
        t = safe_text(idx)
        try:
            return float(t)
        except (ValueError, TypeError):
            return None

    # Column positions (assumed; adjust after M2 manual QA)
    I_FINISH = 0
    I_POST = 1
    I_HORSE = 2
    I_SEX_AGE = 3
    I_WEIGHT_CARRIED = 4
    I_JOCKEY = 5
    I_TIME = 6
    I_MARGIN = 7
    I_HORSE_WEIGHT = 8
    I_ODDS = 9
    I_POPULARITY = 10
    I_TRAINER = 11

    horse_link = tds[I_HORSE].find("a", href=True) if I_HORSE < len(tds) else None
    if horse_link is None:
        return None
    horse_id = _extract_id_from_href(horse_link["href"], "horse")
    if horse_id is None:
        return None

    entry = ParsedEntry(race_id=race_id, horse_id=horse_id)
    entry.finish_position = safe_int(I_FINISH)
    entry.post_position = safe_int(I_POST)

    sex_age = safe_text(I_SEX_AGE)
    if sex_age:
        entry.sex = sex_age[0] if sex_age[0] in ("牡", "牝", "セ") else None
        try:
            entry.age = int(sex_age[1:])
        except (ValueError, IndexError):
            pass

    entry.weight_carried = safe_float(I_WEIGHT_CARRIED)

    jockey_link = tds[I_JOCKEY].find("a", href=True) if I_JOCKEY < len(tds) else None
    if jockey_link:
        entry.jockey_id = _extract_id_from_href(jockey_link["href"], "jockey")

    entry.finish_time = _parse_time_to_seconds(safe_text(I_TIME))
    entry.margin = safe_text(I_MARGIN) or None

    hw_text = safe_text(I_HORSE_WEIGHT)
    m = _WEIGHT_RE.search(hw_text)
    if m:
        entry.horse_weight = int(m.group(1))
        entry.horse_weight_diff = int(m.group(2))

    entry.odds_win = safe_float(I_ODDS)
    entry.popularity = safe_int(I_POPULARITY)

    trainer_link = tds[I_TRAINER].find("a", href=True) if I_TRAINER < len(tds) else None
    if trainer_link:
        entry.trainer_id = _extract_id_from_href(trainer_link["href"], "trainer")

    return entry


def parse_race_result(html: str, race_id: str) -> ParsedRaceResult:
    """Parse a race result page into ParsedRaceResult.

    Raises:
        ParseError: If the results table is missing entirely.
    """
    soup = BeautifulSoup(html, "lxml")
    result = ParsedRaceResult(race_id=race_id)

    _parse_header(soup, result)
    result.entries = _parse_entries(soup, race_id)
    result.n_runners = len(result.entries)

    payout_win, payout_place = parse_payout(html)
    result.payout_win = payout_win
    result.payout_place = json.dumps(payout_place, ensure_ascii=False) if payout_place else None

    return result
