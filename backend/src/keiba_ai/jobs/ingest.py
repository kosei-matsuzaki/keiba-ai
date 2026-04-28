"""Daily race ingest job.

Usage:
    python -m keiba_ai.jobs.ingest --date 2024-12-28
    python -m keiba_ai.jobs.ingest --date 2024-12-28 --limit 3
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import sqlite3
from dataclasses import asdict

import httpx

from keiba_ai.core.config import load_settings
from keiba_ai.core.logging import configure_logging, get_logger
from keiba_ai.core.paths import db_path
from keiba_ai.db.inline_schema import init_db
from keiba_ai.db.session import connect, transaction
from keiba_ai.scraper import cache as cache_module
from keiba_ai.scraper import stop_flag
from keiba_ai.scraper.netkeiba import NetkeibaClient
from keiba_ai.scraper.parsers.race_calendar import parse_race_ids_from_calendar
from keiba_ai.scraper.parsers.race_result import ParsedRaceResult, parse_race_result
from keiba_ai.scraper.rate_limiter import AsyncRateLimiter
from keiba_ai.scraper.robots import RobotsCache
from keiba_ai.scraper.stop_flag import ScraperStopped

logger = get_logger(__name__)

_CALENDAR_URL = "https://race.netkeiba.com/top/race_list.html?kaisai_date={date}"
_RESULT_URL = "https://db.netkeiba.com/race/{race_id}/"


def _already_scraped(conn: sqlite3.Connection, url: str) -> bool:
    row = conn.execute(
        "SELECT id FROM scrape_log WHERE url = ? AND status = 'ok' LIMIT 1", (url,)
    ).fetchone()
    return row is not None


def _record_scrape_log(
    conn: sqlite3.Connection,
    url: str,
    status: str,
    content_hash: str | None = None,
) -> None:
    fetched_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO scrape_log (url, fetched_at, status, content_hash) VALUES (?, ?, ?, ?)",
        (url, fetched_at, status, content_hash),
    )


def _upsert_race(conn: sqlite3.Connection, result: ParsedRaceResult) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO races
            (race_id, date, course, surface, distance, weather, track_condition,
             race_class, n_runners, payout_win, payout_place)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.race_id,
            result.date or "",
            result.course or "",
            result.surface or "",
            result.distance or 0,
            result.weather,
            result.track_condition,
            result.race_class,
            result.n_runners,
            result.payout_win,
            result.payout_place,
        ),
    )


def _insert_entries(conn: sqlite3.Connection, result: ParsedRaceResult) -> None:
    # Remove existing entries for this race to avoid duplicates on re-ingest
    conn.execute("DELETE FROM entries WHERE race_id = ?", (result.race_id,))
    for e in result.entries:
        conn.execute(
            """
            INSERT INTO entries
                (race_id, horse_id, post_position, jockey_id, trainer_id,
                 weight_carried, age, sex, horse_weight, horse_weight_diff,
                 odds_win, popularity, finish_position, finish_time, margin)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                e.race_id, e.horse_id, e.post_position, e.jockey_id, e.trainer_id,
                e.weight_carried, e.age, e.sex, e.horse_weight, e.horse_weight_diff,
                e.odds_win, e.popularity, e.finish_position, e.finish_time, e.margin,
            ),
        )


async def run_ingest(
    date_str: str,
    client: NetkeibaClient,
    conn: sqlite3.Connection,
    limit: int | None = None,
) -> dict[str, int]:
    """Core ingest logic; returns summary counters."""
    counters = {"fetched": 0, "skipped": 0, "errors": 0}

    # Step 1: fetch calendar
    calendar_url = _CALENDAR_URL.format(date=date_str.replace("-", ""))
    logger.info("Fetching calendar: %s", calendar_url)
    calendar_html = await client.fetch(calendar_url, cache_max_age_hours=24)
    race_ids = parse_race_ids_from_calendar(calendar_html)

    if limit is not None:
        race_ids = race_ids[:limit]
        logger.info("Limiting to %d races (--limit)", limit)

    logger.info("Found %d race IDs for %s", len(race_ids), date_str)

    for race_id in race_ids:
        if stop_flag.is_stopped():
            raise ScraperStopped("stop flag set during race loop")

        result_url = _RESULT_URL.format(race_id=race_id)

        if _already_scraped(conn, result_url):
            logger.debug("Skipping already-scraped race: %s", race_id)
            counters["skipped"] += 1
            continue

        try:
            html = await client.fetch(result_url, cache_max_age_hours=24 * 30)
            parsed = parse_race_result(html, race_id)
            parsed.date = date_str

            with transaction(conn):
                _upsert_race(conn, parsed)
                _insert_entries(conn, parsed)
                _record_scrape_log(conn, result_url, "ok", cache_module.content_hash(html))

            counters["fetched"] += 1
            logger.info("Ingested race %s (%d entries)", race_id, len(parsed.entries))

        except ScraperStopped:
            raise
        except Exception as exc:
            logger.error("Error ingesting race %s: %s", race_id, exc)
            try:
                with transaction(conn):
                    _record_scrape_log(conn, result_url, "error")
            except Exception:
                pass
            counters["errors"] += 1

    return counters


async def main(args: argparse.Namespace) -> int:
    configure_logging()
    settings = load_settings()
    db = connect(db_path())
    init_db(db)

    rate_limiter = AsyncRateLimiter(settings)
    robots_cache = RobotsCache(settings.user_agent)

    async with httpx.AsyncClient() as http_client:
        client = NetkeibaClient(rate_limiter, robots_cache, http_client, settings)
        try:
            counters = await run_ingest(args.date, client, db, limit=args.limit)
        except ScraperStopped:
            logger.warning("Scraper stopped by stop flag")
            return 1

    logger.info(
        "Ingest complete — fetched=%d skipped=%d errors=%d",
        counters["fetched"], counters["skipped"], counters["errors"],
    )
    db.close()
    return 0 if counters["errors"] == 0 else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest netkeiba race results for a given date")
    parser.add_argument(
        "--date",
        required=True,
        metavar="YYYY-MM-DD",
        help="Race date to ingest (e.g. 2024-12-28)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of races to fetch (debug use)",
    )
    return parser.parse_args()


def cli_main() -> int:
    args = _parse_args()
    return asyncio.run(main(args))


if __name__ == "__main__":
    raise SystemExit(cli_main())
