"""Range ingest CLI: ingest multiple dates in sequence with resume support.

Usage:
    python -m jobs.ingest_range --start 2020-01-01 --end 2024-12-31
    python -m jobs.ingest_range --start 2020-01-01 --end 2024-12-31 --limit-per-day 5
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import sys

import httpx
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.config import load_settings
from core.logging import configure_logging, get_logger
from core.paths import db_path
from db.base import Base
from db.models.scrape_log import ScrapeLog
from db.session import make_engine, session_scope
from jobs.ingest import run_ingest
from scraper import stop_flag
from scraper.cache import clear_misc_cache
from scraper.netkeiba import NetkeibaClient
from scraper.rate_limiter import AsyncRateLimiter
from scraper.robots import RobotsCache
from scraper.stop_flag import ScraperStopped

logger = get_logger(__name__)

_RESULT_URL_PREFIX = "https://db.netkeiba.com/race/"


def is_date_completed(session: Session, date_str: str) -> bool:
    """Return True if the given date has at least one 'ok' scrape_log entry.

    The simplified check: if any ok-status result URL for that date's prefix
    exists in scrape_log, we treat the date as completed and skip it.
    This avoids fetching the calendar again just to count races.
    The date prefix in race IDs is the 8-digit YYYYMMDD portion.
    """
    date_compact = date_str.replace("-", "")
    url_prefix = f"{_RESULT_URL_PREFIX}{date_compact}%"
    count = session.execute(
        select(func.count()).select_from(ScrapeLog).where(
            ScrapeLog.url.like(url_prefix),
            ScrapeLog.status == "ok",
        )
    ).scalar_one()
    return count > 0


def _date_range(start: datetime.date, end: datetime.date) -> list[datetime.date]:
    days = (end - start).days + 1
    return [start + datetime.timedelta(days=i) for i in range(days)]


async def run_range(
    start_str: str,
    end_str: str,
    client: NetkeibaClient,
    engine,
    limit_per_day: int | None = None,
) -> dict[str, int]:
    """Ingest all dates in [start, end], skipping already-completed dates.

    Returns aggregate counters across all dates.
    """
    start = datetime.date.fromisoformat(start_str)
    end = datetime.date.fromisoformat(end_str)
    dates = _date_range(start, end)

    total = {"fetched": 0, "skipped_races": 0, "errors": 0, "dates_skipped": 0, "dates_done": 0}

    for d in dates:
        date_str = d.isoformat()

        if stop_flag.is_stopped():
            logger.warning("Stop flag set; aborting range ingest at %s", date_str)
            break

        # Resume check: skip dates already fully ingested
        with session_scope(engine) as session:
            completed = is_date_completed(session, date_str)

        if completed:
            progress = {"date": date_str, "status": "skipped_already_done"}
            print(json.dumps(progress), flush=True)
            logger.info("Skipping completed date: %s", date_str)
            total["dates_skipped"] += 1
            continue

        logger.info("Ingesting date: %s", date_str)
        try:
            with session_scope(engine) as session:
                counters = await run_ingest(date_str, client, session, limit=limit_per_day)
            total["fetched"] += counters["fetched"]
            total["skipped_races"] += counters["skipped"]
            total["errors"] += counters["errors"]
            total["dates_done"] += 1

            progress = {
                "date": date_str,
                "status": "done",
                "fetched": counters["fetched"],
                "skipped": counters["skipped"],
                "errors": counters["errors"],
            }
            print(json.dumps(progress), flush=True)

            # Bound disk usage during long-running ingests by dropping the
            # one-time-use horse_detail / horse_pedigree / calendar HTML cache
            # after each successful day. Set KEIBA_KEEP_MISC_CACHE=1 to opt out.
            if os.getenv("KEIBA_KEEP_MISC_CACHE", "0") != "1":
                removed = clear_misc_cache()
                if removed:
                    logger.info("Cleared misc cache after %s: %d files", date_str, removed)

        except ScraperStopped:
            logger.warning("Scraper stopped during ingest of %s", date_str)
            break
        except Exception as exc:
            logger.error("Unexpected error ingesting %s: %s", date_str, exc)
            total["errors"] += 1
            progress = {"date": date_str, "status": "error", "message": str(exc)}
            print(json.dumps(progress), flush=True)

    return total


async def main(args: argparse.Namespace) -> int:
    configure_logging()
    engine = make_engine(db_path())
    Base.metadata.create_all(engine)

    settings = load_settings()
    rate_limiter = AsyncRateLimiter(settings)
    robots_cache = RobotsCache(settings.user_agent)

    async with httpx.AsyncClient() as http_client:
        client = NetkeibaClient(rate_limiter, robots_cache, http_client, settings)
        totals = await run_range(
            args.start,
            args.end,
            client,
            engine,
            limit_per_day=args.limit_per_day,
        )

    logger.info(
        "Range ingest complete — dates_done=%d dates_skipped=%d fetched=%d errors=%d",
        totals["dates_done"],
        totals["dates_skipped"],
        totals["fetched"],
        totals["errors"],
    )
    summary = {"summary": totals}
    print(json.dumps(summary), flush=True)
    return 0 if totals["errors"] == 0 else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest a date range of netkeiba race results")
    parser.add_argument("--start", required=True, metavar="YYYY-MM-DD", help="Start date (inclusive)")
    parser.add_argument("--end", required=True, metavar="YYYY-MM-DD", help="End date (inclusive)")
    parser.add_argument(
        "--limit-per-day",
        type=int,
        default=None,
        metavar="N",
        help="Maximum races per day (debug use)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(asyncio.run(main(args)))
