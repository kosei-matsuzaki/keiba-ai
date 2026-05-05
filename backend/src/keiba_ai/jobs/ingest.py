"""Daily race ingest job.

Usage:
    python -m keiba_ai.jobs.ingest --date 2024-12-28
    python -m keiba_ai.jobs.ingest --date 2024-12-28 --limit 3
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import os

import httpx
import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from keiba_ai.core.config import load_settings
from keiba_ai.core.logging import configure_logging, get_logger
from keiba_ai.core.paths import db_path
from keiba_ai.db.base import Base
from keiba_ai.db.models.entry import Entry
from keiba_ai.db.models.horse import Horse
from keiba_ai.db.models.jockey import Jockey
from keiba_ai.db.models.payout import Payout
from keiba_ai.db.models.race import Race
from keiba_ai.db.models.scrape_log import ScrapeLog
from keiba_ai.db.models.trainer import Trainer
from keiba_ai.db.session import make_engine, session_scope
from keiba_ai.scraper import cache as cache_module
from keiba_ai.scraper import stop_flag
from keiba_ai.scraper.netkeiba import NetkeibaClient
from keiba_ai.scraper.parsers.horse_detail import parse_horse_detail
from keiba_ai.scraper.parsers.horse_pedigree import parse_horse_pedigree
from keiba_ai.scraper.parsers.payout import parse_payouts
from keiba_ai.scraper.parsers.race_calendar import parse_race_ids_from_calendar
from keiba_ai.scraper.parsers.race_result import ParsedRaceResult, parse_race_result
from keiba_ai.scraper.rate_limiter import AsyncRateLimiter
from keiba_ai.scraper.robots import RobotsCache
from keiba_ai.scraper.stop_flag import ScraperStopped

logger = get_logger(__name__)

_CALENDAR_URL = "https://db.netkeiba.com/race/list/{date}/"
_RESULT_URL = "https://db.netkeiba.com/race/{race_id}/"


def _already_scraped(session: Session, url: str) -> bool:
    row = session.execute(
        select(ScrapeLog).where(ScrapeLog.url == url, ScrapeLog.status == "ok").limit(1)
    ).first()
    return row is not None


def _record_scrape_log(
    session: Session,
    url: str,
    status: str,
    content_hash: str | None = None,
) -> None:
    fetched_at = datetime.datetime.now(datetime.UTC).isoformat()
    session.add(ScrapeLog(url=url, fetched_at=fetched_at, status=status, content_hash=content_hash))


def _upsert_race(session: Session, result: ParsedRaceResult) -> None:
    # SQLite upsert: INSERT OR REPLACE semantics via dialect-specific insert
    stmt = sqlite_insert(Race).values(
        race_id=result.race_id,
        date=result.date or "",
        course=result.course or "",
        surface=result.surface or "",
        distance=result.distance or 0,
        weather=result.weather,
        track_condition=result.track_condition,
        race_class=result.race_class,
        n_runners=result.n_runners,
        payout_win=result.payout_win,
        payout_place=result.payout_place,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["race_id"],
        set_={
            "date": stmt.excluded.date,
            "course": stmt.excluded.course,
            "surface": stmt.excluded.surface,
            "distance": stmt.excluded.distance,
            "weather": stmt.excluded.weather,
            "track_condition": stmt.excluded.track_condition,
            "race_class": stmt.excluded.race_class,
            "n_runners": stmt.excluded.n_runners,
            "payout_win": stmt.excluded.payout_win,
            "payout_place": stmt.excluded.payout_place,
        },
    )
    session.execute(stmt)


async def _ensure_masters(
    session: Session,
    result: ParsedRaceResult,
    client: NetkeibaClient | None = None,
) -> None:
    """Upsert horses, jockeys, trainers referenced by entries.

    For new horses (no existing name in DB), additionally fetches the horse
    detail page and pedigree page to populate name, sex, birth_date, sire, dam.
    Fetching is best-effort: failures log a warning but do not fail the ingest.
    """
    horses_seen: set[str] = set()
    jockeys_seen: set[str] = set()
    trainers_seen: set[str] = set()

    for e in result.entries:
        if e.horse_id and e.horse_id not in horses_seen:
            horses_seen.add(e.horse_id)

            # Check whether horse already has a name to decide if detail fetch is needed.
            existing = session.get(Horse, e.horse_id)
            need_detail = client is not None and (existing is None or existing.name is None)

            horse_kwargs: dict[str, object] = {"horse_id": e.horse_id, "name": e.horse_name}

            if need_detail:
                try:
                    detail_html = await client.fetch(
                        f"https://db.netkeiba.com/horse/{e.horse_id}/"
                    )
                    detail = parse_horse_detail(detail_html, e.horse_id)
                    if detail.name:
                        horse_kwargs["name"] = detail.name
                    horse_kwargs["sex"] = detail.sex
                    horse_kwargs["birth_date"] = detail.birth_date
                except Exception as exc:
                    logger.warning("horse_detail fetch failed for %s: %s", e.horse_id, exc)

                try:
                    ped_html = await client.fetch(
                        f"https://db.netkeiba.com/horse/ped/{e.horse_id}/"
                    )
                    ped = parse_horse_pedigree(ped_html, e.horse_id)
                    horse_kwargs["sire"] = ped.sire_name
                    horse_kwargs["dam"] = ped.dam_name
                except Exception as exc:
                    logger.warning("horse_pedigree fetch failed for %s: %s", e.horse_id, exc)

            stmt = sqlite_insert(Horse).values(**horse_kwargs)
            stmt = stmt.on_conflict_do_update(
                index_elements=["horse_id"],
                set_={
                    "name": sa.func.coalesce(stmt.excluded.name, Horse.name),
                    "sex": sa.func.coalesce(stmt.excluded.sex, Horse.sex),
                    "birth_date": sa.func.coalesce(stmt.excluded.birth_date, Horse.birth_date),
                    "sire": sa.func.coalesce(stmt.excluded.sire, Horse.sire),
                    "dam": sa.func.coalesce(stmt.excluded.dam, Horse.dam),
                },
            )
            session.execute(stmt)

        if e.jockey_id and e.jockey_id not in jockeys_seen:
            stmt = sqlite_insert(Jockey).values(jockey_id=e.jockey_id, name=e.jockey_name)
            stmt = stmt.on_conflict_do_update(
                index_elements=["jockey_id"],
                set_={"name": sa.func.coalesce(stmt.excluded.name, Jockey.name)},
            )
            session.execute(stmt)
            jockeys_seen.add(e.jockey_id)

        if e.trainer_id and e.trainer_id not in trainers_seen:
            stmt = sqlite_insert(Trainer).values(trainer_id=e.trainer_id, name=e.trainer_name)
            stmt = stmt.on_conflict_do_update(
                index_elements=["trainer_id"],
                set_={"name": sa.func.coalesce(stmt.excluded.name, Trainer.name)},
            )
            session.execute(stmt)
            trainers_seen.add(e.trainer_id)


def _insert_entries(session: Session, result: ParsedRaceResult) -> None:
    # Remove existing entries for this race to avoid duplicates on re-ingest
    session.execute(
        Entry.__table__.delete().where(Entry.race_id == result.race_id)
    )
    for e in result.entries:
        session.add(Entry(
            race_id=e.race_id,
            horse_id=e.horse_id,
            post_position=e.post_position,
            jockey_id=e.jockey_id,
            trainer_id=e.trainer_id,
            weight_carried=e.weight_carried,
            age=e.age,
            sex=e.sex,
            horse_weight=e.horse_weight,
            horse_weight_diff=e.horse_weight_diff,
            odds_win=e.odds_win,
            popularity=e.popularity,
            finish_position=e.finish_position,
            finish_time=e.finish_time,
            margin=e.margin,
            agari_3f=e.agari_3f,
            passing=e.passing,
        ))


def _upsert_payouts(session: Session, race_id: str, html: str) -> int:
    """払戻テーブルを DELETE → INSERT で冪等に更新する。

    既存 race の再 ingest 時に重複行が生じないよう、対象 race_id の全
    payouts 行を先に削除してから parse_payouts() 結果を INSERT する。

    Returns:
        挿入した行数（0 の場合は払戻データが取得できなかったことを意味する）
    """
    session.execute(
        Payout.__table__.delete().where(Payout.race_id == race_id)
    )
    payout_rows = parse_payouts(html)
    for row in payout_rows:
        session.add(Payout(
            race_id=race_id,
            bet_type=row.bet_type,
            combo=row.combo,
            amount=row.amount,
            popularity=row.popularity,
        ))
    return len(payout_rows)


async def run_ingest(
    date_str: str,
    client: NetkeibaClient,
    session: Session,
    limit: int | None = None,
) -> dict[str, int]:
    """Core ingest logic; returns summary counters."""
    counters = {"fetched": 0, "skipped": 0, "errors": 0}

    # Step 1: fetch calendar
    calendar_url = _CALENDAR_URL.format(date=date_str.replace("-", ""))
    logger.info("Fetching calendar: %s", calendar_url)
    calendar_html = await client.fetch(calendar_url, cache_max_age_hours=24)
    # Central JRA only by default. Set KEIBA_INCLUDE_NAR=1 to also ingest 地方.
    include_nar = os.getenv("KEIBA_INCLUDE_NAR", "0") == "1"
    race_ids = parse_race_ids_from_calendar(calendar_html, include_nar=include_nar)

    if limit is not None:
        race_ids = race_ids[:limit]
        logger.info("Limiting to %d races (--limit)", limit)

    logger.info("Found %d race IDs for %s", len(race_ids), date_str)

    for race_id in race_ids:
        if stop_flag.is_stopped():
            raise ScraperStopped("stop flag set during race loop")

        result_url = _RESULT_URL.format(race_id=race_id)

        if _already_scraped(session, result_url):
            logger.debug("Skipping already-scraped race: %s", race_id)
            counters["skipped"] += 1
            continue

        try:
            html = await client.fetch(result_url, cache_max_age_hours=24 * 30)
            parsed = parse_race_result(html, race_id)
            parsed.date = date_str

            _upsert_race(session, parsed)
            await _ensure_masters(session, parsed, client=client)
            _insert_entries(session, parsed)
            n_payouts = _upsert_payouts(session, race_id, html)
            _record_scrape_log(session, result_url, "ok", cache_module.content_hash(html))
            session.commit()

            counters["fetched"] += 1
            logger.info(
                "Ingested race %s (%d entries, %d payouts)",
                race_id, len(parsed.entries), n_payouts,
            )

        except ScraperStopped:
            raise
        except Exception as exc:
            logger.error("Error ingesting race %s: %s", race_id, exc)
            session.rollback()
            try:
                _record_scrape_log(session, result_url, "error")
                session.commit()
            except Exception:
                session.rollback()
            counters["errors"] += 1

    return counters


async def main(args: argparse.Namespace) -> int:
    configure_logging()
    engine = make_engine(db_path())

    # Create all tables (idempotent; Alembic manages schema in production,
    # but create_all ensures the CLI works even without running migrations first)
    Base.metadata.create_all(engine)

    rate_limiter = AsyncRateLimiter(load_settings())
    robots_cache = RobotsCache(load_settings().user_agent)

    async with httpx.AsyncClient() as http_client:
        client = NetkeibaClient(rate_limiter, robots_cache, http_client, load_settings())
        with session_scope(engine) as session:
            try:
                counters = await run_ingest(args.date, client, session, limit=args.limit)
            except ScraperStopped:
                logger.warning("Scraper stopped by stop flag")
                return 1

    logger.info(
        "Ingest complete — fetched=%d skipped=%d errors=%d",
        counters["fetched"], counters["skipped"], counters["errors"],
    )
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
