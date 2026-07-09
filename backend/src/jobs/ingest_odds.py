"""Backfill confirmed combination odds into odds.db.

Fetches netkeiba's ``api_get_jra_odds.html`` JSON for every race already in
keiba.db and stores the full per-combo odds (all bet types) in the separate
odds.db (see db/odds_db.py). This is the data source that lets backtests price
*losing* combos at their real market odds instead of the Plackett-Luce estimate
in ai/bet_odds.py.

Design notes:
  - **Read-only on keiba.db.** This job only SELECTs the race list; it never
    writes the main DB, so it cannot corrupt the irreplaceable data.
  - **Newest race first** by default — the most-used recent seasons land first,
    and the whole thing is resumable, so a multi-day run can be stopped/restarted
    freely (Ctrl-C, KEIBA_SCRAPER_STOP=1, or the UI stop flag).
  - **Resume granularity = bet type.** A row in race_odds means that bet type is
    done; a partially-fetched race continues from where it stopped. Races the API
    has no odds for get a ``__none__`` sentinel so they are not re-probed.
  - 7 requests/race (types 1,3,4,5,6,7,8) → ~260k requests over 2015→. Politeness
    is the existing AsyncRateLimiter; expect a multi-day wall-clock run.

CLI:
    python -m jobs.ingest_odds                         # all races, newest first
    python -m jobs.ingest_odds --start 2023-01-01      # date-bounded (inclusive)
    python -m jobs.ingest_odds --start 2024-01-01 --end 2024-12-31
    python -m jobs.ingest_odds --limit 100             # debug: first N races
    python -m jobs.ingest_odds --oldest-first
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import httpx
from sqlalchemy import select

from core.config import load_settings
from core.logging import configure_logging, get_logger
from core.paths import db_path
from db.models.race import Race
from db.odds_db import (
    fetched_bet_types,
    init_odds_db,
    make_odds_engine,
    odds_session_scope,
    upsert_race_odds,
)
from db.session import make_engine, session_scope
from scraper.netkeiba import NetkeibaClient
from scraper.parsers.odds import parse_live_win_odds, parse_odds_payload
from scraper.rate_limiter import AsyncRateLimiter
from scraper.robots import RobotsCache
from scraper.stop_flag import ScraperStopped

logger = get_logger(__name__)

_ODDS_API = "https://race.netkeiba.com/api/api_get_jra_odds.html"

# (netkeiba type param, bet types that call returns). type=1 bundles 単+複.
_FETCH_UNITS: list[tuple[int, tuple[str, ...]]] = [
    (1, ("単勝", "複勝")),
    (3, ("枠連",)),
    (4, ("馬連",)),
    (5, ("ワイド",)),
    (6, ("馬単",)),
    (7, ("三連複",)),
    (8, ("三連単",)),
]

_ALL_BET_TYPES: frozenset[str] = frozenset(
    bt for _, bts in _FETCH_UNITS for bt in bts
)

# Marker stored when the API returns no odds at all for a race (cancelled /
# non-JRA), so resume skips it instead of re-probing all 7 types every run.
_SENTINEL = "__none__"

_PROGRESS_EVERY = 50


def _odds_url(race_id: str, type_code: int) -> str:
    return (
        f"{_ODDS_API}?pid=api_get_jra_odds&race_id={race_id}"
        f"&type={type_code}&action=update"
    )


def _select_race_ids(
    session,
    start: str | None,
    end: str | None,
    oldest_first: bool,
    limit: int | None,
) -> list[str]:
    stmt = select(Race.race_id)
    if start is not None:
        stmt = stmt.where(Race.date >= start)
    if end is not None:
        stmt = stmt.where(Race.date <= end)
    stmt = stmt.order_by(Race.date.asc() if oldest_first else Race.date.desc())
    if limit is not None:
        stmt = stmt.limit(limit)
    return [r[0] for r in session.execute(stmt).all()]


async def _ingest_race(
    client: NetkeibaClient,
    odds_engine,
    race_id: str,
) -> str:
    """Fetch+store all missing bet types for one race.

    Returns a status string: "done" | "resumed_skip" | "no_odds" | "partial".
    Raises ScraperStopped to abort the whole run.
    """
    with odds_session_scope(odds_engine) as s:
        # confirmed_only: 発走前ライブ行 (is_confirmed=0) は resume skip せず、確定値で
        # 上書きするため "未取得" 扱いにする。
        have = fetched_bet_types(s, race_id, confirmed_only=True)

    if _SENTINEL in have or have >= _ALL_BET_TYPES:
        return "resumed_skip"

    stored_any = False
    for type_code, bet_types in _FETCH_UNITS:
        if all(bt in have for bt in bet_types):
            continue

        raw = await client.fetch(
            _odds_url(race_id, type_code),
            use_cache=False,
            write_to_cache=False,
        )
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Non-JSON odds response for %s type=%d", race_id, type_code)
            continue

        official_dt, parsed = parse_odds_payload(payload)

        if not parsed:
            # type=1 empty => race has no odds in the feed at all. Mark & stop.
            if type_code == 1 and not stored_any:
                with odds_session_scope(odds_engine) as s:
                    upsert_race_odds(s, race_id, _SENTINEL, official_dt, {})
                return "no_odds"
            continue

        with odds_session_scope(odds_engine) as s:
            for bet_type, combos in parsed.items():
                upsert_race_odds(s, race_id, bet_type, official_dt, combos)
        stored_any = True

    return "done" if stored_any else "partial"


async def ingest_live_odds_for_race(
    client: NetkeibaClient,
    odds_engine,
    race_id: str,
) -> dict[int, tuple[float | None, int | None]]:
    """Fetch **live** (発走前) odds for ALL bet types and store them in odds.db.

    run_backfill の確定オッズ版に対する「当日ライブ版」。違いは:
      - ``parse_odds_payload(accept_live=True)`` で status="middle" も受理する。
      - resume-skip しない（オッズは変動するので毎回上書きして最新化する）。
      - ``__none__`` sentinel は書かない（未公開のレースは後で再取得すれば埋まる）。

    出馬表 ingest から呼ぶことで、entries.odds_win/popularity（type=1 由来）と
    odds.db の全 combo 実オッズを 1 経路で取得する。ネットワーク/JSON 失敗は
    per-type で握りつぶし best-effort（ScraperStopped のみ伝播）。

    Returns:
        ``{馬番: (単勝オッズ or None, 人気 or None)}``（type=1 由来、entries 補完用）。
    """
    win_map: dict[int, tuple[float | None, int | None]] = {}
    for type_code, _bet_types in _FETCH_UNITS:
        try:
            raw = await client.fetch(
                _odds_url(race_id, type_code), use_cache=False, write_to_cache=False
            )
        except ScraperStopped:
            raise
        except Exception as exc:  # noqa: BLE001 — best-effort、他の type は続行
            logger.warning("Live odds fetch failed for %s type=%d: %s", race_id, type_code, exc)
            continue

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Non-JSON live odds for %s type=%d", race_id, type_code)
            continue

        if type_code == 1:
            win_map = parse_live_win_odds(payload)

        # status="result"（発走後に再取得した場合）は確定として保存し、確定バックフィルが
        # 後で再取得しないようにする。発走前ライブ ("middle") は is_confirmed=0。
        is_confirmed = isinstance(payload, dict) and payload.get("status") == "result"
        official_dt, parsed = parse_odds_payload(payload, accept_live=True)
        if not parsed:
            continue
        with odds_session_scope(odds_engine) as s:
            for bet_type, combos in parsed.items():
                upsert_race_odds(
                    s, race_id, bet_type, official_dt, combos, is_confirmed=is_confirmed
                )

    return win_map


async def run_backfill(
    client: NetkeibaClient,
    keiba_engine,
    odds_engine,
    *,
    start: str | None,
    end: str | None,
    oldest_first: bool,
    limit: int | None,
) -> dict[str, int]:
    with session_scope(keiba_engine) as session:
        race_ids = _select_race_ids(session, start, end, oldest_first, limit)

    # keiba.db is only needed for the race-id list above. Drop the engine's
    # pooled connection now so the multi-hour fetch loop below never keeps an
    # open keiba.db handle: on /mnt/c DrvFs, SQLite cannot be opened
    # concurrently, so a lingering connection makes a parallel train_nn /
    # experiment fail at `PRAGMA journal_mode=WAL` ("unable to open database
    # file"). From here on the loop only touches odds.db.
    keiba_engine.dispose()

    total = len(race_ids)
    logger.info("Odds backfill: %d races queued", total)
    counters = {"done": 0, "resumed_skip": 0, "no_odds": 0, "partial": 0, "errors": 0}

    for idx, race_id in enumerate(race_ids, start=1):
        try:
            status = await _ingest_race(client, odds_engine, race_id)
            counters[status] += 1
        except ScraperStopped:
            logger.warning("Stop flag set; aborting odds backfill at %s", race_id)
            break
        except Exception as exc:  # noqa: BLE001 — keep going, log, count
            counters["errors"] += 1
            logger.error("Error ingesting odds for %s: %s", race_id, exc)
            print(
                json.dumps({"race_id": race_id, "status": "error", "message": str(exc)}),
                flush=True,
            )

        if idx % _PROGRESS_EVERY == 0 or idx == total:
            print(
                json.dumps(
                    {"progress": idx, "total": total, "race_id": race_id, **counters}
                ),
                flush=True,
            )

    return counters


async def main(args: argparse.Namespace) -> int:
    configure_logging()
    keiba_engine = make_engine(db_path())
    odds_engine = make_odds_engine()
    init_odds_db(odds_engine)

    settings = load_settings()
    rate_limiter = AsyncRateLimiter(settings)
    robots_cache = RobotsCache(settings.user_agent)

    async with httpx.AsyncClient() as http_client:
        client = NetkeibaClient(rate_limiter, robots_cache, http_client, settings)
        counters = await run_backfill(
            client,
            keiba_engine,
            odds_engine,
            start=args.start,
            end=args.end,
            oldest_first=args.oldest_first,
            limit=args.limit,
        )

    logger.info(
        "Odds backfill complete — done=%d skipped=%d no_odds=%d partial=%d errors=%d",
        counters["done"],
        counters["resumed_skip"],
        counters["no_odds"],
        counters["partial"],
        counters["errors"],
    )
    print(json.dumps({"summary": counters}), flush=True)
    return 0 if counters["errors"] == 0 else 1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill confirmed combo odds into odds.db")
    p.add_argument("--start", metavar="YYYY-MM-DD", default=None, help="Earliest race date (inclusive)")
    p.add_argument("--end", metavar="YYYY-MM-DD", default=None, help="Latest race date (inclusive)")
    p.add_argument("--oldest-first", action="store_true", help="Process oldest races first (default: newest)")
    p.add_argument("--limit", type=int, default=None, metavar="N", help="Max races (debug)")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main(_parse_args())))
