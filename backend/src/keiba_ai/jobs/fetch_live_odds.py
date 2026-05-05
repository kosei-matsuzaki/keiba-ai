"""当日リアルタイム連系オッズ取得ジョブ。

Usage:
    uv run python -m keiba_ai.jobs.fetch_live_odds --race-id 202506050911
    uv run python -m keiba_ai.jobs.fetch_live_odds --race-ids 202506050911,202506050912
    uv run python -m keiba_ai.jobs.fetch_live_odds --race-id 202506050911 --types b1,b4,b5

フロー (1 race):
  1. 指定 type (b1/b4/b5/b6/b7/b8) ごとにオッズページを fetch
  2. 各 HTML を券種別パーサで LiveOddsRow リストに変換
  3. live_odds テーブルへ DELETE → INSERT (upsert 相当)

robots / stop_flag / rate_limiter は既存 ingest ジョブの流儀に準拠。
"""

from __future__ import annotations

import argparse
import asyncio
import datetime

import httpx
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from keiba_ai.core.config import load_settings
from keiba_ai.core.logging import configure_logging, get_logger
from keiba_ai.core.paths import db_path
from keiba_ai.db.base import Base
from keiba_ai.db.models.live_odds import LiveOdds
from keiba_ai.db.models.scrape_log import ScrapeLog
from keiba_ai.db.session import make_engine, session_scope
from keiba_ai.scraper import stop_flag
from keiba_ai.scraper.netkeiba import NetkeibaClient
from keiba_ai.scraper.parsers.odds import (
    LiveOddsRow,
    parse_sanrenpuku_odds,
    parse_sanrentan_odds,
    parse_tan_fuku_odds,
    parse_umaren_odds,
    parse_umatan_odds,
    parse_wakuren_odds,
    parse_wide_odds,
)
from keiba_ai.scraper.rate_limiter import AsyncRateLimiter
from keiba_ai.scraper.robots import RobotsCache
from keiba_ai.scraper.stop_flag import ScraperStopped

logger = get_logger(__name__)

_ODDS_URL = "https://race.netkeiba.com/odds/index.html?race_id={race_id}&type={odds_type}"

# 券種 type コード → (パーサ関数, cache_max_age_hours)
# オッズは頻繁に変動するため短い TTL を使う（30 分）
_TYPE_PARSERS: dict[str, tuple] = {
    "b1": (parse_tan_fuku_odds, 0.5),
    "b3": (parse_wakuren_odds, 0.5),
    "b4": (parse_umaren_odds, 0.5),
    "b5": (parse_wide_odds, 0.5),
    "b6": (parse_umatan_odds, 0.5),
    "b7": (parse_sanrenpuku_odds, 0.5),
    "b8": (parse_sanrentan_odds, 0.5),
}

_DEFAULT_TYPES = ["b1", "b4", "b5", "b6", "b7", "b8"]


def _upsert_live_odds(
    session: Session,
    race_id: str,
    bet_type: str,
    rows: list[LiveOddsRow],
    fetched_at: str,
) -> int:
    """bet_type 単位で live_odds を DELETE → INSERT する。

    DELETE → INSERT は既存 odds をすべて最新値で置き換える方式。
    bet_type ごとに atomic に実行することで、一部 type の fetch 失敗が
    他 type のデータを消さないように範囲を限定する。

    Returns:
        挿入した行数
    """
    session.execute(
        # SQLAlchemy Core DELETE で bet_type 単位削除
        LiveOdds.__table__.delete().where(
            LiveOdds.race_id == race_id,
            LiveOdds.bet_type == bet_type,
        )
    )

    if not rows:
        return 0

    insert_count = 0
    for row in rows:
        stmt = sqlite_insert(LiveOdds).values(
            race_id=race_id,
            bet_type=row.bet_type,
            combo=row.combo,
            odds=row.odds,
            odds_max=row.odds_max,
            popularity=row.popularity,
            fetched_at=fetched_at,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["race_id", "bet_type", "combo"],
            set_={
                "odds": stmt.excluded.odds,
                "odds_max": stmt.excluded.odds_max,
                "popularity": stmt.excluded.popularity,
                "fetched_at": stmt.excluded.fetched_at,
            },
        )
        session.execute(stmt)
        insert_count += 1

    return insert_count


def _record_scrape_log(
    session: Session,
    url: str,
    status: str,
) -> None:
    fetched_at = datetime.datetime.now(datetime.UTC).isoformat()
    session.add(ScrapeLog(url=url, fetched_at=fetched_at, status=status, content_hash=None))


async def fetch_odds_for_race(
    race_id: str,
    types: list[str],
    client: NetkeibaClient,
    session: Session,
) -> dict[str, int]:
    """単一レースのオッズを指定 type 分 fetch して ingest する。

    Returns:
        {"fetched": N, "skipped": N, "errors": N}
    """
    counters = {"fetched": 0, "skipped": 0, "errors": 0}
    fetched_at = datetime.datetime.now(datetime.UTC).isoformat()

    for odds_type in types:
        if stop_flag.is_stopped():
            raise ScraperStopped("stop flag set during fetch_live_odds")

        parser_fn, cache_max_age = _TYPE_PARSERS.get(odds_type, (None, 0.5))
        if parser_fn is None:
            logger.warning("Unknown odds type: %s — skipping", odds_type)
            counters["skipped"] += 1
            continue

        url = _ODDS_URL.format(race_id=race_id, odds_type=odds_type)

        try:
            html = await client.fetch(url, cache_max_age_hours=cache_max_age)
            rows: list[LiveOddsRow] = parser_fn(html)

            # bet_type は rows から収集（b1 は 単勝/複勝 が混在）
            bet_types_in_rows = {r.bet_type for r in rows}
            for bet_type in bet_types_in_rows:
                typed_rows = [r for r in rows if r.bet_type == bet_type]
                count = _upsert_live_odds(session, race_id, bet_type, typed_rows, fetched_at)
                logger.debug(
                    "Ingested %d live_odds rows for %s / %s (type=%s)",
                    count, race_id, bet_type, odds_type,
                )

            _record_scrape_log(session, url, "ok")
            session.commit()
            counters["fetched"] += 1
            logger.info(
                "Fetched live odds %s type=%s (%d rows)",
                race_id, odds_type, len(rows),
            )

        except ScraperStopped:
            raise
        except Exception as exc:
            logger.error("Error fetching live odds %s type=%s: %s", race_id, odds_type, exc)
            session.rollback()
            try:
                _record_scrape_log(session, url, "error")
                session.commit()
            except Exception:
                session.rollback()
            counters["errors"] += 1

    return counters


async def run_fetch_live_odds(
    race_ids: list[str],
    types: list[str],
    client: NetkeibaClient,
    session: Session,
) -> dict[str, int]:
    """複数レースのオッズを順次 fetch する。"""
    total = {"fetched": 0, "skipped": 0, "errors": 0}
    for race_id in race_ids:
        if stop_flag.is_stopped():
            raise ScraperStopped("stop flag set in run_fetch_live_odds")
        sub = await fetch_odds_for_race(race_id, types, client, session)
        for k in total:
            total[k] += sub[k]
    return total


async def main(args: argparse.Namespace) -> int:
    configure_logging()
    engine = make_engine(db_path())
    Base.metadata.create_all(engine)

    settings = load_settings()
    rate_limiter = AsyncRateLimiter(settings)
    robots_cache = RobotsCache(settings.user_agent)

    types = [t.strip() for t in args.types.split(",") if t.strip()] if args.types else _DEFAULT_TYPES

    race_ids: list[str] = []
    if args.race_id:
        race_ids = [args.race_id.strip()]
    elif args.race_ids:
        race_ids = [rid.strip() for rid in args.race_ids.split(",") if rid.strip()]

    if not race_ids:
        logger.error("No race IDs specified. Use --race-id or --race-ids.")
        return 1

    async with httpx.AsyncClient() as http_client:
        client = NetkeibaClient(rate_limiter, robots_cache, http_client, settings)
        with session_scope(engine) as session:
            try:
                counters = await run_fetch_live_odds(race_ids, types, client, session)
            except ScraperStopped:
                logger.warning("Scraper stopped by stop flag")
                return 1

    logger.info(
        "fetch_live_odds complete — fetched=%d skipped=%d errors=%d",
        counters["fetched"], counters["skipped"], counters["errors"],
    )
    return 0 if counters["errors"] == 0 else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch live combination odds from netkeiba")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--race-id",
        default=None,
        metavar="ID",
        help="Single race ID (e.g. 202506050911)",
    )
    group.add_argument(
        "--race-ids",
        default=None,
        metavar="ID1,ID2,...",
        help="Comma-separated race IDs",
    )
    parser.add_argument(
        "--types",
        default=None,
        metavar="b1,b4,...",
        help=(
            "Comma-separated odds type codes to fetch "
            "(b1=単勝複勝, b3=枠連, b4=馬連, b5=ワイド, b6=馬単, b7=三連複, b8=三連単). "
            f"Default: {','.join(_DEFAULT_TYPES)}"
        ),
    )
    return parser.parse_args()


def cli_main() -> int:
    args = _parse_args()
    return asyncio.run(main(args))


if __name__ == "__main__":
    raise SystemExit(cli_main())
