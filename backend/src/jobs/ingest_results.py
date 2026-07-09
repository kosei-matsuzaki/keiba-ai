"""確定したレースを期間指定で取り込むジョブ（結果＋確定オッズ、未取得分だけ）。

指定期間 [start, end]（JST、今日は未確定のため自動的に昨日までにクランプ）の各開催日について:
  - db.netkeiba カレンダーからレースを発見し、**レース本体（出走馬・着順・払戻）を取り込む**
    （keiba.db に未登録のレースも作成される）。
  - keiba.db に既にあるレースも対象に含める（出馬表だけ取込済みで結果未取得のもの等）。
  - 各レースの確定オッズ（odds.db）を race.netkeiba API から取り込む。

スキップ方針（範囲内でも取込済みはスキップ）:
  - 結果は scrape_log=ok のレースをスキップ。
  - 確定オッズは確定済み（is_confirmed）をスキップし、ライブのみの行は確定値で上書き。

db.netkeiba は直近レースのアーカイブ反映が数日遅れるため、結果ページ未掲載のうちは
pending として静かに見送り（ok を残さず後日再取得）、確定オッズだけ先に取り込む。

Usage:
    python -m jobs.ingest_results --start 2026-05-01 --end 2026-06-13
    python -m jobs.ingest_results --days 14          # 直近 14 日（昨日まで）
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import os

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from core.config import load_settings
from core.dates import today_jst
from core.logging import configure_logging, get_logger
from core.paths import db_path
from db.base import Base
from db.models.race import Race
from db.odds_db import init_odds_db, make_odds_engine
from db.session import make_engine, session_scope
from jobs.ingest import _CALENDAR_URL, ingest_one_race_result
from jobs.ingest_odds import _ingest_race
from scraper import stop_flag
from scraper.netkeiba import NetkeibaClient
from scraper.parsers.race_calendar import parse_race_ids_from_calendar
from scraper.rate_limiter import AsyncRateLimiter
from scraper.robots import RobotsCache
from scraper.stop_flag import ScraperStopped

logger = get_logger(__name__)

_DEFAULT_DAYS = 14
_MAX_SPAN_DAYS = 90  # UI/エンドポイントからの 1 回の取込上限（過大な遡及を防止）


def _clamp_range(
    start: datetime.date, end: datetime.date, base_day: datetime.date
) -> tuple[datetime.date, datetime.date]:
    """今日を除外（昨日までにクランプ）し、span 上限を適用する。"""
    yesterday = base_day - datetime.timedelta(days=1)
    if end > yesterday:
        end = yesterday
    # span 上限（end から遡って _MAX_SPAN_DAYS 日まで）
    earliest = end - datetime.timedelta(days=_MAX_SPAN_DAYS - 1)
    if start < earliest:
        start = earliest
    return start, end


async def _discover_race_ids(
    client: NetkeibaClient,
    start: datetime.date,
    end: datetime.date,
    include_nar: bool,
) -> dict[str, str]:
    """[start, end] の各日について db.netkeiba カレンダーから race_id を発見する。

    Returns: {race_id: date_str}。開催の無い日 / 未アーカイブの日は静かにスキップ。
    """
    found: dict[str, str] = {}
    n_days = (end - start).days + 1
    for i in range(n_days):
        if stop_flag.is_stopped():
            raise ScraperStopped("stop flag set during calendar discovery")
        d = start + datetime.timedelta(days=i)
        date_str = d.isoformat()
        url = _CALENDAR_URL.format(date=date_str.replace("-", ""))
        try:
            html = await client.fetch(url, cache_max_age_hours=6)
        except ScraperStopped:
            raise
        except Exception as exc:  # noqa: BLE001 — 1 日分の失敗は致命的でない
            logger.warning("Calendar fetch failed for %s: %s", date_str, exc)
            continue
        ids = parse_race_ids_from_calendar(html, include_nar=include_nar, raise_if_empty=False)
        for rid in ids:
            found.setdefault(rid, date_str)
    return found


async def run_ingest_recent_results(
    client: NetkeibaClient,
    session: Session,
    odds_engine,
    *,
    start: str,
    end: str,
    today: datetime.date | None = None,
) -> dict[str, int]:
    """期間 [start, end] の確定レースの結果＋確定オッズを未取得分だけ取り込む。

    Returns 集計 counters: results / odds / pending / skipped / errors / races。
    """
    base_day = today or today_jst()
    start_d, end_d = _clamp_range(
        datetime.date.fromisoformat(start), datetime.date.fromisoformat(end), base_day
    )

    counters = {
        "results": 0,   # 新たに結果を取り込んだレース数
        "odds": 0,      # 新たに確定オッズを取り込んだレース数
        "pending": 0,   # 結果がまだ db.netkeiba に未掲載（後日再取得）
        "skipped": 0,   # 取込済みでスキップ
        "errors": 0,
        "races": 0,     # 対象レース総数
    }
    if start_d > end_d:
        logger.info("Recent results: empty range after clamp (%s..%s)", start, end)
        return counters

    include_nar = os.getenv("KEIBA_INCLUDE_NAR", "0") == "1"

    # 1) カレンダーからレースを発見（keiba.db 未登録のものも作成対象にする）
    race_dates = await _discover_race_ids(client, start_d, end_d, include_nar)

    # 2) keiba.db に既にあるレース（出馬表のみ取込済み等）も対象に加える
    start_str, end_str = start_d.isoformat(), end_d.isoformat()
    rows = session.execute(
        select(Race.race_id, Race.date).where(Race.date >= start_str, Race.date <= end_str)
    ).all()
    for rid, rdate in rows:
        race_dates.setdefault(rid, rdate)

    counters["races"] = len(race_dates)
    logger.info(
        "Recent results: %d candidate race(s) in %s..%s", len(race_dates), start_str, end_str
    )

    for race_id, race_date in sorted(race_dates.items(), key=lambda kv: (kv[1], kv[0]), reverse=True):
        if stop_flag.is_stopped():
            raise ScraperStopped("stop flag set during recent-results loop")

        # 1) 結果（レース本体・着順・払戻）。取込済みは skip、未掲載(未アーカイブ)は pending。
        status = await ingest_one_race_result(client, session, race_id, race_date)
        if status == "fetched":
            counters["results"] += 1
        elif status == "no_results":
            counters["pending"] += 1
        elif status == "error":
            counters["errors"] += 1
        # "skipped" → 既に結果あり

        # 2) 確定オッズ（odds.db）。終了済みレースなので API は status=result を返す。
        #    確定済みは _ingest_race が resume skip、ライブのみの行は確定値で上書き。
        try:
            odds_status = await _ingest_race(client, odds_engine, race_id)
        except ScraperStopped:
            raise
        except Exception as exc:  # noqa: BLE001 — 1 レースのオッズ失敗は致命的でない
            logger.warning("Confirmed odds fetch failed for %s: %s", race_id, exc)
            counters["errors"] += 1
            continue
        if odds_status in ("done", "partial"):
            counters["odds"] += 1

    return counters


def _resolve_range(args: argparse.Namespace, base_day: datetime.date) -> tuple[str, str]:
    """CLI 引数から [start, end] を決める（--start/--end 優先、無ければ --days）。"""
    if args.start and args.end:
        return args.start, args.end
    days = args.days or _DEFAULT_DAYS
    end = base_day - datetime.timedelta(days=1)
    start = end - datetime.timedelta(days=days - 1)
    return start.isoformat(), end.isoformat()


async def main(args: argparse.Namespace) -> int:
    configure_logging()
    engine = make_engine(db_path())
    Base.metadata.create_all(engine)
    odds_engine = make_odds_engine()
    init_odds_db(odds_engine)

    base_day = today_jst()
    start, end = _resolve_range(args, base_day)

    settings = load_settings()
    async with httpx.AsyncClient() as http_client:
        client = NetkeibaClient(
            AsyncRateLimiter(settings), RobotsCache(settings.user_agent), http_client, settings
        )
        with session_scope(engine) as session:
            try:
                counters = await run_ingest_recent_results(
                    client, session, odds_engine, start=start, end=end, today=base_day
                )
            except ScraperStopped:
                logger.warning("Scraper stopped by stop flag")
                odds_engine.dispose()
                return 1
    odds_engine.dispose()

    logger.info(
        "Recent results complete — results=%d odds=%d pending=%d skipped=%d errors=%d (races=%d)",
        counters["results"], counters["odds"], counters["pending"],
        counters["skipped"], counters["errors"], counters["races"],
    )
    return 0 if counters["errors"] == 0 else 1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest results + confirmed odds for finalized races in a date range"
    )
    p.add_argument("--start", metavar="YYYY-MM-DD", default=None, help="Range start (inclusive)")
    p.add_argument("--end", metavar="YYYY-MM-DD", default=None, help="Range end (inclusive)")
    p.add_argument(
        "--days", type=int, default=None, metavar="N",
        help=f"--start/--end 未指定時、直近 N 日（昨日まで）。既定 {_DEFAULT_DAYS}",
    )
    return p.parse_args()


def cli_main() -> int:
    return asyncio.run(main(_parse_args()))


if __name__ == "__main__":
    raise SystemExit(cli_main())
