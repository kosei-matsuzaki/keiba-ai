"""Scraper management endpoints: status, recent activity, run, stop."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta
from typing import Annotated
from zoneinfo import ZoneInfo

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.deps import get_job_registry, get_session
from api.jobs import JobRegistry
from api.schemas import (
    DiscoverThisWeekendRaceIdsResponse,
    DiscoverTodayRaceIdsResponse,
    JobAccepted,
    ScraperRecentActivity,
    ScraperRunRequest,
    ScraperRunResultsRequest,
    ScraperRunShutubaRequest,
    ScraperStatus,
)
from core.config import load_settings
from core.paths import db_path
from db.models.scrape_log import ScrapeLog
from db.odds_db import init_odds_db, make_odds_engine
from db.session import make_engine, session_scope
from jobs.ingest import run_ingest
from jobs.ingest_results import run_ingest_recent_results
from jobs.ingest_shutuba import run_ingest_shutuba
from scraper import discovery, stop_flag
from scraper.discovery import DiscoveryError
from scraper.netkeiba import NetkeibaClient
from scraper.rate_limiter import AsyncRateLimiter
from scraper.robots import RobotsCache

# netkeiba の race_id は JST 基準で YYYYMMDD を含むので JST 起算の date を使う
_JST = ZoneInfo("Asia/Tokyo")

router = APIRouter()

_RESULT_URL_PREFIX = "https://db.netkeiba.com/race/"
_RACE_ID_FROM_URL_RE = re.compile(r"/race/(\d{12})")


def _count_missing_dates(session: Session, days: int = 30) -> int:
    """Count days in the recent window that have no 'ok' scrape_log entries.

    A day is considered 'completed' if at least one result URL for that date's
    compact prefix (YYYYMMDD) has status='ok' in scrape_log.
    """
    today = datetime.now(_JST).date()
    start = today - timedelta(days=days - 1)  # inclusive range of `days` days

    # Collect distinct date prefixes that have at least one 'ok' entry.
    # Race result URLs are https://db.netkeiba.com/race/<YYYYMMDD><suffix>/
    # We extract the 8-char date prefix from urls matching the base prefix.
    rows = session.execute(
        select(ScrapeLog.url).where(
            ScrapeLog.url.like(f"{_RESULT_URL_PREFIX}%"),
            ScrapeLog.status == "ok",
        )
    ).scalars().all()

    completed_dates: set[str] = set()
    prefix_len = len(_RESULT_URL_PREFIX)
    for url in rows:
        # URL format: https://db.netkeiba.com/race/YYYYMMDDXXXX/
        suffix = url[prefix_len:]  # e.g. "202412280101/"
        if len(suffix) >= 8:
            date_str = suffix[:8]  # "20241228"
            try:
                d = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
                if start <= d <= today:
                    completed_dates.add(date_str)
            except ValueError:
                pass

    missing = days - len(completed_dates)
    return max(missing, 0)


@router.get("/scraper/status", response_model=ScraperStatus)
def get_scraper_status(
    session: Annotated[Session, Depends(get_session)],
    registry: Annotated[JobRegistry, Depends(get_job_registry)],
    range_days: int = Query(default=30, ge=1, le=365, alias="range"),
) -> ScraperStatus:
    # Latest fetched_at from scrape_log (status='ok')
    row = session.execute(
        select(func.max(ScrapeLog.fetched_at)).where(ScrapeLog.status == "ok")
    ).scalar()
    last_fetched = row if row else None

    missing = _count_missing_dates(session, days=range_days)

    return ScraperStatus(
        stopped=stop_flag.is_stopped(),
        last_fetched_date=last_fetched,
        missing_dates_count=missing,
        current_job_id=registry.current_ingest_job_id(),
    )


@router.get("/scraper/recent_activity", response_model=ScraperRecentActivity)
def recent_activity(
    session: Annotated[Session, Depends(get_session)],
    minutes: int = Query(default=10, ge=1, le=1440),
) -> ScraperRecentActivity:
    """Aggregate scrape_log over the last N minutes.

    Designed to surface CLI-driven ingest progress alongside UI-launched jobs.
    fetched_at is stored as UTC ISO 8601, so a string `>=` cutoff works.
    """
    cutoff = (datetime.now(UTC) - timedelta(minutes=minutes)).isoformat()

    # Phase 2 ingest が走っている時は scrape_log が高頻度 INSERT され、
    # 直近 10 分で数千〜万行に達することがある。UI 側は集計とごく最新の
    # latest_race_id しか参照しないので、新しい順に上限件数だけ取る。
    # 数値は控えめに 2000 行 (= ~5 秒/件 で 10000 秒分 = 約 2.7 時間相当の
    # ピーク fetch 数) で、通常運用では cutoff より十分多い枠。
    rows = session.execute(
        select(ScrapeLog.url, ScrapeLog.status, ScrapeLog.fetched_at)
        .where(ScrapeLog.fetched_at >= cutoff)
        .order_by(ScrapeLog.fetched_at.desc())
        .limit(2000)
    ).all()

    total = len(rows)
    ok = sum(1 for r in rows if r.status == "ok")
    err = sum(1 for r in rows if r.status == "error")
    skipped = sum(1 for r in rows if r.status == "skipped")

    latest_fetched_at = rows[0].fetched_at if rows else None
    latest_race_id: str | None = None
    if rows:
        m = _RACE_ID_FROM_URL_RE.search(rows[0].url)
        if m:
            latest_race_id = m.group(1)

    rate_per_min = ok / max(minutes, 1)

    return ScraperRecentActivity(
        window_minutes=minutes,
        total_fetched=total,
        ok_count=ok,
        error_count=err,
        skipped_count=skipped,
        rate_per_min=round(rate_per_min, 2),
        latest_fetched_at=latest_fetched_at,
        latest_race_id=latest_race_id,
    )


@router.post("/scraper/run", response_model=JobAccepted)
async def run_scraper(
    body: ScraperRunRequest,
    session: Annotated[Session, Depends(get_session)],  # noqa: ARG001
    registry: Annotated[JobRegistry, Depends(get_job_registry)],
) -> JobAccepted:
    date_str = body.date
    limit = body.limit

    async def _coro() -> None:
        settings = load_settings()
        rate_limiter = AsyncRateLimiter(settings)
        robots_cache = RobotsCache(settings.user_agent)
        engine = make_engine(db_path())

        async with httpx.AsyncClient() as http_client:
            client = NetkeibaClient(rate_limiter, robots_cache, http_client, settings)
            with session_scope(engine) as s:
                await run_ingest(date_str, client, s, limit=limit)

    info = registry.start("ingest", _coro)
    return JobAccepted(
        job_id=info.job_id,
        status=info.status,
        started_at=info.started_at,
    )


@router.post("/scraper/run_results", response_model=JobAccepted, status_code=202)
async def run_results_scraper(
    body: ScraperRunResultsRequest,
    session: Annotated[Session, Depends(get_session)],  # noqa: ARG001
    registry: Annotated[JobRegistry, Depends(get_job_registry)],
) -> JobAccepted:
    """確定したレースの結果＋確定オッズを期間指定で未取得分だけ取り込む。

    from/to 指定でその範囲、未指定なら直近 days 日（昨日まで）。取込済みはスキップ、
    今日は未確定のため自動除外。202 を即返し、実処理はバックグラウンドジョブで動く。
    """
    # 期間を解決（today は JST）。今日除外・span 上限は run_ingest_recent_results が再クランプ。
    base_day = datetime.now(_JST).date()
    if body.from_ and body.to:
        start, end = body.from_, body.to
    else:
        yesterday = base_day - timedelta(days=1)
        start = (yesterday - timedelta(days=body.days - 1)).isoformat()
        end = yesterday.isoformat()

    async def _coro() -> None:
        settings = load_settings()
        rate_limiter = AsyncRateLimiter(settings)
        robots_cache = RobotsCache(settings.user_agent)
        engine = make_engine(db_path())
        odds_engine = make_odds_engine()
        init_odds_db(odds_engine)

        async with httpx.AsyncClient() as http_client:
            client = NetkeibaClient(rate_limiter, robots_cache, http_client, settings)
            with session_scope(engine) as s:
                await run_ingest_recent_results(client, s, odds_engine, start=start, end=end)
        odds_engine.dispose()

    info = registry.start("ingest", _coro)
    return JobAccepted(
        job_id=info.job_id,
        status=info.status,
        started_at=info.started_at,
    )


@router.post("/scraper/run_shutuba", response_model=JobAccepted, status_code=202)
async def run_shutuba_scraper(
    body: ScraperRunShutubaRequest,
    session: Annotated[Session, Depends(get_session)],  # noqa: ARG001
    registry: Annotated[JobRegistry, Depends(get_job_registry)],
) -> JobAccepted:
    """Fetch and ingest shutuba (出馬表) pages for the given date or specific race IDs.

    - race_ids 指定時: calendar fetch を skip し指定レースのみ ingest。
    - date のみ指定時: calendar から race_id 一覧を取得して ingest（既存挙動）。
    - 両方指定時: race_ids 優先（CLI 仕様と一致）。

    Returns 202 Accepted immediately; the actual scraping runs as a background job.
    """
    # race_ids 優先。
    # race_ids 指定時は date を None のまま渡す — HTML から日付を抽出させる。
    # date が明示指定された場合のみ date_str に渡す（calendar fetch / CLI 互換）。
    race_ids = body.race_ids or None
    date_str: str | None = body.date  # None は許容される（HTML date を優先させるため）
    limit = body.limit

    async def _coro() -> None:
        settings = load_settings()
        rate_limiter = AsyncRateLimiter(settings)
        robots_cache = RobotsCache(settings.user_agent)
        engine = make_engine(db_path())
        # ライブの全馬券実オッズを odds.db に保存する（推奨買目で実オッズを使う）。
        odds_engine = make_odds_engine()
        init_odds_db(odds_engine)

        async with httpx.AsyncClient() as http_client:
            client = NetkeibaClient(rate_limiter, robots_cache, http_client, settings)
            with session_scope(engine) as s:
                await run_ingest_shutuba(
                    date_str, client, s, limit=limit, race_ids=race_ids, odds_engine=odds_engine
                )
        odds_engine.dispose()

    info = registry.start("ingest_shutuba", _coro)
    return JobAccepted(
        job_id=info.job_id,
        status=info.status,
        started_at=info.started_at,
    )


@router.get("/scraper/discover_today_race_ids", response_model=DiscoverTodayRaceIdsResponse)
async def discover_today_race_ids(
    date: str = Query(
        default="",
        description="YYYY-MM-DD 形式の日付。省略時は JST 当日を使用する。",
        pattern=r"^(\d{4}-\d{2}-\d{2})?$",
    ),
) -> DiscoverTodayRaceIdsResponse:
    """当日（または指定日）の開催 race_id 一覧を netkeiba から自動発見する。

    - date 省略時は JST 当日を使用。
    - 該当日の開催なし → race_ids=[] を返す（404 ではない）。
    - netkeiba 側の通信エラーや想定外レスポンスは 502 で返す。
    """
    # YYYY-MM-DD → YYYYMMDD (date 省略時は JST 当日)
    kaisai_date = date.replace("-", "") if date else datetime.now(_JST).strftime("%Y%m%d")

    try:
        race_ids = await discovery.discover_today_race_ids(kaisai_date)
    except DiscoveryError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return DiscoverTodayRaceIdsResponse(
        race_ids=race_ids,
        discovered_at=datetime.now(UTC).isoformat(),
    )


@router.get(
    "/scraper/discover_this_weekend_race_ids",
    response_model=DiscoverThisWeekendRaceIdsResponse,
)
async def discover_this_weekend_race_ids(
    refresh: bool = Query(
        default=False,
        description="True の場合は in-process キャッシュを無視して再取得する。",
    ),
) -> DiscoverThisWeekendRaceIdsResponse:
    """今週末 (土・日) の JRA 開催 race_id 一覧を netkeiba から自動発見する。

    発見ロジック本体は scraper.discovery.discover_this_weekend_race_ids に委譲。
    開催なし → race_ids=[]（404 ではない）、netkeiba 通信・パース失敗 → 502。
    """
    try:
        result = await discovery.discover_this_weekend_race_ids(refresh=refresh)
    except DiscoveryError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return DiscoverThisWeekendRaceIdsResponse(
        race_ids=result.race_ids,
        saturday_date=result.saturday_date,
        sunday_date=result.sunday_date,
        total_kaisai_days_probed=result.total_kaisai_days_probed,
        discovered_at=datetime.now(UTC).isoformat(),
    )


@router.post("/scraper/stop", status_code=200)
def stop_scraper() -> dict:
    stop_flag.set_stopped()
    return {"ok": True}
