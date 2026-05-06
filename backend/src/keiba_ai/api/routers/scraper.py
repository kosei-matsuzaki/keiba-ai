"""Scraper management endpoints: status, recent activity, run, stop."""

from __future__ import annotations

import asyncio
import re
import time
from datetime import UTC, date, datetime, timedelta
from typing import Annotated
from zoneinfo import ZoneInfo

# netkeiba の race_id は JST 基準で YYYYMMDD を含むので JST 起算の date を使う
_JST = ZoneInfo("Asia/Tokyo")

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from keiba_ai.api.deps import get_job_registry, get_session
from keiba_ai.api.jobs import JobRegistry
from keiba_ai.api.schemas import (
    DiscoverThisWeekendRaceIdsResponse,
    DiscoverTodayRaceIdsResponse,
    FetchLiveOddsRequest,
    JobAccepted,
    ScraperRecentActivity,
    ScraperRunRequest,
    ScraperRunShutubaRequest,
    ScraperStatus,
)
from keiba_ai.scraper.parsers.race_info_top import ParseError as RaceInfoParseError
from keiba_ai.scraper.parsers.race_info_top import (
    extract_jra_race_ids_with_kaisai_groups,
    parse_race_ids,
)
from keiba_ai.scraper.parsers.shutuba import extract_race_date_from_shutuba_html
from keiba_ai.core.dates import this_weekend_dates
from keiba_ai.core.config import load_settings
from keiba_ai.core.paths import db_path
from keiba_ai.db.models.scrape_log import ScrapeLog
from keiba_ai.db.session import make_engine, session_scope
from keiba_ai.jobs.fetch_live_odds import _DEFAULT_TYPES, run_fetch_live_odds
from keiba_ai.jobs.ingest import run_ingest
from keiba_ai.jobs.ingest_shutuba import run_ingest_shutuba
from keiba_ai.scraper import stop_flag
from keiba_ai.scraper.netkeiba import NetkeibaClient
from keiba_ai.scraper.rate_limiter import AsyncRateLimiter
from keiba_ai.scraper.robots import RobotsCache

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

        async with httpx.AsyncClient() as http_client:
            client = NetkeibaClient(rate_limiter, robots_cache, http_client, settings)
            with session_scope(engine) as s:
                await run_ingest_shutuba(date_str, client, s, limit=limit, race_ids=race_ids)

    info = registry.start("ingest_shutuba", _coro)
    return JobAccepted(
        job_id=info.job_id,
        status=info.status,
        started_at=info.started_at,
    )


@router.post("/scraper/fetch_live_odds", response_model=JobAccepted, status_code=202)
async def fetch_live_odds_endpoint(
    body: FetchLiveOddsRequest,
    session: Annotated[Session, Depends(get_session)],  # noqa: ARG001
    registry: Annotated[JobRegistry, Depends(get_job_registry)],
) -> JobAccepted:
    """Fetch live combination odds for the specified race from netkeiba.

    Returns 202 Accepted immediately; the actual fetch runs as a background job.
    """
    race_id = body.race_id
    types = body.types or _DEFAULT_TYPES

    async def _coro() -> None:
        settings = load_settings()
        rate_limiter = AsyncRateLimiter(settings)
        robots_cache = RobotsCache(settings.user_agent)
        engine = make_engine(db_path())

        async with httpx.AsyncClient() as http_client:
            client = NetkeibaClient(rate_limiter, robots_cache, http_client, settings)
            with session_scope(engine) as s:
                await run_fetch_live_odds([race_id], types, client, s)

    info = registry.start("fetch_live_odds", _coro)
    return JobAccepted(
        job_id=info.job_id,
        status=info.status,
        started_at=info.started_at,
    )


_RACE_INFO_TOP_URL = (
    "https://race.netkeiba.com/api/api_get_race_info_top.html?kaisai_date={date}"
)

_SHUTUBA_URL = "https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"

# discover_this_weekend_race_ids の結果キャッシュ。
# キー: (sat_str, sun_str)、値: (cached_at_monotonic, race_ids, total_probed)
# JRA の週末スケジュールはほぼ静的なので 30 分キャッシュで十分。
# 月曜の発走馬編成更新等は ?refresh=1 で手動 invalidate できる。
_DISCOVER_CACHE: dict[tuple[str, str], tuple[float, list[str], int]] = {}
_DISCOVER_CACHE_TTL_SEC = 30 * 60

# discover の shutuba probe は max 13 件・1 ユーザ操作あたり 1 回しか走らないため、
# 通常スクレイピング (3-6s 直列) より積極的な throttle で十分。
# 同時 3 並列 + 1 件あたり 8s timeout で 13 probe ≈ 5 batch ≈ 40s に収まる。
_DISCOVER_PROBE_CONCURRENCY = 3
_DISCOVER_PROBE_TIMEOUT_SEC = 8.0


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
    if date:
        # YYYY-MM-DD → YYYYMMDD
        kaisai_date = date.replace("-", "")
    else:
        kaisai_date = datetime.now(_JST).strftime("%Y%m%d")

    url = _RACE_INFO_TOP_URL.format(date=kaisai_date)

    settings = load_settings()
    robots_cache = RobotsCache(settings.user_agent)

    # robots.txt 確認（既存 scraper の流儀に準拠。同期メソッドのため await 不要）
    if not robots_cache.is_allowed(url):
        raise HTTPException(status_code=502, detail="robots.txt disallows this URL")

    try:
        async with httpx.AsyncClient(
            headers={"User-Agent": settings.user_agent},
            timeout=15.0,
            follow_redirects=True,
        ) as http_client:
            resp = await http_client.get(url)
            resp.raise_for_status()
            payload = resp.json()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"netkeiba API へのアクセスに失敗しました: {exc}",
        ) from exc

    try:
        race_ids = parse_race_ids(payload)
    except RaceInfoParseError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"netkeiba API レスポンスのパースに失敗しました: {exc}",
        ) from exc

    discovered_at = datetime.now(UTC).isoformat()
    return DiscoverTodayRaceIdsResponse(race_ids=race_ids, discovered_at=discovered_at)


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

    手順:
      1. api_get_race_info_top.html を 1 回 fetch（kaisai_date 引数なし → 全 active kaisai）
      2. JRA 場コード (race_id[4:6] in '01'..'10') のみ残す
      3. unique 開催日キー (race_id[:10]) ごとに代表 race_id を選ぶ
      4. 各代表の shutuba ページを fetch して date を抽出（軽量 throttle）
      5. date が今週土 or 今週日に一致する開催日キーの race_id だけ返す

    パフォーマンス:
      - 結果は (sat_str, sun_str) キーで 30 分間 in-process キャッシュ
      - shutuba probe は AsyncRateLimiter (直列 3-6s) ではなく semaphore で
        並列 3 本まで・1 件 8s timeout で投げる
        （13 probe × 1 ユーザ操作なので netkeiba 負荷は極小）
      - 13 probe を最悪 5 batch ≈ 40s で完了する想定

    - 開催なし → race_ids=[] を返す（404 ではない）
    - netkeiba 通信エラー・パース失敗 → 502
    """
    this_sat, this_sun = this_weekend_dates()
    sat_str = this_sat.isoformat()
    sun_str = this_sun.isoformat()

    # ── Cache hit check ──────────────────────────────────────────────────────
    cache_key = (sat_str, sun_str)
    if not refresh:
        cached = _DISCOVER_CACHE.get(cache_key)
        if cached is not None:
            cached_at, cached_ids, cached_probed = cached
            if time.monotonic() - cached_at < _DISCOVER_CACHE_TTL_SEC:
                return DiscoverThisWeekendRaceIdsResponse(
                    race_ids=cached_ids,
                    saturday_date=sat_str,
                    sunday_date=sun_str,
                    total_kaisai_days_probed=cached_probed,
                    discovered_at=datetime.now(UTC).isoformat(),
                )

    settings = load_settings()
    robots_cache = RobotsCache(settings.user_agent)

    # ── Step 1: race_info_top を fetch ───────────────────────────────────────
    # kaisai_date を指定しないと全 active kaisai（複数週分）が返るため、
    # 直近の土曜を渡して同等の「今後の開催全部」を取得する。
    # 実際には date 引数を無視して 156 件返すことが確認されているが、
    # 明示的に今週土曜を渡すことで netkeiba 側キャッシュを正しく引ける。
    top_url = _RACE_INFO_TOP_URL.format(date=this_sat.strftime("%Y%m%d"))

    if not robots_cache.is_allowed(top_url):
        raise HTTPException(status_code=502, detail="robots.txt disallows race_info_top URL")

    try:
        async with httpx.AsyncClient(
            headers={"User-Agent": settings.user_agent},
            timeout=15.0,
            follow_redirects=True,
        ) as http_client:
            resp = await http_client.get(top_url)
            resp.raise_for_status()
            payload = resp.json()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"netkeiba race_info_top へのアクセスに失敗しました: {exc}",
        ) from exc

    try:
        _jra_race_ids, groups = extract_jra_race_ids_with_kaisai_groups(payload)
    except RaceInfoParseError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"netkeiba API レスポンスのパースに失敗しました: {exc}",
        ) from exc

    if not groups:
        # キャッシュにも空結果を入れて、開催なし週に何度叩かれても即返答できるようにする
        _DISCOVER_CACHE[cache_key] = (time.monotonic(), [], 0)
        discovered_at = datetime.now(UTC).isoformat()
        return DiscoverThisWeekendRaceIdsResponse(
            race_ids=[],
            saturday_date=sat_str,
            sunday_date=sun_str,
            total_kaisai_days_probed=0,
            discovered_at=discovered_at,
        )

    # ── Step 2: 各 unique 開催日キーの代表 race_id で shutuba を並列 fetch ────
    # 代表は各グループの先頭（最若番、= race_id が最小のもの）
    semaphore = asyncio.Semaphore(_DISCOVER_PROBE_CONCURRENCY)

    async def _probe_one(
        client: httpx.AsyncClient, key: str, rep_id: str
    ) -> tuple[str, str | None]:
        """1 つの kaisai_day_key について shutuba を fetch して date を返す。

        失敗時は date=None を返し、呼び出し側で skip させる。
        """
        async with semaphore:
            shutuba_url = _SHUTUBA_URL.format(race_id=rep_id)
            if not robots_cache.is_allowed(shutuba_url):
                return key, None
            try:
                sresp = await client.get(shutuba_url)
                sresp.raise_for_status()
                # race.netkeiba.com は Content-Type に charset を付けないため
                # httpx は UTF-8 と推定するが、実体は EUC-JP。明示しないと
                # title 内の "YYYY年MM月DD日" が mojibake 化して正規表現にマッチしない。
                sresp.encoding = "euc-jp"
                return key, extract_race_date_from_shutuba_html(sresp.text)
            except Exception:
                return key, None

    weekend_keys: set[str] = set()

    async with httpx.AsyncClient(
        headers={"User-Agent": settings.user_agent},
        timeout=_DISCOVER_PROBE_TIMEOUT_SEC,
        follow_redirects=True,
    ) as http_client:
        results = await asyncio.gather(
            *(_probe_one(http_client, key, ids[0]) for key, ids in groups.items()),
            return_exceptions=False,
        )

    for key, race_date in results:
        if race_date in (sat_str, sun_str):
            weekend_keys.add(key)

    # ── Step 3: 今週末キーに属する race_id だけ抽出 ─────────────────────────
    this_weekend_ids = sorted(
        rid
        for key, ids in groups.items()
        if key in weekend_keys
        for rid in ids
    )

    # 結果をキャッシュ（次回以降は probe をスキップして即返す）
    _DISCOVER_CACHE[cache_key] = (time.monotonic(), this_weekend_ids, len(groups))

    discovered_at = datetime.now(UTC).isoformat()
    return DiscoverThisWeekendRaceIdsResponse(
        race_ids=this_weekend_ids,
        saturday_date=sat_str,
        sunday_date=sun_str,
        total_kaisai_days_probed=len(groups),
        discovered_at=discovered_at,
    )


@router.post("/scraper/stop", status_code=200)
def stop_scraper() -> dict:
    stop_flag.set_stopped()
    return {"ok": True}
