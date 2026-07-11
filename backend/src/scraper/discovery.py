"""netkeiba からの開催 race_id 自動発見ロジック。

api/routers/scraper.py の discover_* エンドポイントから、ネットワーク取得・パース・
キャッシュ・並列 probe といった処理を切り出したもの。

- fastapi には依存しない。失敗は DiscoveryError で送出し、呼び出し側 router が
  HTTPException(502) にマップする。
- 戻り値は素のデータ（list[str] / WeekendDiscovery）。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import httpx

from core.config import load_settings
from core.dates import this_weekend_dates
from scraper.parsers.race_info_top import ParseError as RaceInfoParseError
from scraper.parsers.race_info_top import (
    extract_jra_race_ids_with_kaisai_groups,
    parse_race_ids,
)
from scraper.parsers.shutuba import extract_race_date_from_shutuba_html
from scraper.robots import RobotsCache

_RACE_INFO_TOP_URL = (
    "https://race.netkeiba.com/api/api_get_race_info_top.html?kaisai_date={date}"
)

_SHUTUBA_URL = "https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"

# discover_this_weekend の結果キャッシュ。
# キー: (sat_str, sun_str)、値: (cached_at_monotonic, race_ids, total_probed)
# JRA の週末スケジュールはほぼ静的なので 30 分キャッシュで十分。
# 月曜の発走馬編成更新等は refresh=True で手動 invalidate できる。
_DISCOVER_CACHE: dict[tuple[str, str], tuple[float, list[str], int]] = {}
_DISCOVER_CACHE_TTL_SEC = 30 * 60

# shutuba probe は max 13 件・1 ユーザ操作あたり 1 回しか走らないため、
# 通常スクレイピング (3-6s 直列) より積極的な throttle で十分。
# 同時 3 並列 + 1 件あたり 8s timeout で 13 probe ≈ 5 batch ≈ 40s に収まる。
_DISCOVER_PROBE_CONCURRENCY = 3
_DISCOVER_PROBE_TIMEOUT_SEC = 8.0


class DiscoveryError(Exception):
    """netkeiba 通信失敗・robots 不許可・レスポンスパース失敗を表す。"""


_EMPTY_BODY_MESSAGE = (
    "netkeiba API が空のレスポンスを返しました。"
    "深夜帯などにデータ提供が一時停止することがあります。時間をおいて再試行してください。"
)


async def _fetch_json(url: str, user_agent: str, timeout: float = 15.0) -> object:
    """netkeiba JSON API を fetch してパース済み payload を返す。

    通信失敗・HTTP エラー・空ボディ・JSON 不正は、原因が区別できる
    メッセージ付きの DiscoveryError にして送出する (router が 502 detail に使う)。
    """
    try:
        async with httpx.AsyncClient(
            headers={"User-Agent": user_agent},
            timeout=timeout,
            follow_redirects=True,
        ) as http_client:
            resp = await http_client.get(url)
            resp.raise_for_status()
    except Exception as exc:
        raise DiscoveryError(f"netkeiba API へのアクセスに失敗しました: {exc}") from exc

    # 200 でもメンテナンス等で本文が空になることがある (2026-07 に実績あり)。
    # json() に任せると "Expecting value" という不親切な文言になるため先に判定する。
    if not resp.content.strip():
        raise DiscoveryError(_EMPTY_BODY_MESSAGE)

    try:
        return resp.json()
    except Exception as exc:
        raise DiscoveryError(
            f"netkeiba API のレスポンスが JSON として解釈できませんでした: {exc}"
        ) from exc


@dataclass
class WeekendDiscovery:
    race_ids: list[str]
    saturday_date: str
    sunday_date: str
    total_kaisai_days_probed: int


async def discover_today_race_ids(kaisai_date: str) -> list[str]:
    """指定日（YYYYMMDD）の開催 race_id 一覧を netkeiba から取得する。

    開催なしの場合は空リストを返す。通信・パース失敗時は DiscoveryError。
    """
    url = _RACE_INFO_TOP_URL.format(date=kaisai_date)

    settings = load_settings()
    robots_cache = RobotsCache(settings.user_agent)

    # robots.txt 確認（既存 scraper の流儀に準拠。同期メソッドのため await 不要）
    if not robots_cache.is_allowed(url):
        raise DiscoveryError("robots.txt disallows this URL")

    payload = await _fetch_json(url, settings.user_agent)

    try:
        return parse_race_ids(payload)
    except RaceInfoParseError as exc:
        raise DiscoveryError(f"netkeiba API レスポンスのパースに失敗しました: {exc}") from exc


async def discover_this_weekend_race_ids(refresh: bool = False) -> WeekendDiscovery:
    """今週末 (土・日) の JRA 開催 race_id 一覧を netkeiba から取得する。

    手順:
      1. api_get_race_info_top.html を 1 回 fetch（全 active kaisai）
      2. JRA 場コード (race_id[4:6] in '01'..'10') のみ残す
      3. unique 開催日キー (race_id[:10]) ごとに代表 race_id を選ぶ
      4. 各代表の shutuba ページを fetch して date を抽出（軽量 throttle）
      5. date が今週土 or 今週日に一致する開催日キーの race_id だけ返す

    結果は (sat_str, sun_str) キーで 30 分間 in-process キャッシュ。
    開催なしは race_ids=[] を返す。通信・パース失敗時は DiscoveryError。
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
                return WeekendDiscovery(
                    race_ids=cached_ids,
                    saturday_date=sat_str,
                    sunday_date=sun_str,
                    total_kaisai_days_probed=cached_probed,
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
        raise DiscoveryError("robots.txt disallows race_info_top URL")

    payload = await _fetch_json(top_url, settings.user_agent)

    try:
        _jra_race_ids, groups = extract_jra_race_ids_with_kaisai_groups(payload)
    except RaceInfoParseError as exc:
        raise DiscoveryError(f"netkeiba API レスポンスのパースに失敗しました: {exc}") from exc

    if not groups:
        # キャッシュにも空結果を入れて、開催なし週に何度叩かれても即返答できるようにする
        _DISCOVER_CACHE[cache_key] = (time.monotonic(), [], 0)
        return WeekendDiscovery(
            race_ids=[],
            saturday_date=sat_str,
            sunday_date=sun_str,
            total_kaisai_days_probed=0,
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

    return WeekendDiscovery(
        race_ids=this_weekend_ids,
        saturday_date=sat_str,
        sunday_date=sun_str,
        total_kaisai_days_probed=len(groups),
    )
