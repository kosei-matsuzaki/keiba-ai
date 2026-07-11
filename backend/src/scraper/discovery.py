"""netkeiba からの開催 race_id 自動発見ロジック。

api/routers/scraper.py の discover_* エンドポイントから、ネットワーク取得・パース・
キャッシュといった処理を切り出したもの。

- fastapi には依存しない。失敗は DiscoveryError で送出し、呼び出し側 router が
  HTTPException(502) にマップする。
- 戻り値は素のデータ（list[str] / WeekendDiscovery）。

データソースは race_list_sub.html (race_list.html が Ajax で読み込む HTML 断片)。
以前は JSON API (api_get_race_info_top.html) を使っていたが、2026-07 に日付・
時間帯を問わず 200 + 空ボディを返すようになった (エンドポイント廃止とみられる)
ため、サイト本体と同じ取得経路に切り替えた。race_list_sub は kaisai_date 指定が
正確に効くため、旧実装の shutuba probe (代表 race_id の出馬表から開催日を推定
する並列 fetch) も不要になった。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from core.config import load_settings
from core.dates import this_weekend_dates
from scraper.parsers.race_card_calendar import ParseError as CardCalendarParseError
from scraper.parsers.race_card_calendar import parse_race_ids_from_card_calendar
from scraper.robots import RobotsCache

_RACE_LIST_SUB_URL = (
    "https://race.netkeiba.com/top/race_list_sub.html?kaisai_date={date}"
)

# discover_this_weekend の結果キャッシュ。
# キー: (sat_str, sun_str)、値: (cached_at_monotonic, race_ids, kaisai_day_count)
# JRA の週末スケジュールはほぼ静的なので 30 分キャッシュで十分。
# 月曜の発走馬編成更新等は refresh=True で手動 invalidate できる。
_DISCOVER_CACHE: dict[tuple[str, str], tuple[float, list[str], int]] = {}
_DISCOVER_CACHE_TTL_SEC = 30 * 60

_FETCH_TIMEOUT_SEC = 15.0


class DiscoveryError(Exception):
    """netkeiba 通信失敗・robots 不許可・レスポンスパース失敗を表す。"""


_EMPTY_BODY_MESSAGE = (
    "netkeiba が空のレスポンスを返しました。"
    "深夜帯などにデータ提供が一時停止することがあります。時間をおいて再試行してください。"
)


async def _fetch_race_list_sub_html(
    kaisai_date: str, user_agent: str, robots_cache: RobotsCache
) -> str:
    """指定日の race_list_sub.html を fetch して HTML 文字列を返す。

    通信失敗・HTTP エラー・空ボディは、原因が区別できるメッセージ付きの
    DiscoveryError にして送出する (router が 502 detail に使う)。
    """
    url = _RACE_LIST_SUB_URL.format(date=kaisai_date)

    # robots.txt 確認（既存 scraper の流儀に準拠。同期メソッドのため await 不要）
    if not robots_cache.is_allowed(url):
        raise DiscoveryError("robots.txt disallows race_list_sub URL")

    try:
        async with httpx.AsyncClient(
            headers={"User-Agent": user_agent},
            timeout=_FETCH_TIMEOUT_SEC,
            follow_redirects=True,
        ) as http_client:
            resp = await http_client.get(url)
            resp.raise_for_status()
    except Exception as exc:
        raise DiscoveryError(f"netkeiba へのアクセスに失敗しました: {exc}") from exc

    # 200 でもメンテナンス等で本文が空になることがある (旧 JSON API で 2026-07 に実績あり)。
    if not resp.content.strip():
        raise DiscoveryError(_EMPTY_BODY_MESSAGE)

    return resp.text


def _parse_race_ids(html: str, *, include_nar: bool) -> list[str]:
    """race_list_sub HTML から race_id 一覧を抽出する (開催なしは空リスト)。"""
    try:
        return parse_race_ids_from_card_calendar(
            html, include_nar=include_nar, allow_empty=True
        )
    except CardCalendarParseError as exc:
        raise DiscoveryError(
            f"netkeiba レスポンスのパースに失敗しました: {exc}"
        ) from exc


@dataclass
class WeekendDiscovery:
    race_ids: list[str]
    saturday_date: str
    sunday_date: str
    # 発見した unique 開催日キー (race_id[:10] = 年+場+回+日) の数
    total_kaisai_days_probed: int


async def discover_today_race_ids(kaisai_date: str) -> list[str]:
    """指定日（YYYYMMDD）の開催 race_id 一覧を netkeiba から取得する。

    開催なしの場合は空リストを返す。通信・パース失敗時は DiscoveryError。
    """
    settings = load_settings()
    robots_cache = RobotsCache(settings.user_agent)

    html = await _fetch_race_list_sub_html(
        kaisai_date, settings.user_agent, robots_cache
    )
    # 旧 JSON API 実装は場コードで絞らず全件返していたため挙動を踏襲する
    # (race.netkeiba.com の race_list_sub は実質 JRA のみ)
    return _parse_race_ids(html, include_nar=True)


async def discover_this_weekend_race_ids(refresh: bool = False) -> WeekendDiscovery:
    """今週末 (土・日) の JRA 開催 race_id 一覧を netkeiba から取得する。

    手順:
      1. 今週土曜・日曜それぞれの race_list_sub.html を fetch（計 2 リクエスト）
      2. race_id を抽出し、JRA 場コード (race_id[4:6] in '01'..'10') のみ残す
      3. 両日分を union して返す

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
            cached_at, cached_ids, cached_day_count = cached
            if time.monotonic() - cached_at < _DISCOVER_CACHE_TTL_SEC:
                return WeekendDiscovery(
                    race_ids=cached_ids,
                    saturday_date=sat_str,
                    sunday_date=sun_str,
                    total_kaisai_days_probed=cached_day_count,
                )

    settings = load_settings()
    robots_cache = RobotsCache(settings.user_agent)

    all_ids: set[str] = set()
    for day in (this_sat, this_sun):
        html = await _fetch_race_list_sub_html(
            day.strftime("%Y%m%d"), settings.user_agent, robots_cache
        )
        all_ids.update(_parse_race_ids(html, include_nar=False))

    this_weekend_ids = sorted(all_ids)
    kaisai_day_count = len({rid[:10] for rid in this_weekend_ids})

    # 結果をキャッシュ（次回以降は fetch をスキップして即返す）
    _DISCOVER_CACHE[cache_key] = (time.monotonic(), this_weekend_ids, kaisai_day_count)

    return WeekendDiscovery(
        race_ids=this_weekend_ids,
        saturday_date=sat_str,
        sunday_date=sun_str,
        total_kaisai_days_probed=kaisai_day_count,
    )
