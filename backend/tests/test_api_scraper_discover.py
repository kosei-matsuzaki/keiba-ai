"""Tests for GET /api/scraper/discover_today_race_ids and
GET /api/scraper/discover_this_weekend_race_ids endpoints."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient


def _make_netkeiba_response(race_ids: list[str]) -> MagicMock:
    """Build a mock httpx.Response returning a valid netkeiba payload."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "status": "OK",
        "data": {
            "info": [{"RaceId": rid} for rid in race_ids],
        },
    }
    return mock_resp


def _make_empty_netkeiba_response() -> MagicMock:
    """Build a mock response with no races (valid OK, empty info)."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"status": "OK", "data": {"info": []}}
    return mock_resp


class TestDiscoverTodayRaceIds:
    def test_returns_race_ids_on_success(self, api_client: TestClient) -> None:
        """正常系: race_ids が返ること。"""
        expected_ids = ["202506050101", "202506050102", "202506050201"]
        mock_resp = _make_netkeiba_response(expected_ids)

        with (
            patch(
                "api.routers.scraper.RobotsCache.is_allowed",
                return_value=True,
            ),
            patch(
                "httpx.AsyncClient.get",
                new=AsyncMock(return_value=mock_resp),
            ),
        ):
            resp = api_client.get("/api/scraper/discover_today_race_ids")

        assert resp.status_code == 200
        data = resp.json()
        assert sorted(data["race_ids"]) == sorted(expected_ids)
        assert "discovered_at" in data

    def test_returns_empty_list_when_no_races(self, api_client: TestClient) -> None:
        """空応答: race_ids=[] でも 200 を返すこと（404 ではない）。"""
        mock_resp = _make_empty_netkeiba_response()

        with (
            patch(
                "api.routers.scraper.RobotsCache.is_allowed",
                return_value=True,
            ),
            patch(
                "httpx.AsyncClient.get",
                new=AsyncMock(return_value=mock_resp),
            ),
        ):
            resp = api_client.get("/api/scraper/discover_today_race_ids")

        assert resp.status_code == 200
        assert resp.json()["race_ids"] == []

    def test_returns_502_on_netkeiba_error(self, api_client: TestClient) -> None:
        """netkeiba 通信エラー時は 502 を返すこと。"""
        with (
            patch(
                "api.routers.scraper.RobotsCache.is_allowed",
                return_value=True,
            ),
            patch(
                "httpx.AsyncClient.get",
                new=AsyncMock(side_effect=Exception("connection error")),
            ),
        ):
            resp = api_client.get("/api/scraper/discover_today_race_ids")

        assert resp.status_code == 502
        assert "netkeiba" in resp.json()["detail"]

    def test_returns_502_on_bad_response_status(self, api_client: TestClient) -> None:
        """netkeiba が status=NG を返した場合は 502 を返すこと。"""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"status": "NG", "data": {"info": []}}

        with (
            patch(
                "api.routers.scraper.RobotsCache.is_allowed",
                return_value=True,
            ),
            patch(
                "httpx.AsyncClient.get",
                new=AsyncMock(return_value=mock_resp),
            ),
        ):
            resp = api_client.get("/api/scraper/discover_today_race_ids")

        assert resp.status_code == 502

    def test_accepts_date_query_param(self, api_client: TestClient) -> None:
        """date クエリパラメータを指定できること。"""
        mock_resp = _make_netkeiba_response(["202505050101"])

        with (
            patch(
                "api.routers.scraper.RobotsCache.is_allowed",
                return_value=True,
            ),
            patch(
                "httpx.AsyncClient.get",
                new=AsyncMock(return_value=mock_resp),
            ) as mock_get,
        ):
            resp = api_client.get(
                "/api/scraper/discover_today_race_ids",
                params={"date": "2025-05-05"},
            )

        assert resp.status_code == 200
        # URL に kaisai_date=20250505 が含まれていることを確認
        call_url = mock_get.call_args[0][0]
        assert "20250505" in call_url

    def test_invalid_date_format_returns_422(self, api_client: TestClient) -> None:
        """不正な date フォーマットは 422 を返すこと。"""
        resp = api_client.get(
            "/api/scraper/discover_today_race_ids",
            params={"date": "2025/05/05"},
        )
        assert resp.status_code == 422


# ── /api/scraper/discover_this_weekend_race_ids ───────────────────────────────


def _make_top_response(race_ids: list[str]) -> MagicMock:
    """Build a mock race_info_top response containing the given race_ids."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "status": "OK",
        "data": {"info": [{"RaceId": rid} for rid in race_ids]},
    }
    return mock_resp


def _make_shutuba_html(race_date: str) -> str:
    """Minimal shutuba HTML containing the given date in the <title> tag."""
    y, m, d = race_date.split("-")
    return (
        f"<html><head><title>テストレース 出馬表 | {int(y)}年{int(m)}月{int(d)}日</title></head>"
        "<body><table class='Shutuba_Table'></table></body></html>"
    )


class TestDiscoverThisWeekendRaceIds:
    def _patch_this_weekend(self, sat: date, sun: date):
        """Patch this_weekend_dates in the scraper router to return fixed dates."""
        return patch(
            "api.routers.scraper.this_weekend_dates",
            return_value=(sat, sun),
        )

    def _clear_cache(self) -> None:
        """Module-level discover cache をテスト間でリセットする。"""
        from api.routers.scraper import _DISCOVER_CACHE
        _DISCOVER_CACHE.clear()

    def setup_method(self) -> None:  # pytest hook: called before each test
        self._clear_cache()

    def test_returns_only_weekend_jra_ids(self, api_client: TestClient) -> None:
        """JRA の土・日 race_id のみ返り、NAR や他週の race_id は除外される。"""
        sat = date(2026, 5, 9)
        sun = date(2026, 5, 10)

        # kaisai_day_key = race_id[:10]
        # 今週土: YYYYMMDD = 20260509, NN = 01
        # 今週日: YYYYMMDD = 20260510, NN = 01
        # 来週土: YYYYMMDD = 20260516, NN = 01
        # NAR  : 20260509 だが venue = 11
        race_ids_all = [
            "202605090501",  # 今週土 / 東京(05) / JRA
            "202605090502",  # 今週土 / 東京(05) / JRA
            "202605100601",  # 今週日 / 中山(06) / JRA
            "202605160501",  # 来週土 / JRA → 除外
            "202605091101",  # 今週土 / NAR(11) → 除外
        ]

        shutuba_html_sat = _make_shutuba_html("2026-05-09")
        shutuba_html_sun = _make_shutuba_html("2026-05-10")
        shutuba_html_next = _make_shutuba_html("2026-05-16")

        mock_top = _make_top_response(race_ids_all)

        async def _async_get(url: str, *args, **kwargs) -> MagicMock:
            """並列 probe 環境では URL ベースで dispatch しないと順序が壊れる。"""
            mock = MagicMock()
            mock.raise_for_status = MagicMock()
            if "api_get_race_info_top" in url:
                # race_info_top JSON は元の mock_top をそのまま返す
                return mock_top
            if "race_id=202605090501" in url:
                mock.text = shutuba_html_sat
            elif "race_id=202605100601" in url:
                mock.text = shutuba_html_sun
            elif "race_id=202605160501" in url:
                mock.text = shutuba_html_next
            else:
                mock.text = "<html></html>"
            return mock

        with (
            self._patch_this_weekend(sat, sun),
            patch(
                "api.routers.scraper.RobotsCache.is_allowed",
                return_value=True,
            ),
            patch(
                "httpx.AsyncClient.get",
                new=AsyncMock(side_effect=_async_get),
            ),
        ):
            resp = api_client.get("/api/scraper/discover_this_weekend_race_ids")

        assert resp.status_code == 200
        data = resp.json()
        assert set(data["race_ids"]) == {"202605090501", "202605090502", "202605100601"}
        assert data["saturday_date"] == "2026-05-09"
        assert data["sunday_date"] == "2026-05-10"
        # 3 つの unique 開催日キー（土x1, 日x1, 来週土x1）をプローブ
        assert data["total_kaisai_days_probed"] == 3

    def test_returns_empty_when_no_jra_races(self, api_client: TestClient) -> None:
        """JRA レースが 0 件の場合は race_ids=[] で 200 を返す。"""
        sat = date(2026, 5, 9)
        sun = date(2026, 5, 10)

        mock_top = _make_top_response([])

        with (
            self._patch_this_weekend(sat, sun),
            patch(
                "api.routers.scraper.RobotsCache.is_allowed",
                return_value=True,
            ),
            patch(
                "httpx.AsyncClient.get",
                new=AsyncMock(return_value=mock_top),
            ),
        ):
            resp = api_client.get("/api/scraper/discover_this_weekend_race_ids")

        assert resp.status_code == 200
        data = resp.json()
        assert data["race_ids"] == []
        assert data["total_kaisai_days_probed"] == 0

    def test_returns_502_on_netkeiba_error(self, api_client: TestClient) -> None:
        """race_info_top への通信エラーは 502 を返す。"""
        sat = date(2026, 5, 9)
        sun = date(2026, 5, 10)

        with (
            self._patch_this_weekend(sat, sun),
            patch(
                "api.routers.scraper.RobotsCache.is_allowed",
                return_value=True,
            ),
            patch(
                "httpx.AsyncClient.get",
                new=AsyncMock(side_effect=Exception("connection refused")),
            ),
        ):
            resp = api_client.get("/api/scraper/discover_this_weekend_race_ids")

        assert resp.status_code == 502

    def test_decodes_eucjp_shutuba_pages(self, api_client: TestClient) -> None:
        """race.netkeiba.com は Content-Type charset 空で EUC-JP を返す。

        httpx のデフォルト UTF-8 デコードのままでは title 内の "YYYY年..." が
        mojibake 化して日付抽出に失敗するため、明示的に encoding="euc-jp" を
        セットする必要がある。実体は httpx.Response (bytes ベース) で組み立てて、
        encoding 切替に依存する挙動を再現する。
        """
        sat = date(2026, 5, 9)
        sun = date(2026, 5, 10)
        race_ids_all = ["202605090501"]

        # 実 race.netkeiba.com 同様、EUC-JP のバイト列を charset なしで返す
        shutuba_html_eucjp = _make_shutuba_html("2026-05-09").encode("euc-jp")
        eucjp_response = httpx.Response(
            200,
            content=shutuba_html_eucjp,
            headers={"Content-Type": "text/html"},  # charset を意図的に省略
        )

        mock_top = _make_top_response(race_ids_all)

        async def _async_get(url: str, *args, **kwargs):
            if "api_get_race_info_top" in url:
                return mock_top
            return eucjp_response

        with (
            self._patch_this_weekend(sat, sun),
            patch(
                "api.routers.scraper.RobotsCache.is_allowed",
                return_value=True,
            ),
            patch(
                "httpx.AsyncClient.get",
                new=AsyncMock(side_effect=_async_get),
            ),
        ):
            resp = api_client.get("/api/scraper/discover_this_weekend_race_ids")

        assert resp.status_code == 200
        # encoding="euc-jp" がセットされていれば日付が抽出され race_id が残る
        assert resp.json()["race_ids"] == ["202605090501"]

    def test_filters_nar_venues(self, api_client: TestClient) -> None:
        """NAR 場コード (11〜) の race_id は shutuba fetch 前に除外される。"""
        sat = date(2026, 5, 9)
        sun = date(2026, 5, 10)

        # NAR only — venue codes 11, 12
        nar_ids = ["202605091101", "202605091201"]
        mock_top = _make_top_response(nar_ids)

        with (
            self._patch_this_weekend(sat, sun),
            patch(
                "api.routers.scraper.RobotsCache.is_allowed",
                return_value=True,
            ),
            patch(
                "httpx.AsyncClient.get",
                new=AsyncMock(return_value=mock_top),
            ),
        ):
            resp = api_client.get("/api/scraper/discover_this_weekend_race_ids")

        assert resp.status_code == 200
        # NAR は groups に入らないので probed=0、race_ids=[]
        assert resp.json()["race_ids"] == []
        assert resp.json()["total_kaisai_days_probed"] == 0

    def test_caches_result_within_ttl(self, api_client: TestClient) -> None:
        """初回 probe 後 TTL 内の 2 回目呼び出しは netkeiba を再 fetch しない。

        race_info_top も含めて httpx.get が 0 回でないことだけを確認すると
        厳密性を欠くので、初回コール直後の 2 回目で `httpx.get` 呼び出し数が
        増えないことをアサートする。
        """
        sat = date(2026, 5, 9)
        sun = date(2026, 5, 10)
        race_ids_all = ["202605090501"]

        shutuba_html = _make_shutuba_html("2026-05-09")
        mock_top = _make_top_response(race_ids_all)

        async def _async_get(url: str, *args, **kwargs):
            if "api_get_race_info_top" in url:
                return mock_top
            mock = MagicMock()
            mock.raise_for_status = MagicMock()
            mock.text = shutuba_html
            return mock

        get_mock = AsyncMock(side_effect=_async_get)

        with (
            self._patch_this_weekend(sat, sun),
            patch(
                "api.routers.scraper.RobotsCache.is_allowed",
                return_value=True,
            ),
            patch("httpx.AsyncClient.get", new=get_mock),
        ):
            r1 = api_client.get("/api/scraper/discover_this_weekend_race_ids")
            calls_after_first = get_mock.call_count
            r2 = api_client.get("/api/scraper/discover_this_weekend_race_ids")
            calls_after_second = get_mock.call_count

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["race_ids"] == r2.json()["race_ids"] == ["202605090501"]
        # 2 回目はキャッシュヒットで httpx を 1 回も叩かない
        assert calls_after_second == calls_after_first

    def test_refresh_query_param_bypasses_cache(self, api_client: TestClient) -> None:
        """?refresh=true は in-process キャッシュを無視して再 fetch する。"""
        sat = date(2026, 5, 9)
        sun = date(2026, 5, 10)
        race_ids_all = ["202605090501"]

        shutuba_html = _make_shutuba_html("2026-05-09")
        mock_top = _make_top_response(race_ids_all)

        async def _async_get(url: str, *args, **kwargs):
            if "api_get_race_info_top" in url:
                return mock_top
            mock = MagicMock()
            mock.raise_for_status = MagicMock()
            mock.text = shutuba_html
            return mock

        get_mock = AsyncMock(side_effect=_async_get)

        with (
            self._patch_this_weekend(sat, sun),
            patch(
                "api.routers.scraper.RobotsCache.is_allowed",
                return_value=True,
            ),
            patch("httpx.AsyncClient.get", new=get_mock),
        ):
            api_client.get("/api/scraper/discover_this_weekend_race_ids")
            calls_after_first = get_mock.call_count
            api_client.get(
                "/api/scraper/discover_this_weekend_race_ids?refresh=true"
            )
            calls_after_refresh = get_mock.call_count

        # refresh=true は再フェッチするので少なくとも 1 件以上増える
        assert calls_after_refresh > calls_after_first
