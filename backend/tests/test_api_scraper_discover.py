"""Tests for GET /api/scraper/discover_today_race_ids and
GET /api/scraper/discover_this_weekend_race_ids endpoints.

データソースは race_list_sub.html (HTML 断片)。旧 JSON API
(api_get_race_info_top.html) は 2026-07 に空レスポンスを返すようになったため廃止。
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


def _make_race_list_sub_html(race_ids: list[str]) -> str:
    """Build a minimal race_list_sub.html fragment containing the given race_ids."""
    items = "".join(
        f'<li class="RaceList_DataItem">'
        f'<a href="../race/shutuba.html?race_id={rid}&rf=race_list">{i + 1}R</a>'
        f"</li>"
        for i, rid in enumerate(race_ids)
    )
    return (
        '<div class="RaceList_Body RaceList_Top" id="RaceTopRace">'
        '<div class="RaceList_Box clearfix">'
        f'<dl class="RaceList_DataList"><dd><ul>{items}</ul></dd></dl>'
        "</div></div>"
    )


def _make_no_kaisai_html() -> str:
    """開催なし日の race_list_sub.html (空の RaceList_Box) を模す。"""
    return (
        '<div class="RaceList_Body RaceList_Top" id="RaceTopRace">'
        '<div class="RaceList_Box clearfix"></div></div>'
    )


def _make_html_response(html: str) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.content = html.encode("utf-8")
    mock_resp.text = html
    return mock_resp


def _make_empty_body_response() -> MagicMock:
    """Build a mock response that is HTTP 200 but has an empty body.

    netkeiba はメンテナンス等で 200 + 空ボディを返すことがある (2026-07 実績)。
    """
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.content = b""
    return mock_resp


class TestDiscoverTodayRaceIds:
    def test_returns_race_ids_on_success(self, api_client: TestClient) -> None:
        """正常系: race_ids が返ること。"""
        expected_ids = ["202506050101", "202506050102", "202506050201"]
        mock_resp = _make_html_response(_make_race_list_sub_html(expected_ids))

        with (
            patch(
                "scraper.discovery.RobotsCache.is_allowed",
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
        """開催なし: race_ids=[] でも 200 を返すこと（404 ではない）。"""
        mock_resp = _make_html_response(_make_no_kaisai_html())

        with (
            patch(
                "scraper.discovery.RobotsCache.is_allowed",
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
                "scraper.discovery.RobotsCache.is_allowed",
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

    def test_returns_502_with_clear_message_on_empty_body(
        self, api_client: TestClient
    ) -> None:
        """200 + 空ボディ (メンテナンス等) は明確な detail 付きの 502 を返すこと。"""
        with (
            patch(
                "scraper.discovery.RobotsCache.is_allowed",
                return_value=True,
            ),
            patch(
                "httpx.AsyncClient.get",
                new=AsyncMock(return_value=_make_empty_body_response()),
            ),
        ):
            resp = api_client.get("/api/scraper/discover_today_race_ids")

        assert resp.status_code == 502
        assert "空のレスポンス" in resp.json()["detail"]

    def test_returns_502_on_unrecognized_html(self, api_client: TestClient) -> None:
        """race_id もRaceList 構造も無い HTML はパース失敗として 502 を返すこと。"""
        mock_resp = _make_html_response("<html><body>maintenance</body></html>")

        with (
            patch(
                "scraper.discovery.RobotsCache.is_allowed",
                return_value=True,
            ),
            patch(
                "httpx.AsyncClient.get",
                new=AsyncMock(return_value=mock_resp),
            ),
        ):
            resp = api_client.get("/api/scraper/discover_today_race_ids")

        assert resp.status_code == 502
        assert "パース" in resp.json()["detail"]

    def test_accepts_date_query_param(self, api_client: TestClient) -> None:
        """date クエリパラメータを指定できること。"""
        mock_resp = _make_html_response(_make_race_list_sub_html(["202505050101"]))

        with (
            patch(
                "scraper.discovery.RobotsCache.is_allowed",
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


class TestDiscoverThisWeekendRaceIds:
    def _patch_this_weekend(self, sat: date, sun: date):
        """Patch this_weekend_dates in the discovery module to return fixed dates."""
        return patch(
            "scraper.discovery.this_weekend_dates",
            return_value=(sat, sun),
        )

    def _clear_cache(self) -> None:
        """Module-level discover cache をテスト間でリセットする。"""
        from scraper.discovery import _DISCOVER_CACHE
        _DISCOVER_CACHE.clear()

    def setup_method(self) -> None:  # pytest hook: called before each test
        self._clear_cache()

    @staticmethod
    def _dispatch_by_date(responses: dict[str, str]):
        """kaisai_date=YYYYMMDD で応答 HTML を出し分ける AsyncClient.get の side_effect。"""

        async def _async_get(url: str, *args, **kwargs) -> MagicMock:
            for kaisai_date, html in responses.items():
                if f"kaisai_date={kaisai_date}" in url:
                    return _make_html_response(html)
            return _make_html_response(_make_no_kaisai_html())

        return _async_get

    def test_returns_weekend_jra_ids_from_both_days(
        self, api_client: TestClient
    ) -> None:
        """土・日それぞれの race_list_sub から JRA race_id を union して返す。"""
        sat = date(2026, 5, 9)
        sun = date(2026, 5, 10)

        # race_id 形式: YYYY + 場(2) + 回(2) + 日(2) + R(2)。場 01-10=JRA / 11+=NAR。
        sat_html = _make_race_list_sub_html(
            ["202605090501", "202605090502", "202611050901"]  # 3件目は NAR(11=大井)
        )
        sun_html = _make_race_list_sub_html(["202605100601"])

        with (
            self._patch_this_weekend(sat, sun),
            patch(
                "scraper.discovery.RobotsCache.is_allowed",
                return_value=True,
            ),
            patch(
                "httpx.AsyncClient.get",
                new=AsyncMock(
                    side_effect=self._dispatch_by_date(
                        {"20260509": sat_html, "20260510": sun_html}
                    )
                ),
            ),
        ):
            resp = api_client.get("/api/scraper/discover_this_weekend_race_ids")

        assert resp.status_code == 200
        data = resp.json()
        assert set(data["race_ids"]) == {
            "202605090501",
            "202605090502",
            "202605100601",
        }
        assert data["saturday_date"] == "2026-05-09"
        assert data["sunday_date"] == "2026-05-10"
        # unique 開催日キー (race_id[:10]) は 土x1 + 日x1 = 2
        assert data["total_kaisai_days_probed"] == 2

    def test_returns_empty_when_no_jra_races(self, api_client: TestClient) -> None:
        """土日とも開催なしの場合は race_ids=[] で 200 を返す。"""
        sat = date(2026, 5, 9)
        sun = date(2026, 5, 10)

        with (
            self._patch_this_weekend(sat, sun),
            patch(
                "scraper.discovery.RobotsCache.is_allowed",
                return_value=True,
            ),
            patch(
                "httpx.AsyncClient.get",
                new=AsyncMock(
                    return_value=_make_html_response(_make_no_kaisai_html())
                ),
            ),
        ):
            resp = api_client.get("/api/scraper/discover_this_weekend_race_ids")

        assert resp.status_code == 200
        data = resp.json()
        assert data["race_ids"] == []
        assert data["total_kaisai_days_probed"] == 0

    def test_filters_nar_venues(self, api_client: TestClient) -> None:
        """NAR 場コード (11〜) の race_id は除外される。"""
        sat = date(2026, 5, 9)
        sun = date(2026, 5, 10)

        # NAR only — venue code は race_id[4:6] (年の直後 2 桁)。11=大井, 12=川崎。
        nar_html = _make_race_list_sub_html(["202611050901", "202612050901"])

        with (
            self._patch_this_weekend(sat, sun),
            patch(
                "scraper.discovery.RobotsCache.is_allowed",
                return_value=True,
            ),
            patch(
                "httpx.AsyncClient.get",
                new=AsyncMock(return_value=_make_html_response(nar_html)),
            ),
        ):
            resp = api_client.get("/api/scraper/discover_this_weekend_race_ids")

        assert resp.status_code == 200
        assert resp.json()["race_ids"] == []
        assert resp.json()["total_kaisai_days_probed"] == 0

    def test_returns_502_on_netkeiba_error(self, api_client: TestClient) -> None:
        """race_list_sub への通信エラーは 502 を返す。"""
        sat = date(2026, 5, 9)
        sun = date(2026, 5, 10)

        with (
            self._patch_this_weekend(sat, sun),
            patch(
                "scraper.discovery.RobotsCache.is_allowed",
                return_value=True,
            ),
            patch(
                "httpx.AsyncClient.get",
                new=AsyncMock(side_effect=Exception("connection refused")),
            ),
        ):
            resp = api_client.get("/api/scraper/discover_this_weekend_race_ids")

        assert resp.status_code == 502

    def test_returns_502_with_clear_message_on_empty_body(
        self, api_client: TestClient
    ) -> None:
        """race_list_sub が 200 + 空ボディを返した場合も明確な detail 付き 502。"""
        sat = date(2026, 5, 9)
        sun = date(2026, 5, 10)

        with (
            self._patch_this_weekend(sat, sun),
            patch(
                "scraper.discovery.RobotsCache.is_allowed",
                return_value=True,
            ),
            patch(
                "httpx.AsyncClient.get",
                new=AsyncMock(return_value=_make_empty_body_response()),
            ),
        ):
            resp = api_client.get("/api/scraper/discover_this_weekend_race_ids")

        assert resp.status_code == 502
        assert "空のレスポンス" in resp.json()["detail"]

    def test_caches_result_within_ttl(self, api_client: TestClient) -> None:
        """初回取得後 TTL 内の 2 回目呼び出しは netkeiba を再 fetch しない。"""
        sat = date(2026, 5, 9)
        sun = date(2026, 5, 10)

        sat_html = _make_race_list_sub_html(["202605090501"])
        get_mock = AsyncMock(
            side_effect=self._dispatch_by_date({"20260509": sat_html})
        )

        with (
            self._patch_this_weekend(sat, sun),
            patch(
                "scraper.discovery.RobotsCache.is_allowed",
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

        sat_html = _make_race_list_sub_html(["202605090501"])
        get_mock = AsyncMock(
            side_effect=self._dispatch_by_date({"20260509": sat_html})
        )

        with (
            self._patch_this_weekend(sat, sun),
            patch(
                "scraper.discovery.RobotsCache.is_allowed",
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
