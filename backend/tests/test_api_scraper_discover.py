"""Tests for GET /api/scraper/discover_today_race_ids endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

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
                "keiba_ai.api.routers.scraper.RobotsCache.is_allowed",
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
                "keiba_ai.api.routers.scraper.RobotsCache.is_allowed",
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
                "keiba_ai.api.routers.scraper.RobotsCache.is_allowed",
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
                "keiba_ai.api.routers.scraper.RobotsCache.is_allowed",
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
                "keiba_ai.api.routers.scraper.RobotsCache.is_allowed",
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
