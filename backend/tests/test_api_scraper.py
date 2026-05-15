"""Tests for /api/scraper endpoints."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from scraper import stop_flag


def test_scraper_status_defaults(api_client: TestClient) -> None:
    stop_flag.clear_stopped()
    resp = api_client.get("/api/scraper/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["stopped"] is False
    assert data["last_fetched_date"] is None
    assert data["current_job_id"] is None


def test_scraper_stop(api_client: TestClient) -> None:
    stop_flag.clear_stopped()
    resp = api_client.post("/api/scraper/stop")
    assert resp.status_code == 200
    assert stop_flag.is_stopped() is True
    # Cleanup for subsequent tests
    stop_flag.clear_stopped()


def test_scraper_status_after_stop(api_client: TestClient) -> None:
    stop_flag.clear_stopped()
    api_client.post("/api/scraper/stop")
    resp = api_client.get("/api/scraper/status")
    assert resp.status_code == 200
    assert resp.json()["stopped"] is True
    stop_flag.clear_stopped()


def test_scraper_run_returns_job_accepted(api_client: TestClient) -> None:
    """POST /api/scraper/run should accept the job immediately.

    The actual HTTP fetch is never triggered because we don't await the task.
    TestClient uses anyio in its event loop; asyncio.create_task will schedule
    but tests exit before it runs — so no real network call happens.
    """
    stop_flag.clear_stopped()

    # Patch the ingest coroutine so nothing touches the network
    async def _noop(*args, **kwargs) -> dict:
        return {"fetched": 0, "skipped": 0, "errors": 0}

    with patch("jobs.ingest.run_ingest", new=_noop):
        resp = api_client.post(
            "/api/scraper/run",
            json={"date": "2025-01-01", "limit": 1},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "running"


# ── /scraper/fetch_live_odds ──────────────────────────────────────────────────

def test_fetch_live_odds_returns_job_accepted(api_client: TestClient) -> None:
    """POST /api/scraper/fetch_live_odds は 202 と job_id を返すこと。"""
    stop_flag.clear_stopped()

    async def _noop(*args, **kwargs) -> dict:
        return {"fetched": 0, "skipped": 0, "errors": 0}

    with patch("jobs.fetch_live_odds.run_fetch_live_odds", new=_noop):
        resp = api_client.post(
            "/api/scraper/fetch_live_odds",
            json={"race_id": "202506050911"},
        )
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "running"


def test_fetch_live_odds_with_types(api_client: TestClient) -> None:
    """types を指定した場合も 202 を返すこと。"""
    stop_flag.clear_stopped()

    async def _noop(*args, **kwargs) -> dict:
        return {"fetched": 0, "skipped": 0, "errors": 0}

    with patch("jobs.fetch_live_odds.run_fetch_live_odds", new=_noop):
        resp = api_client.post(
            "/api/scraper/fetch_live_odds",
            json={"race_id": "202506050911", "types": ["b1", "b4"]},
        )
    assert resp.status_code == 202


def test_fetch_live_odds_race_id_required(api_client: TestClient) -> None:
    """race_id を省略すると 422 を返すこと。"""
    resp = api_client.post("/api/scraper/fetch_live_odds", json={})
    assert resp.status_code == 422


def test_fetch_live_odds_race_id_must_be_12_digits(api_client: TestClient) -> None:
    """race_id が 12 桁でない場合は 422 を返すこと。"""
    resp = api_client.post(
        "/api/scraper/fetch_live_odds",
        json={"race_id": "short"},
    )
    assert resp.status_code == 422


def test_fetch_live_odds_invalid_type_code(api_client: TestClient) -> None:
    """不正な券種コードは 422 を返すこと。"""
    resp = api_client.post(
        "/api/scraper/fetch_live_odds",
        json={"race_id": "202506050911", "types": ["b99"]},
    )
    assert resp.status_code == 422
