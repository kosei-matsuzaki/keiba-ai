"""Tests for POST /api/scraper/run_shutuba endpoint."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from keiba_ai.scraper import stop_flag


def test_run_shutuba_returns_job_accepted(api_client: TestClient) -> None:
    """POST /api/scraper/run_shutuba should accept the job and return 202."""
    stop_flag.clear_stopped()

    async def _noop(*args, **kwargs) -> dict:
        return {"fetched": 0, "skipped": 0, "errors": 0}

    with patch("keiba_ai.jobs.ingest_shutuba.run_ingest_shutuba", new=_noop):
        resp = api_client.post(
            "/api/scraper/run_shutuba",
            json={"date": "2025-05-05", "limit": 1},
        )
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "running"


def test_run_shutuba_validates_date_format(api_client: TestClient) -> None:
    """不正な日付フォーマットは 422 Unprocessable Entity を返すこと。"""
    resp = api_client.post(
        "/api/scraper/run_shutuba",
        json={"date": "2025/05/05"},
    )
    assert resp.status_code == 422


def test_run_shutuba_without_limit(api_client: TestClient) -> None:
    """limit なしでも受け付けること。"""
    stop_flag.clear_stopped()

    async def _noop(*args, **kwargs) -> dict:
        return {"fetched": 0, "skipped": 0, "errors": 0}

    with patch("keiba_ai.jobs.ingest_shutuba.run_ingest_shutuba", new=_noop):
        resp = api_client.post(
            "/api/scraper/run_shutuba",
            json={"date": "2025-05-05"},
        )
    assert resp.status_code == 202
