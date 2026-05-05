"""Tests for POST /api/scraper/run_shutuba endpoint."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from keiba_ai.scraper import stop_flag


def test_run_shutuba_returns_job_accepted(api_client: TestClient) -> None:
    """POST /api/scraper/run_shutuba should accept the job and return 202 (date path)."""
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
    """不正な日付フォーマットは 422 を返すこと。"""
    resp = api_client.post(
        "/api/scraper/run_shutuba",
        json={"date": "2025/05/05"},
    )
    assert resp.status_code == 422


def test_run_shutuba_without_limit(api_client: TestClient) -> None:
    """limit なしでも受け付けること（date パス）。"""
    stop_flag.clear_stopped()

    async def _noop(*args, **kwargs) -> dict:
        return {"fetched": 0, "skipped": 0, "errors": 0}

    with patch("keiba_ai.jobs.ingest_shutuba.run_ingest_shutuba", new=_noop):
        resp = api_client.post(
            "/api/scraper/run_shutuba",
            json={"date": "2025-05-05"},
        )
    assert resp.status_code == 202


def test_run_shutuba_race_ids_path(api_client: TestClient) -> None:
    """race_ids 指定時も 202 を返し、calendar fetch を skip すること。"""
    stop_flag.clear_stopped()

    called_with: dict = {}

    async def _capture(date_str, client, session, *, limit=None, race_ids=None) -> dict:
        called_with["date_str"] = date_str
        called_with["race_ids"] = race_ids
        called_with["limit"] = limit
        return {"fetched": 1, "skipped": 0, "errors": 0}

    with patch("keiba_ai.jobs.ingest_shutuba.run_ingest_shutuba", new=_capture):
        resp = api_client.post(
            "/api/scraper/run_shutuba",
            json={"race_ids": ["202506050911", "202506050912"]},
        )
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "running"


def test_run_shutuba_race_ids_and_date_race_ids_wins(api_client: TestClient) -> None:
    """date と race_ids 両方指定時は race_ids が渡されること（race_ids 優先）。"""
    stop_flag.clear_stopped()

    called_with: dict = {}

    async def _capture(date_str, client, session, *, limit=None, race_ids=None) -> dict:
        called_with["date_str"] = date_str
        called_with["race_ids"] = race_ids
        return {"fetched": 1, "skipped": 0, "errors": 0}

    with patch("keiba_ai.jobs.ingest_shutuba.run_ingest_shutuba", new=_capture):
        resp = api_client.post(
            "/api/scraper/run_shutuba",
            json={"date": "2025-05-05", "race_ids": ["202506050911"]},
        )
    assert resp.status_code == 202


def test_run_shutuba_requires_date_or_race_ids(api_client: TestClient) -> None:
    """date も race_ids も指定しない場合は 422 を返すこと。"""
    resp = api_client.post("/api/scraper/run_shutuba", json={})
    assert resp.status_code == 422


def test_run_shutuba_validates_race_id_format(api_client: TestClient) -> None:
    """race_ids に 12 桁以外の値が含まれると 422 を返すこと。"""
    resp = api_client.post(
        "/api/scraper/run_shutuba",
        json={"race_ids": ["short"]},
    )
    assert resp.status_code == 422
