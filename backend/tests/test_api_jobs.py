"""Tests for GET /api/jobs/{job_id} and GET /api/jobs."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient


def test_get_job_not_found(api_client: TestClient) -> None:
    resp = api_client.get("/api/jobs/nonexistent-job-id")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_list_jobs_empty(api_client: TestClient) -> None:
    resp = api_client.get("/api/jobs")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_job_returns_job_info(api_client: TestClient) -> None:
    """Start a job via scraper/run, then retrieve it by job_id."""
    async def _noop(*args, **kwargs) -> dict:
        return {"fetched": 0, "skipped": 0, "errors": 0}

    from keiba_ai.scraper import stop_flag
    stop_flag.clear_stopped()

    with patch("keiba_ai.jobs.ingest.run_ingest", new=_noop):
        run_resp = api_client.post(
            "/api/scraper/run",
            json={"date": "2025-01-01", "limit": 1},
        )
    assert run_resp.status_code == 200
    job_id = run_resp.json()["job_id"]

    resp = api_client.get(f"/api/jobs/{job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == job_id
    assert data["type"] == "ingest"
    assert data["status"] in ("running", "completed", "failed")
    assert "started_at" in data


def test_list_jobs_after_start(api_client: TestClient) -> None:
    """After starting a job, GET /api/jobs returns at least that job."""
    async def _noop(*args, **kwargs) -> dict:
        return {"fetched": 0, "skipped": 0, "errors": 0}

    from keiba_ai.scraper import stop_flag
    stop_flag.clear_stopped()

    with patch("keiba_ai.jobs.ingest.run_ingest", new=_noop):
        api_client.post(
            "/api/scraper/run",
            json={"date": "2025-02-01", "limit": 1},
        )

    resp = api_client.get("/api/jobs")
    assert resp.status_code == 200
    jobs = resp.json()
    assert len(jobs) >= 1
    assert all("job_id" in j for j in jobs)
    assert all("status" in j for j in jobs)


def test_job_info_schema_fields(api_client: TestClient) -> None:
    """Job info response contains all expected fields."""
    async def _noop(*args, **kwargs) -> dict:
        return {"fetched": 0, "skipped": 0, "errors": 0}

    from keiba_ai.scraper import stop_flag
    stop_flag.clear_stopped()

    with patch("keiba_ai.jobs.ingest.run_ingest", new=_noop):
        run_resp = api_client.post(
            "/api/scraper/run",
            json={"date": "2025-03-01", "limit": 1},
        )
    job_id = run_resp.json()["job_id"]

    resp = api_client.get(f"/api/jobs/{job_id}")
    data = resp.json()
    for field in ("job_id", "type", "status", "started_at"):
        assert field in data, f"Missing field: {field}"
