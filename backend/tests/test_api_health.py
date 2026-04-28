"""Tests for GET /api/health."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_ok(api_client: TestClient) -> None:
    resp = api_client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "db_path" in data


def test_health_version(api_client: TestClient) -> None:
    resp = api_client.get("/api/health")
    assert resp.json()["version"] == "0.1.0"
