"""Tests for /api/settings endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_get_settings_defaults(api_client: TestClient) -> None:
    resp = api_client.get("/api/settings")
    assert resp.status_code == 200
    data = resp.json()
    assert "user_agent" in data
    assert "rate_min_seconds" in data
    assert data["rate_min_seconds"] == 3.0
    assert data["win_ev_threshold"] == 1.1


def test_put_settings_partial_update(api_client: TestClient) -> None:
    resp = api_client.put(
        "/api/settings",
        json={"rate_min_seconds": 5.0, "win_ev_threshold": 1.2},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["rate_min_seconds"] == 5.0
    assert data["win_ev_threshold"] == 1.2
    # Unchanged fields should still be defaults
    assert data["rate_max_seconds"] == 6.0


def test_settings_persistence(api_client: TestClient) -> None:
    """PUT then GET should return the updated value."""
    api_client.put("/api/settings", json={"night_min_seconds": 10.0})
    resp = api_client.get("/api/settings")
    assert resp.json()["night_min_seconds"] == 10.0


def test_put_settings_scraper_stopped(api_client: TestClient) -> None:
    resp = api_client.put("/api/settings", json={"scraper_stopped": True})
    assert resp.status_code == 200
    assert resp.json()["scraper_stopped"] is True
