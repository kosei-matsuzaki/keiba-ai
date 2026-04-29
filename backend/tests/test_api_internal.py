"""Tests for POST /api/internal/shutdown."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient


def test_shutdown_returns_200(api_client: TestClient) -> None:
    """Shutdown endpoint should return 200 and {"ok": True}.

    os.kill is monkeypatched to prevent the test process from being killed.
    """
    with patch("keiba_ai.api.routers.internal.os.kill"):
        resp = api_client.post("/api/internal/shutdown")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}


def test_shutdown_calls_os_kill(api_client: TestClient) -> None:
    """os.kill should be called with the current pid and SIGTERM."""
    import os
    import signal

    with patch("keiba_ai.api.routers.internal.os.kill") as mock_kill:
        api_client.post("/api/internal/shutdown")

    # TestClient runs tasks synchronously in the background — verify call was made
    # (BackgroundTasks are executed before the response is returned in TestClient)
    mock_kill.assert_called_once_with(os.getpid(), signal.SIGTERM)
