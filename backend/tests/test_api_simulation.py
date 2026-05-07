"""Tests for /api/simulation/active_model validation."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_simulation_rejects_window_too_long(api_client: TestClient):
    """期間が 186 日 (≒ 6 か月) を超えると 400 を返す。

    1 年規模だと逐次 predict + settle で数分かかり HTTP timeout する想定なので、
    バックエンドで早めに弾く。
    """
    response = api_client.get(
        "/api/simulation/active_model",
        params={
            "start": "2024-01-01",
            "end": "2024-12-31",  # 365 日
            "budget": 100_000,
            "strategy": "balanced",
        },
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "期間が長すぎます" in detail


def test_simulation_rejects_invalid_date_format(api_client: TestClient):
    response = api_client.get(
        "/api/simulation/active_model",
        params={
            "start": "2024/01/01",
            "end": "2024-06-30",
            "budget": 100_000,
            "strategy": "balanced",
        },
    )
    assert response.status_code == 400
    assert "YYYY-MM-DD" in response.json()["detail"]


def test_simulation_rejects_end_before_start(api_client: TestClient):
    response = api_client.get(
        "/api/simulation/active_model",
        params={
            "start": "2024-06-30",
            "end": "2024-01-01",
            "budget": 100_000,
            "strategy": "balanced",
        },
    )
    assert response.status_code == 400
    assert "end は start 以降" in response.json()["detail"]


def test_simulation_window_within_cap_proceeds(api_client: TestClient):
    """6 か月以内なら window check をパスし、active model 不在で 503 を返す。

    (window check が active-model check より前に走ることの確認)
    """
    response = api_client.get(
        "/api/simulation/active_model",
        params={
            "start": "2024-01-01",
            "end": "2024-06-30",  # 181 日
            "budget": 100_000,
            "strategy": "balanced",
        },
    )
    # 期間 OK だが active model がないので 503
    assert response.status_code == 503
    assert "アクティブなモデル" in response.json()["detail"]
