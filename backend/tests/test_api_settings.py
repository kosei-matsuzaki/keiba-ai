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


def test_put_settings_partial_update(api_client: TestClient) -> None:
    resp = api_client.put(
        "/api/settings",
        json={"rate_min_seconds": 5.0, "win_min_odds": 1.2},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["rate_min_seconds"] == 5.0
    assert data["win_min_odds"] == 1.2
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


def test_combo_min_hit_prob_defaults_are_returned(api_client: TestClient) -> None:
    """連系の点数は固定値ではなく **的中確率の下限** で決まる。

    券種で的中確率の桁が違う (ワイドの本命は 20% 前後、三連単は 1.6% 前後) ので、
    1 つの数字では表せず券種ごとの dict になっている。
    """
    data = api_client.get("/api/settings").json()
    floors = data["combo_min_hit_prob"]
    assert set(floors) == {"馬連", "ワイド", "馬単", "三連複", "三連単"}
    assert floors["ワイド"] > floors["三連単"]


def test_put_combo_min_hit_prob(api_client: TestClient) -> None:
    resp = api_client.put(
        "/api/settings",
        json={"combo_min_hit_prob": {"馬連": 0.08, "ワイド": 0.25}},
    )
    assert resp.status_code == 200
    assert resp.json()["combo_min_hit_prob"] == {"馬連": 0.08, "ワイド": 0.25}


def test_put_combo_min_hit_prob_rejects_out_of_range(api_client: TestClient) -> None:
    resp = api_client.put("/api/settings", json={"combo_min_hit_prob": {"馬連": 1.5}})
    assert resp.status_code == 422


def test_settings_round_trip(api_client: TestClient) -> None:
    """保存した値がそのまま返る。

    以前 `_dict_to_response` が旧キー名 (`place_min_confidence=`) を渡していて、
    pydantic が未知の引数を黙って捨て、**保存しても応答は既定値**という壊れ方を
    していた。画面上は「保存したのに戻る」ように見える。
    """
    resp = api_client.put(
        "/api/settings",
        json={"place_min_hit_prob": 0.42, "race_budget": 3_000},
    )
    assert resp.status_code == 200
    assert resp.json()["place_min_hit_prob"] == 0.42
    assert resp.json()["race_budget"] == 3_000

    again = api_client.get("/api/settings").json()
    assert again["place_min_hit_prob"] == 0.42
    assert again["race_budget"] == 3_000
