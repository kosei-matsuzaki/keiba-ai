"""Tests for /api/simulation/active_model validation."""

from __future__ import annotations

import inspect

from fastapi.testclient import TestClient

from api.routers import simulation as simulation_router


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
            "race_budget": 5_000,
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
            "race_budget": 5_000,
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
            "race_budget": 5_000,
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
            "race_budget": 5_000,
        },
    )
    # 期間 OK だが active model がないので 503
    assert response.status_code == 503
    assert "アクティブなモデル" in response.json()["detail"]


# ── /api/simulation/start (background job) ──────────────────────────────────


def test_simulation_start_rejects_window_too_long(api_client: TestClient):
    """background job 版でも 1 年超は弾く。"""
    response = api_client.post(
        "/api/simulation/start",
        params={
            "start": "2023-01-01",
            "end": "2024-12-31",  # 730 日
            "race_budget": 5_000,
        },
    )
    assert response.status_code == 400
    assert "期間が長すぎます" in response.json()["detail"]


def test_simulation_start_returns_503_without_active_model(api_client: TestClient):
    """active model 不在で 503。"""
    response = api_client.post(
        "/api/simulation/start",
        params={
            "start": "2024-01-01",
            "end": "2024-06-30",
            "race_budget": 5_000,
        },
    )
    assert response.status_code == 503
    assert "アクティブなモデル" in response.json()["detail"]


def test_simulation_start_rejects_budget_below_one_point(api_client: TestClient):
    """1 点も買えない予算は受け付けない (戦略プリセットは廃止した)。"""
    response = api_client.post(
        "/api/simulation/start",
        params={"start": "2024-01-01", "end": "2024-03-01", "race_budget": 50},
    )
    assert response.status_code == 422


def test_simulation_start_accepts_full_year_window(api_client: TestClient):
    """sync 版は 6 か月止まりだが、background 版は 1 年まで OK (active model 不在で 503 になる)。"""
    response = api_client.post(
        "/api/simulation/start",
        params={
            "start": "2024-01-01",
            "end": "2024-12-31",  # 365 日
            "race_budget": 5_000,
        },
    )
    # 期間 OK で先に進む → active model がないので 503
    assert response.status_code == 503


# ── model_id 指定 ────────────────────────────────────────────────────────────


def test_simulation_sync_unknown_model_id_404(api_client: TestClient):
    """存在しない model_id を指定すると 404 (active fallback ではない)。"""
    response = api_client.get(
        "/api/simulation/active_model",
        params={
            "start": "2024-01-01",
            "end": "2024-06-30",
            "race_budget": 5_000,
            "model_id": 99999,
        },
    )
    assert response.status_code == 404
    assert "model id=99999" in response.json()["detail"]


def test_simulation_start_unknown_model_id_404(api_client: TestClient):
    response = api_client.post(
        "/api/simulation/start",
        params={
            "start": "2024-01-01",
            "end": "2024-06-30",
            "race_budget": 5_000,
            "model_id": 99999,
        },
    )
    assert response.status_code == 404
    assert "model id=99999" in response.json()["detail"]


def test_both_routes_pass_the_settings_derived_bet_params():
    """同期・ジョブの両方が settings 由来の買い方を engine に渡すこと。

    2 つの経路は 93% 同じ形をしていて、片方だけ引数が落ちても誰も気づかない。
    実際に win_min_odds が両方で落ちていて、Settings で単勝オッズ下限を変えても
    シミュレーションだけ engine 側の既定 1.1 で走っていた (着順精度は変わらない
    ので、テストからも結果からも見えなかった)。
    """
    for fn in (simulation_router.run_simulation, simulation_router.start_simulation_job):
        src = inspect.getsource(fn)
        assert "resolve_betting_settings" in src, fn.__name__
        for kwarg in (
            "win_min_odds=bet_settings.win_min_odds",
            "probability_model_path=bet_settings.probability_model_path",
            "place_min_confidence=bet_settings.place_min_confidence",
            "min_hit_prob_by_bet_type=bet_settings.combo_min_hit_prob",
        ):
            assert kwarg in src, f"{fn.__name__}: {kwarg}"
