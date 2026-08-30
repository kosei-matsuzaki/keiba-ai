"""Tests for /api/models endpoints."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from db.models.model_run import ModelRun


def _seed_runs(session, tmp_path: Path) -> list[int]:
    ids = []
    for i in range(2):
        model_dir = tmp_path / f"model_{i}"
        model_dir.mkdir(parents=True, exist_ok=True)
        run = ModelRun(
            created_at=f"2026-0{i + 1}-01T00:00:00+00:00",
            model_path=str(model_dir),
            params_json=json.dumps({"num_leaves": 63}),
            metrics_json=json.dumps({"valid_ndcg3": 0.5 + i * 0.1}),
            is_active=0,
        )
        session.add(run)
        session.flush()
        ids.append(run.id)
    session.commit()
    return ids


def test_list_models_empty(api_client: TestClient) -> None:
    resp = api_client.get("/api/models")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_models(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    from core.paths import db_path
    from db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        _seed_runs(session, tmp_path)

    with TestClient(app_with_temp_db) as client:
        resp = client.get("/api/models")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_get_model_not_found(api_client: TestClient) -> None:
    resp = api_client.get("/api/models/9999")
    assert resp.status_code == 404


def test_get_model_detail(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    from core.paths import db_path
    from db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        ids = _seed_runs(session, tmp_path)

    with TestClient(app_with_temp_db) as client:
        resp = client.get(f"/api/models/{ids[0]}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == ids[0]
    assert data["params"]["num_leaves"] == 63


def test_activate_model_not_found(api_client: TestClient) -> None:
    resp = api_client.post("/api/models/9999/activate")
    assert resp.status_code == 404


def test_activate_model(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    from core.paths import db_path
    from db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        ids = _seed_runs(session, tmp_path)

    with TestClient(app_with_temp_db) as client:
        resp = client.post(f"/api/models/{ids[1]}/activate")
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_active"] is True

    # Verify in DB
    with session_scope(engine) as session:
        run0 = session.get(ModelRun, ids[0])
        run1 = session.get(ModelRun, ids[1])
        assert run0.is_active == 0
        assert run1.is_active == 1


def test_evaluate_endpoint_returns_job_accepted(
    app_with_temp_db: FastAPI,
) -> None:
    """POST /api/models/{id}/evaluate は即座に job_id を返す (実行は裏)。

    学習時の指標は実運用の賭けルールと別物で、log-loss に至っては学習側に無い。
    画面の「未算出」をここから埋められるようにしている。
    """
    from db.session import session_scope

    with (
        patch("api.routers.models.evaluate", return_value={}) as mock_eval,
        TestClient(app_with_temp_db) as client,
    ):
        # engine は lifespan で作られるので、TestClient に入ってから触る
        with session_scope(app_with_temp_db.state.engine) as session:
            run = ModelRun(
                created_at="2026-08-30T00:00:00",
                model_path="data/models/dummy-nn",
                model_type="nn",
                is_active=0,
            )
            session.add(run)
            session.flush()
            model_id = run.id

        resp = client.post(f"/api/models/{model_id}/evaluate")
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "running"
        # persist=True でないと metrics_json に書き戻らない (画面が埋まらない)
        for _ in range(50):
            if mock_eval.call_count:
                break
            time.sleep(0.05)
        assert mock_eval.call_args.kwargs["persist"] is True


def test_evaluate_endpoint_404_for_unknown_model(app_with_temp_db: FastAPI) -> None:
    with TestClient(app_with_temp_db) as client:
        assert client.post("/api/models/9999/evaluate").status_code == 404


def test_train_endpoint_returns_job_accepted(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    """POST /api/models/train should return JobAccepted immediately without blocking."""
    async def _fake_train(*args, **kwargs) -> dict:
        return {}

    with (
        patch("api.routers.models.train_nn", return_value={}),
        TestClient(app_with_temp_db) as client,
    ):
        resp = client.post("/api/models/train", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "running"
