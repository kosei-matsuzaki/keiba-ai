"""Tests for /api/models endpoints."""

from __future__ import annotations

import json
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


def test_train_endpoint_returns_job_accepted(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    """POST /api/models/train should return JobAccepted immediately without blocking."""
    async def _fake_train(*args, **kwargs) -> dict:
        return {}

    with (
        patch("api.routers.models.train", return_value={}),
        TestClient(app_with_temp_db) as client,
    ):
        resp = client.post("/api/models/train", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "running"
