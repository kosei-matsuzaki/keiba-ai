"""Tests for /api/metrics endpoints."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from keiba_ai.db.models.model_run import ModelRun


def _seed_model_runs(session) -> None:
    for i in range(3):
        metrics = {
            "valid_ndcg1": 0.3 + i * 0.05,
            "valid_ndcg3": 0.5 + i * 0.05,
        }
        run = ModelRun(
            created_at=f"2026-0{i + 1}-01T00:00:00+00:00",
            model_path=f"/models/m{i}",
            metrics_json=json.dumps(metrics),
            is_active=1 if i == 2 else 0,
        )
        session.add(run)
    session.commit()


def test_metrics_summary_no_models(api_client: TestClient) -> None:
    resp = api_client.get("/api/metrics/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] is None


def test_metrics_summary_with_models(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    from keiba_ai.core.paths import db_path
    from keiba_ai.db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        _seed_model_runs(session)

    with TestClient(app_with_temp_db) as client:
        resp = client.get("/api/metrics/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ndcg3"] is not None
    assert data["model_id"] is not None


def test_metrics_timeseries_empty(api_client: TestClient) -> None:
    resp = api_client.get("/api/metrics/timeseries?metric=ndcg3")
    assert resp.status_code == 200
    data = resp.json()
    assert data["metric"] == "ndcg3"
    assert data["points"] == []


def test_metrics_timeseries_with_runs(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    from keiba_ai.core.paths import db_path
    from keiba_ai.db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        _seed_model_runs(session)

    with TestClient(app_with_temp_db) as client:
        resp = client.get("/api/metrics/timeseries?metric=ndcg3&range=180d")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["points"]) == 3
    # Each point that has ndcg3 in metrics_json should have a non-None value
    values_with_data = [p["value"] for p in data["points"] if p["value"] is not None]
    assert len(values_with_data) == 3
