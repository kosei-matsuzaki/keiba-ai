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


def test_metrics_summary_falls_back_to_test_when_valid_is_nan(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    """`--valid-months 0` で学習したモデルでは valid_ndcg* が NaN になる。
    その場合は test_ndcg* に fallback して何かしら値が出ることを確認する
    (Phase 1 で全 MetricCard が「—」表示になっていた回帰の防止)。
    """
    from keiba_ai.core.paths import db_path
    from keiba_ai.db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        # Phase 1 学習結果と同形式: valid 系は NaN、test 系のみ有効
        run = ModelRun(
            created_at="2026-05-02T22:40:15+00:00",
            model_path="/models/phase1",
            metrics_json=json.dumps(
                {
                    "valid_ndcg1": float("nan"),
                    "valid_ndcg3": float("nan"),
                    "test_ndcg1": 0.486,
                    "test_ndcg3": 0.524,
                }
            ),
            is_active=1,
        )
        session.add(run)
        session.commit()

    with TestClient(app_with_temp_db) as client:
        resp = client.get("/api/metrics/summary")
    assert resp.status_code == 200
    data = resp.json()
    # valid が NaN なので test_* に fallback
    assert data["ndcg1"] == 0.486
    assert data["ndcg3"] == 0.524
    # evaluate.py --persist が走っていないので top1_hit 系は依然 None
    assert data["top1_hit"] is None
    assert data["payback_win"] is None


def test_metrics_summary_picks_persisted_evaluation_keys(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    """evaluate.py --persist で merge された top1_hit / payback_win 等が
    Dashboard 用 endpoint に反映されることを確認する。"""
    from keiba_ai.core.paths import db_path
    from keiba_ai.db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        run = ModelRun(
            created_at="2026-05-02T22:40:15+00:00",
            model_path="/models/persisted",
            metrics_json=json.dumps(
                {
                    "valid_ndcg1": float("nan"),
                    "valid_ndcg3": float("nan"),
                    "test_ndcg1": 0.486,
                    "test_ndcg3": 0.524,
                    # 以下は evaluate.py --persist が merge した想定
                    "top1_hit": 0.278,
                    "place_hit": 0.861,
                    "payback_win": 0.885,
                    "payback_place": 0.929,
                    "n_races": 108,
                }
            ),
            is_active=1,
        )
        session.add(run)
        session.commit()

    with TestClient(app_with_temp_db) as client:
        resp = client.get("/api/metrics/summary")
    data = resp.json()
    assert data["top1_hit"] == 0.278
    assert data["place_hit"] == 0.861
    assert data["payback_win"] == 0.885
    assert data["n_races"] == 108
