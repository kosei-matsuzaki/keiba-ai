"""Tests for GET /api/predictions/{race_id}."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from db.models.entry import Entry
from db.models.horse import Horse
from db.models.model_run import ModelRun
from db.models.race import Race


def _seed_race_and_entries(session, race_id: str, n_horses: int = 4) -> None:
    session.add(Race(
        race_id=race_id,
        date=date.today().isoformat(),
        course="東京",
        surface="芝",
        distance=2000,
        n_runners=n_horses,
    ))
    session.flush()
    for i in range(n_horses):
        hid = f"HP_{race_id}_{i}"
        if not session.get(Horse, hid):
            session.add(Horse(horse_id=hid, name=None))
        session.flush()
        session.add(Entry(
            race_id=race_id,
            horse_id=hid,
            post_position=i + 1,
            age=4,
            sex="牡",
            odds_win=5.0 + i,
            popularity=i + 1,
            horse_weight=480,
        ))
    session.commit()


def _seed_active_model(session, model_path: str) -> int:
    run = ModelRun(
        created_at="2026-01-01T00:00:00+00:00",
        model_path=model_path,
        is_active=1,
        params_json=None,
        metrics_json=None,
    )
    session.add(run)
    session.commit()
    return run.id


def test_predictions_no_active_model(api_client: TestClient) -> None:
    resp = api_client.get("/api/predictions/SOMERACE")
    assert resp.status_code == 503


def test_predictions_race_not_found(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    from core.paths import db_path
    from db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        _seed_active_model(session, str(tmp_path / "fake_model"))

    with TestClient(app_with_temp_db) as client:
        resp = client.get("/api/predictions/NONEXISTENT_RACE")
    assert resp.status_code == 404


def test_predictions_success(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    """Happy path: active model + known race → list of HorsePrediction."""
    from core.paths import db_path
    from db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        _seed_race_and_entries(session, "PRED_RACE1", n_horses=3)
        _seed_active_model(session, str(tmp_path / "fake_model"))

    # Mock out load_model and predict_race_gbdt so we don't need a real LightGBM model
    fake_df = pd.DataFrame({
        "horse_id": ["HP_PRED_RACE1_0", "HP_PRED_RACE1_1", "HP_PRED_RACE1_2"],
        "score": [2.0, 1.5, 1.0],
        "win_prob": [0.5, 0.3, 0.2],
        "place_prob": [0.7, 0.5, 0.3],
    })

    # fake_df does not include top_features — router uses list(row.get(...) or []) fallback
    with (
        patch("api.routers.predictions.load_model", return_value=MagicMock()),
        patch("api.routers.predictions.predict_race_with_shap_gbdt", return_value=fake_df),
        TestClient(app_with_temp_db) as client,
    ):
        resp = client.get("/api/predictions/PRED_RACE1")

    assert resp.status_code == 200
    data = resp.json()
    assert data["race_id"] == "PRED_RACE1"
    assert len(data["predictions"]) == 3
    for p in data["predictions"]:
        assert "win_prob" in p
        assert "place_prob" in p
        assert isinstance(p["top_features"], list)


def test_predictions_include_combinations_default(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    """By default (include_combinations=true), response contains a combinations field."""
    from core.paths import db_path
    from db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        _seed_race_and_entries(session, "COMBO_RACE1", n_horses=4)
        _seed_active_model(session, str(tmp_path / "fake_model_combo1"))

    fake_df = pd.DataFrame({
        "horse_id": [f"HP_COMBO_RACE1_{i}" for i in range(4)],
        "score": [2.0, 1.5, 1.0, 0.5],
        "win_prob": [0.4, 0.3, 0.2, 0.1],
        "place_prob": [0.7, 0.6, 0.5, 0.3],
    })

    with (
        patch("api.routers.predictions.load_model", return_value=MagicMock()),
        patch("api.routers.predictions.predict_race_with_shap_gbdt", return_value=fake_df),
        patch(
            "api.routers.predictions.predict_race_with_combinations_gbdt",
            return_value={
                "単勝": [], "複勝": [], "馬連": [], "ワイド": [],
                "馬単": [], "三連複": [], "三連単": [],
            },
        ),
        TestClient(app_with_temp_db) as client,
    ):
        resp = client.get("/api/predictions/COMBO_RACE1")

    assert resp.status_code == 200
    data = resp.json()
    assert "combinations" in data
    assert data["combinations"] is not None
    combs = data["combinations"]
    for key in ("tansho", "fukusho", "umaren", "wide", "umatan", "sanrenpuku", "sanrentan"):
        assert key in combs, f"Missing key: {key}"


def test_predictions_include_combinations_false(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    """include_combinations=false skips combination computation (combinations is null)."""
    from core.paths import db_path
    from db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        _seed_race_and_entries(session, "COMBO_RACE2", n_horses=3)
        _seed_active_model(session, str(tmp_path / "fake_model_combo2"))

    fake_df = pd.DataFrame({
        "horse_id": [f"HP_COMBO_RACE2_{i}" for i in range(3)],
        "score": [2.0, 1.5, 1.0],
        "win_prob": [0.5, 0.3, 0.2],
        "place_prob": [0.7, 0.5, 0.3],
    })

    with (
        patch("api.routers.predictions.load_model", return_value=MagicMock()),
        patch("api.routers.predictions.predict_race_with_shap_gbdt", return_value=fake_df),
        TestClient(app_with_temp_db) as client,
    ):
        resp = client.get("/api/predictions/COMBO_RACE2?include_combinations=false")

    assert resp.status_code == 200
    data = resp.json()
    assert data["combinations"] is None


# ── /predictions/bulk tests ───────────────────────────────────────────────────

def test_bulk_predictions_no_active_model(api_client: TestClient) -> None:
    """active モデルが無い場合は空の predictions map を返すこと（503 でなく 200）。"""
    resp = api_client.get("/api/predictions/bulk?race_ids=202406010101,202406010102")
    assert resp.status_code == 200
    data = resp.json()
    assert "predictions" in data
    # 全 race が空の top_horses で返ること
    for race_id in ("202406010101", "202406010102"):
        assert race_id in data["predictions"]
        assert data["predictions"][race_id]["top_horses"] == []


def test_bulk_predictions_empty_race_ids(api_client: TestClient) -> None:
    """race_ids 省略 / 空文字の場合は空の predictions map を返すこと。"""
    resp = api_client.get("/api/predictions/bulk")
    assert resp.status_code == 200
    data = resp.json()
    assert data["predictions"] == {}


def test_bulk_predictions_success(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    """正常系: active モデルあり + entries あり → top_horses が返ること。"""
    from core.paths import db_path
    from db.session import make_engine, session_scope

    RACE_ID = "BULK_RACE1"

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        _seed_race_and_entries(session, RACE_ID, n_horses=4)
        _seed_active_model(session, str(tmp_path / "fake_model_bulk1"))

    fake_df = pd.DataFrame({
        "horse_id": [f"HP_{RACE_ID}_{i}" for i in range(4)],
        "score": [2.0, 1.5, 1.0, 0.5],
        "win_prob": [0.4, 0.3, 0.2, 0.1],
        "place_prob": [0.7, 0.6, 0.5, 0.3],
        "post_position": [1, 2, 3, 4],
    })

    with (
        patch("api.routers.predictions.load_model", return_value=MagicMock()),
        patch("api.routers.predictions.predict_race_gbdt", return_value=fake_df.head(3)),
        TestClient(app_with_temp_db) as client,
    ):
        resp = client.get(f"/api/predictions/bulk?race_ids={RACE_ID}&top_n=3")

    assert resp.status_code == 200
    data = resp.json()
    assert RACE_ID in data["predictions"]
    top_horses = data["predictions"][RACE_ID]["top_horses"]
    assert len(top_horses) == 3
    for h in top_horses:
        assert "win_prob" in h
        assert "horse_name" in h
        assert "post_position" in h
