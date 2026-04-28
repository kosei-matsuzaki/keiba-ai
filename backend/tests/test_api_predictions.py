"""Tests for GET /api/predictions/{race_id}."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from keiba_ai.db.models.entry import Entry
from keiba_ai.db.models.horse import Horse
from keiba_ai.db.models.model_run import ModelRun
from keiba_ai.db.models.race import Race


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
    from keiba_ai.core.paths import db_path
    from keiba_ai.db.session import make_engine, session_scope

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
    from keiba_ai.core.paths import db_path
    from keiba_ai.db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        _seed_race_and_entries(session, "PRED_RACE1", n_horses=3)
        _seed_active_model(session, str(tmp_path / "fake_model"))

    # Mock out load_model and predict_race so we don't need a real LightGBM model
    fake_df = pd.DataFrame({
        "horse_id": ["HP_PRED_RACE1_0", "HP_PRED_RACE1_1", "HP_PRED_RACE1_2"],
        "score": [2.0, 1.5, 1.0],
        "win_prob": [0.5, 0.3, 0.2],
        "place_prob": [0.7, 0.5, 0.3],
    })

    with (
        patch("keiba_ai.api.routers.predictions.load_model", return_value=MagicMock()),
        patch("keiba_ai.api.routers.predictions.predict_race", return_value=fake_df),
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
        assert p["top_features"] == []
