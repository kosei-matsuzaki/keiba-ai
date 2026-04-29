"""Tests for top_features in GET /api/predictions/{race_id}."""

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
from keiba_ai.features.builder import FEATURE_COLUMNS


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
        hid = f"TF_{race_id}_{i}"
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


def test_predictions_top_features_non_empty(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    """top_features should be a non-empty list of valid FEATURE_COLUMNS strings."""
    from keiba_ai.core.paths import db_path
    from keiba_ai.db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        _seed_race_and_entries(session, "TF_RACE1", n_horses=3)
        _seed_active_model(session, str(tmp_path / "fake_model"))

    # Return DataFrame that includes top_features column (as predict_race_with_shap would)
    fake_df = pd.DataFrame({
        "horse_id": ["TF_TF_RACE1_0", "TF_TF_RACE1_1", "TF_TF_RACE1_2"],
        "score": [2.0, 1.5, 1.0],
        "win_prob": [0.5, 0.3, 0.2],
        "place_prob": [0.7, 0.5, 0.3],
        "top_features": [
            ["odds_win", "distance", "age"],
            ["log_odds_win", "post_position", "horse_weight"],
            ["popularity", "n_runners", "distance"],
        ],
    })

    with (
        patch("keiba_ai.api.routers.predictions.load_model", return_value=MagicMock()),
        patch("keiba_ai.api.routers.predictions.predict_race_with_shap", return_value=fake_df),
        TestClient(app_with_temp_db) as client,
    ):
        resp = client.get("/api/predictions/TF_RACE1")

    assert resp.status_code == 200
    data = resp.json()
    assert data["race_id"] == "TF_RACE1"
    assert len(data["predictions"]) == 3

    for p in data["predictions"]:
        features = p["top_features"]
        assert isinstance(features, list), "top_features must be a list"
        assert len(features) > 0, "top_features must not be empty"
        for f in features:
            assert isinstance(f, str), f"Feature names must be strings, got {type(f)}"
            assert f in FEATURE_COLUMNS, f"Unknown feature name: {f!r}"


def test_predictions_top_features_empty_fallback(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    """If predict_race_with_shap returns None/empty top_features, response is empty list."""
    from keiba_ai.core.paths import db_path
    from keiba_ai.db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        _seed_race_and_entries(session, "TF_RACE2", n_horses=2)
        _seed_active_model(session, str(tmp_path / "fake_model2"))

    # top_features is None (simulate SHAP failure / fallback)
    fake_df = pd.DataFrame({
        "horse_id": ["TF_TF_RACE2_0", "TF_TF_RACE2_1"],
        "score": [2.0, 1.0],
        "win_prob": [0.6, 0.4],
        "place_prob": [0.8, 0.6],
        "top_features": [None, None],
    })

    with (
        patch("keiba_ai.api.routers.predictions.load_model", return_value=MagicMock()),
        patch("keiba_ai.api.routers.predictions.predict_race_with_shap", return_value=fake_df),
        TestClient(app_with_temp_db) as client,
    ):
        resp = client.get("/api/predictions/TF_RACE2")

    assert resp.status_code == 200
    for p in resp.json()["predictions"]:
        assert p["top_features"] == []
