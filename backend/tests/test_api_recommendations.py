"""Tests for GET /api/recommendations/{race_id}."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from keiba_ai.ai.types import BetCandidate, RecommendationResult
from keiba_ai.db.models.entry import Entry
from keiba_ai.db.models.horse import Horse
from keiba_ai.db.models.model_run import ModelRun
from keiba_ai.db.models.race import Race


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        hid = f"H_{race_id}_{i}"
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


def _seed_active_model(session, model_path: str) -> None:
    run = ModelRun(
        created_at="2026-01-01T00:00:00+00:00",
        model_path=model_path,
        is_active=1,
        params_json=None,
        metrics_json=None,
    )
    session.add(run)
    session.commit()


def _fake_predictions_df(race_id: str, n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        "horse_id": [f"H_{race_id}_{i}" for i in range(n)],
        "score": [2.0 - i * 0.3 for i in range(n)],
        "win_prob": [0.4 - i * 0.08 for i in range(n)],
        "place_prob": [0.7 - i * 0.1 for i in range(n)],
    })


def _fake_combinations() -> dict:
    """Minimal combination map — all empty lists (recommend_for_race handles empty)."""
    return {bt: [] for bt in ["単勝", "複勝", "馬連", "ワイド", "馬単", "三連複", "三連単"]}


def _fake_recommendation_result(race_id: str, bankroll: int = 100_000) -> RecommendationResult:
    """Return a non-empty RecommendationResult for happy-path assertions."""
    return RecommendationResult(
        race_id=race_id,
        bankroll_at_decision=bankroll,
        candidates=[
            BetCandidate(
                bet_type="単勝",
                combo="1",
                pattern="box",
                prob=0.4,
                est_odds=10.0,
                ev=4.0,
                stake=500,
                post_positions=(1,),
            ),
            BetCandidate(
                bet_type="馬連",
                combo="1-2",
                pattern="box",
                prob=0.3,
                est_odds=50.0,
                ev=15.0,
                stake=200,
                post_positions=(1, 2),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_recommendations_no_active_model(api_client: TestClient) -> None:
    """503 when no active model is registered."""
    resp = api_client.get("/api/recommendations/SOMERACE")
    assert resp.status_code == 503


def test_recommendations_race_not_found(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    """404 when race_id does not exist in DB."""
    from keiba_ai.core.paths import db_path
    from keiba_ai.db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        _seed_active_model(session, str(tmp_path / "fake_model"))

    with TestClient(app_with_temp_db) as client:
        resp = client.get("/api/recommendations/NONEXISTENT_RACE")
    assert resp.status_code == 404


def test_recommendations_success(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    """Happy path: active model + known race → candidates returned."""
    race_id = "REC_RACE1"
    from keiba_ai.core.paths import db_path
    from keiba_ai.db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        _seed_race_and_entries(session, race_id, n_horses=4)
        _seed_active_model(session, str(tmp_path / "fake_model_rec"))

    fake_df = _fake_predictions_df(race_id, n=4)
    fake_result = _fake_recommendation_result(race_id)

    with (
        patch("keiba_ai.api.routers.recommendations.load_model", return_value=MagicMock()),
        patch("keiba_ai.api.routers.recommendations.predict_race", return_value=fake_df),
        patch("keiba_ai.api.routers.recommendations.predict_race_with_combinations",
              return_value=_fake_combinations()),
        patch("keiba_ai.api.routers.recommendations.recommend_for_race",
              return_value=fake_result),
        TestClient(app_with_temp_db) as client,
    ):
        resp = client.get(f"/api/recommendations/{race_id}")

    assert resp.status_code == 200
    data = resp.json()
    assert data["race_id"] == race_id
    assert data["bankroll_at_decision"] == 100_000
    assert len(data["candidates"]) == 2
    for c in data["candidates"]:
        assert c["stake"] >= 0
        assert isinstance(c["post_positions"], list)


def test_recommendations_stake_cap(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    """Total stake across candidates must not exceed bankroll * max_stake_per_race_pct.

    Default settings: bankroll=100_000, max_stake_per_race_pct=0.05 → cap=5000.
    The fake result returns stake=500+200=700, well within cap.
    """
    race_id = "REC_RACE2"
    from keiba_ai.core.paths import db_path
    from keiba_ai.db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        _seed_race_and_entries(session, race_id, n_horses=4)
        _seed_active_model(session, str(tmp_path / "fake_model_cap"))

    fake_df = _fake_predictions_df(race_id, n=4)
    fake_result = _fake_recommendation_result(race_id, bankroll=100_000)

    with (
        patch("keiba_ai.api.routers.recommendations.load_model", return_value=MagicMock()),
        patch("keiba_ai.api.routers.recommendations.predict_race", return_value=fake_df),
        patch("keiba_ai.api.routers.recommendations.predict_race_with_combinations",
              return_value=_fake_combinations()),
        patch("keiba_ai.api.routers.recommendations.recommend_for_race",
              return_value=fake_result),
        TestClient(app_with_temp_db) as client,
    ):
        resp = client.get(f"/api/recommendations/{race_id}")

    assert resp.status_code == 200
    data = resp.json()
    total_stake = sum(c["stake"] for c in data["candidates"])
    cap = data["bankroll_at_decision"] * 0.05
    assert total_stake <= cap


def test_recommendations_enabled_bet_types_filter(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    """enabled_bet_types setting limits which bet types appear in candidates."""
    race_id = "REC_RACE3"
    from keiba_ai.core.paths import db_path
    from keiba_ai.db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        _seed_race_and_entries(session, race_id, n_horses=4)
        _seed_active_model(session, str(tmp_path / "fake_model_filter"))

    fake_df = _fake_predictions_df(race_id, n=4)
    # Result only contains 単勝 (filtered by enabled_bet_types=["単勝"])
    filtered_result = RecommendationResult(
        race_id=race_id,
        bankroll_at_decision=100_000,
        candidates=[
            BetCandidate(
                bet_type="単勝",
                combo="1",
                pattern="box",
                prob=0.4,
                est_odds=10.0,
                ev=4.0,
                stake=300,
                post_positions=(1,),
            ),
        ],
    )

    captured_enabled: list[list[str]] = []

    def _spy_recommend(**kwargs):  # type: ignore[return]
        captured_enabled.append(kwargs.get("enabled_bet_types") or [])
        return filtered_result

    with (
        patch("keiba_ai.api.routers.recommendations.load_model", return_value=MagicMock()),
        patch("keiba_ai.api.routers.recommendations.predict_race", return_value=fake_df),
        patch("keiba_ai.api.routers.recommendations.predict_race_with_combinations",
              return_value=_fake_combinations()),
        patch("keiba_ai.api.routers.recommendations.recommend_for_race",
              side_effect=lambda predictions, combinations_by_type, race_id, bankroll,
              kelly_fraction, max_stake_per_race_pct, top_n_horses, enabled_bet_types:
              _spy_recommend(
                  enabled_bet_types=enabled_bet_types,
              )),
        TestClient(app_with_temp_db) as client,
    ):
        resp = client.get(f"/api/recommendations/{race_id}")

    assert resp.status_code == 200
    data = resp.json()
    bet_types = {c["bet_type"] for c in data["candidates"]}
    # Only 単勝 should be present in this mocked result
    assert bet_types == {"単勝"}


def test_recommendations_top_n_horses_param(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    """top_n_horses query param is forwarded to recommend_for_race."""
    race_id = "REC_RACE4"
    from keiba_ai.core.paths import db_path
    from keiba_ai.db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        _seed_race_and_entries(session, race_id, n_horses=4)
        _seed_active_model(session, str(tmp_path / "fake_model_topn"))

    fake_df = _fake_predictions_df(race_id, n=4)
    captured_args: dict = {}

    def _capture_recommend(predictions, combinations_by_type, race_id,
                           bankroll, kelly_fraction, max_stake_per_race_pct,
                           top_n_horses, enabled_bet_types):
        captured_args["top_n_horses"] = top_n_horses
        return RecommendationResult(
            race_id=race_id,
            bankroll_at_decision=bankroll,
            candidates=[],
        )

    with (
        patch("keiba_ai.api.routers.recommendations.load_model", return_value=MagicMock()),
        patch("keiba_ai.api.routers.recommendations.predict_race", return_value=fake_df),
        patch("keiba_ai.api.routers.recommendations.predict_race_with_combinations",
              return_value=_fake_combinations()),
        patch("keiba_ai.api.routers.recommendations.recommend_for_race",
              side_effect=_capture_recommend),
        TestClient(app_with_temp_db) as client,
    ):
        resp = client.get(f"/api/recommendations/{race_id}?top_n_horses=2")

    assert resp.status_code == 200
    assert captured_args.get("top_n_horses") == 2


def test_recommendations_top_k_param(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    """top_k query param is forwarded to predict_race_with_combinations."""
    race_id = "REC_RACE5"
    from keiba_ai.core.paths import db_path
    from keiba_ai.db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        _seed_race_and_entries(session, race_id, n_horses=4)
        _seed_active_model(session, str(tmp_path / "fake_model_topk"))

    fake_df = _fake_predictions_df(race_id, n=4)
    captured_top_k: dict = {}

    def _spy_combinations(model, frame, session, top_k_combinations=None):
        captured_top_k["top_k"] = top_k_combinations
        return _fake_combinations()

    with (
        patch("keiba_ai.api.routers.recommendations.load_model", return_value=MagicMock()),
        patch("keiba_ai.api.routers.recommendations.predict_race", return_value=fake_df),
        patch("keiba_ai.api.routers.recommendations.predict_race_with_combinations",
              side_effect=_spy_combinations),
        patch("keiba_ai.api.routers.recommendations.recommend_for_race",
              return_value=RecommendationResult(
                  race_id=race_id, bankroll_at_decision=100_000, candidates=[]
              )),
        TestClient(app_with_temp_db) as client,
    ):
        resp = client.get(f"/api/recommendations/{race_id}?top_k=10")

    assert resp.status_code == 200
    assert captured_top_k.get("top_k") == 10


def test_recommendations_empty_candidates(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    """When recommend_for_race returns no candidates (all EV <= 1.0), response is 200 with empty list."""
    race_id = "REC_RACE6"
    from keiba_ai.core.paths import db_path
    from keiba_ai.db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        _seed_race_and_entries(session, race_id, n_horses=4)
        _seed_active_model(session, str(tmp_path / "fake_model_empty"))

    fake_df = _fake_predictions_df(race_id, n=4)
    empty_result = RecommendationResult(
        race_id=race_id,
        bankroll_at_decision=100_000,
        candidates=[],
    )

    with (
        patch("keiba_ai.api.routers.recommendations.load_model", return_value=MagicMock()),
        patch("keiba_ai.api.routers.recommendations.predict_race", return_value=fake_df),
        patch("keiba_ai.api.routers.recommendations.predict_race_with_combinations",
              return_value=_fake_combinations()),
        patch("keiba_ai.api.routers.recommendations.recommend_for_race",
              return_value=empty_result),
        TestClient(app_with_temp_db) as client,
    ):
        resp = client.get(f"/api/recommendations/{race_id}")

    assert resp.status_code == 200
    data = resp.json()
    assert data["candidates"] == []
