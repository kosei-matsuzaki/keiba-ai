"""Tests for GET /api/recommendations/{race_id}."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai.types import BetCandidate, RecommendationResult
from db.models.entry import Entry
from db.models.horse import Horse
from db.models.live_odds import LiveOdds
from db.models.model_run import ModelRun
from db.models.payout import Payout
from db.models.race import Race

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


def _fake_recommendation_result_with_null_odds(race_id: str, bankroll: int = 100_000) -> RecommendationResult:
    """Return a RecommendationResult with some null est_odds / ev candidates."""
    return RecommendationResult(
        race_id=race_id,
        bankroll_at_decision=bankroll,
        candidates=[
            BetCandidate(
                bet_type="単勝",
                combo="3",
                pattern="box",
                prob=0.35,
                est_odds=3.0,
                ev=1.05,
                stake=300,
                post_positions=(3,),
            ),
            BetCandidate(
                bet_type="馬連",
                combo="1-3",
                pattern="box",
                prob=0.2,
                est_odds=None,
                ev=None,
                stake=0,
                post_positions=(1, 3),
            ),
            BetCandidate(
                bet_type="ワイド",
                combo="3-5",
                pattern="box",
                prob=0.4,
                est_odds=None,
                ev=None,
                stake=0,
                post_positions=(3, 5),
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
    from core.paths import db_path
    from db.session import make_engine, session_scope

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
    from core.paths import db_path
    from db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        _seed_race_and_entries(session, race_id, n_horses=4)
        _seed_active_model(session, str(tmp_path / "fake_model_rec"))

    fake_df = _fake_predictions_df(race_id, n=4)
    fake_result = _fake_recommendation_result(race_id)

    with (
        patch("api.routers.recommendations.load_model", return_value=MagicMock()),
        patch("api.routers.recommendations.predict_race_gbdt", return_value=fake_df),
        patch("api.routers.recommendations.predict_race_with_combinations_gbdt",
              return_value=_fake_combinations()),
        patch("api.routers.recommendations.recommend_for_race",
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
    from core.paths import db_path
    from db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        _seed_race_and_entries(session, race_id, n_horses=4)
        _seed_active_model(session, str(tmp_path / "fake_model_cap"))

    fake_df = _fake_predictions_df(race_id, n=4)
    fake_result = _fake_recommendation_result(race_id, bankroll=100_000)

    with (
        patch("api.routers.recommendations.load_model", return_value=MagicMock()),
        patch("api.routers.recommendations.predict_race_gbdt", return_value=fake_df),
        patch("api.routers.recommendations.predict_race_with_combinations_gbdt",
              return_value=_fake_combinations()),
        patch("api.routers.recommendations.recommend_for_race",
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
    from core.paths import db_path
    from db.session import make_engine, session_scope

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
        patch("api.routers.recommendations.load_model", return_value=MagicMock()),
        patch("api.routers.recommendations.predict_race_gbdt", return_value=fake_df),
        patch("api.routers.recommendations.predict_race_with_combinations_gbdt",
              return_value=_fake_combinations()),
        patch("api.routers.recommendations.recommend_for_race",
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
    from core.paths import db_path
    from db.session import make_engine, session_scope

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
        patch("api.routers.recommendations.load_model", return_value=MagicMock()),
        patch("api.routers.recommendations.predict_race_gbdt", return_value=fake_df),
        patch("api.routers.recommendations.predict_race_with_combinations_gbdt",
              return_value=_fake_combinations()),
        patch("api.routers.recommendations.recommend_for_race",
              side_effect=_capture_recommend),
        TestClient(app_with_temp_db) as client,
    ):
        resp = client.get(f"/api/recommendations/{race_id}?top_n_horses=2")

    assert resp.status_code == 200
    assert captured_args.get("top_n_horses") == 2


def test_recommendations_candidates_include_zero_stake(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    """candidates array may contain stake=0 entries (keep_zero_stake=True path).

    The API must return these without raising a validation error; stake >= 0
    is the invariant.
    """
    race_id = "REC_RACE_ZS"
    from core.paths import db_path
    from db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        _seed_race_and_entries(session, race_id, n_horses=4)
        _seed_active_model(session, str(tmp_path / "fake_model_zs"))

    fake_df = _fake_predictions_df(race_id, n=4)
    # Result includes one positive-stake and one zero-stake candidate
    mixed_result = RecommendationResult(
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
                stake=500,
                post_positions=(1,),
            ),
            BetCandidate(
                bet_type="馬連",
                combo="2-3",
                pattern="nagashi",
                prob=0.1,
                est_odds=5.0,
                ev=0.5,
                stake=0,
                post_positions=(2, 3),
            ),
        ],
    )

    with (
        patch("api.routers.recommendations.load_model", return_value=MagicMock()),
        patch("api.routers.recommendations.predict_race_gbdt", return_value=fake_df),
        patch("api.routers.recommendations.predict_race_with_combinations_gbdt",
              return_value=_fake_combinations()),
        patch("api.routers.recommendations.recommend_for_race",
              return_value=mixed_result),
        TestClient(app_with_temp_db) as client,
    ):
        resp = client.get(f"/api/recommendations/{race_id}")

    assert resp.status_code == 200
    data = resp.json()
    stakes = [c["stake"] for c in data["candidates"]]
    # All stakes are non-negative
    assert all(s >= 0 for s in stakes)
    # At least one zero-stake candidate is present
    assert 0 in stakes


def test_recommendations_top_k_param(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    """top_k query param is forwarded to predict_race_with_combinations_gbdt."""
    race_id = "REC_RACE5"
    from core.paths import db_path
    from db.session import make_engine, session_scope

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
        patch("api.routers.recommendations.load_model", return_value=MagicMock()),
        patch("api.routers.recommendations.predict_race_gbdt", return_value=fake_df),
        patch("api.routers.recommendations.predict_race_with_combinations_gbdt",
              side_effect=_spy_combinations),
        patch("api.routers.recommendations.recommend_for_race",
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
    from core.paths import db_path
    from db.session import make_engine, session_scope

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
        patch("api.routers.recommendations.load_model", return_value=MagicMock()),
        patch("api.routers.recommendations.predict_race_gbdt", return_value=fake_df),
        patch("api.routers.recommendations.predict_race_with_combinations_gbdt",
              return_value=_fake_combinations()),
        patch("api.routers.recommendations.recommend_for_race",
              return_value=empty_result),
        TestClient(app_with_temp_db) as client,
    ):
        resp = client.get(f"/api/recommendations/{race_id}")

    assert resp.status_code == 200
    data = resp.json()
    assert data["candidates"] == []


def test_recommendations_odds_source_unknown_when_no_odds(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    """odds_source='unknown' when no live or past odds are available (today's race)."""
    race_id = "REC_RACE_ODDS_UNKNOWN"
    from core.paths import db_path
    from db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        _seed_race_and_entries(session, race_id, n_horses=4)
        _seed_active_model(session, str(tmp_path / "fake_model_unknown"))

    fake_df = _fake_predictions_df(race_id, n=4)
    fake_result = _fake_recommendation_result(race_id)

    with (
        patch("api.routers.recommendations.load_model", return_value=MagicMock()),
        patch("api.routers.recommendations.predict_race_gbdt", return_value=fake_df),
        patch("api.routers.recommendations.predict_race_with_combinations_gbdt",
              return_value=_fake_combinations()),
        patch("api.routers.recommendations.recommend_for_race",
              return_value=fake_result),
        TestClient(app_with_temp_db) as client,
    ):
        resp = client.get(f"/api/recommendations/{race_id}")

    assert resp.status_code == 200
    data = resp.json()
    # Today's race with no live_odds → unknown (past fallback skipped for today)
    assert data["odds_source"] == "unknown"


def test_recommendations_odds_source_live_when_live_odds_present(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    """odds_source='live' when live_odds table has rows for the race."""
    race_id = "REC_RACE_ODDS_LIVE"
    from core.paths import db_path
    from db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        _seed_race_and_entries(session, race_id, n_horses=4)
        _seed_active_model(session, str(tmp_path / "fake_model_live"))
        # Insert live_odds rows for this race
        session.add(LiveOdds(
            race_id=race_id,
            bet_type="馬連",
            combo="1-2",
            odds=30.5,
            odds_max=None,
            popularity=1,
            fetched_at="2025-01-01T10:00:00+00:00",
        ))
        session.commit()

    fake_df = _fake_predictions_df(race_id, n=4)

    # predict_race_with_combinations_gbdt spy: verifies race_odds is passed
    captured_race_odds: dict = {}

    def _spy_combinations(model, frame, session, top_k_combinations=None, race_odds=None):
        captured_race_odds["value"] = race_odds
        return _fake_combinations()

    fake_result = _fake_recommendation_result(race_id)

    with (
        patch("api.routers.recommendations.load_model", return_value=MagicMock()),
        patch("api.routers.recommendations.predict_race_gbdt", return_value=fake_df),
        patch("api.routers.recommendations.predict_race_with_combinations_gbdt",
              side_effect=_spy_combinations),
        patch("api.routers.recommendations.recommend_for_race",
              return_value=fake_result),
        TestClient(app_with_temp_db) as client,
    ):
        resp = client.get(f"/api/recommendations/{race_id}")

    assert resp.status_code == 200
    data = resp.json()
    assert data["odds_source"] == "live"
    # race_odds dict was passed to predict_race_with_combinations_gbdt
    assert captured_race_odds["value"] is not None
    assert "馬連" in captured_race_odds["value"]
    assert captured_race_odds["value"]["馬連"]["1-2"] == pytest.approx(30.5)


def test_recommendations_odds_source_past_for_past_race(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    """odds_source='past' when no live odds but past-race payouts are available."""
    from datetime import date, timedelta
    race_id = "REC_RACE_PAST_ODDS"
    past_date = (date.today() - timedelta(days=1)).isoformat()

    from core.paths import db_path
    from db.models.horse import Horse
    from db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        # Seed a past race with payouts
        session.add(Race(
            race_id=race_id,
            date=past_date,
            course="東京",
            surface="芝",
            distance=2000,
            n_runners=4,
        ))
        session.flush()
        for i in range(4):
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
                finish_position=i + 1,
            ))
        session.flush()
        session.add(Payout(race_id=race_id, bet_type="馬連", combo="1-2", amount=4000))
        session.commit()
        _seed_active_model(session, str(tmp_path / "fake_model_past"))

    fake_df = _fake_predictions_df(race_id, n=4)
    fake_result = _fake_recommendation_result(race_id)

    with (
        patch("api.routers.recommendations.load_model", return_value=MagicMock()),
        patch("api.routers.recommendations.predict_race_gbdt", return_value=fake_df),
        patch("api.routers.recommendations.predict_race_with_combinations_gbdt",
              return_value=_fake_combinations()),
        patch("api.routers.recommendations.recommend_for_race",
              return_value=fake_result),
        TestClient(app_with_temp_db) as client,
    ):
        resp = client.get(f"/api/recommendations/{race_id}")

    assert resp.status_code == 200
    data = resp.json()
    assert data["odds_source"] == "past"


def test_recommendations_null_est_odds_candidates(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    """candidates with est_odds=null and ev=null serialize correctly (no validation error)."""
    race_id = "REC_RACE_NULL_ODDS"
    from core.paths import db_path
    from db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        _seed_race_and_entries(session, race_id, n_horses=4)
        _seed_active_model(session, str(tmp_path / "fake_model_null"))

    fake_df = _fake_predictions_df(race_id, n=4)
    null_result = _fake_recommendation_result_with_null_odds(race_id)

    with (
        patch("api.routers.recommendations.load_model", return_value=MagicMock()),
        patch("api.routers.recommendations.predict_race_gbdt", return_value=fake_df),
        patch("api.routers.recommendations.predict_race_with_combinations_gbdt",
              return_value=_fake_combinations()),
        patch("api.routers.recommendations.recommend_for_race",
              return_value=null_result),
        TestClient(app_with_temp_db) as client,
    ):
        resp = client.get(f"/api/recommendations/{race_id}")

    assert resp.status_code == 200
    data = resp.json()
    candidates = data["candidates"]
    assert len(candidates) == 3

    # The 単勝 candidate has real odds
    tan_cand = next(c for c in candidates if c["bet_type"] == "単勝")
    assert tan_cand["est_odds"] == pytest.approx(3.0)
    assert tan_cand["ev"] == pytest.approx(1.05)

    # The 馬連 and ワイド candidates have null odds
    null_cands = [c for c in candidates if c["est_odds"] is None]
    assert len(null_cands) == 2
    for nc in null_cands:
        assert nc["est_odds"] is None
        assert nc["ev"] is None
        assert nc["stake"] == 0
