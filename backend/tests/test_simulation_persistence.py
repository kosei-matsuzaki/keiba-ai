"""Tests for ai/simulation_persistence.py and saved-run API endpoints."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import keiba_ai.db.models  # noqa: F401
from keiba_ai.ai.simulation import BankrollPoint, GroupStats, SimulationResult
from keiba_ai.ai.simulation_persistence import (
    MAX_SAVED_RUNS,
    delete_simulation_run,
    get_simulation_run,
    list_simulation_runs,
    save_simulation_result,
)
from keiba_ai.db.base import Base
from keiba_ai.db.models.simulation_run import SimulationRun


@pytest.fixture()
def engine():
    e = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(e)
    yield e
    e.dispose()


def _make_result(budget: int = 100_000, final: int = 105_000) -> SimulationResult:
    """テスト用 SimulationResult をでっち上げる。"""
    return SimulationResult(
        window_start="2024-01-01",
        window_end="2024-03-31",
        model_path="/tmp/dummy",
        strategy="balanced",
        budget=budget,
        n_races=10,
        n_settled_races=10,
        final_bankroll=final,
        peak_bankroll=max(budget, final),
        summary=GroupStats(label="all", n_bets=20, invested=2000, payout=2200, hits=5),
        by_bet_type=[GroupStats(label="単勝", n_bets=10, invested=1000, payout=1100, hits=3)],
        by_race_class=[GroupStats(label="G1", n_bets=5, invested=500, payout=600, hits=2)],
        by_course=[GroupStats(label="東京", n_bets=15, invested=1500, payout=1700, hits=4)],
        bankroll_timeseries=[
            BankrollPoint(date="2024-01-01", bankroll=100_500, invested=100, payout=600, n_bets=1),
            BankrollPoint(date="2024-01-02", bankroll=101_000, invested=100, payout=600, n_bets=1),
        ],
    )


# ---------------------------------------------------------------------------
# save_simulation_result
# ---------------------------------------------------------------------------


def test_save_simulation_result_creates_row(engine):
    with Session(engine) as session:
        result = _make_result(budget=100_000, final=105_000)
        saved = save_simulation_result(session, result)
        # Session 内で属性をすべて読む (commit 後の lazy refresh を避ける)
        assert saved.id is not None
        assert saved.budget == 100_000
        assert saved.final_bankroll == 105_000
        assert saved.strategy == "balanced"
        parsed_summary = json.loads(saved.summary_json)
        assert parsed_summary["n_bets"] == 20
        parsed_ts = json.loads(saved.bankroll_timeseries_json)
        assert len(parsed_ts) == 2


def test_save_simulation_result_prunes_when_over_limit(engine):
    """50 件を超えると古い順に削除される。"""
    with Session(engine) as session:
        # MAX_SAVED_RUNS + 5 件を保存
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i in range(MAX_SAVED_RUNS + 5):
            result = _make_result(budget=100_000 + i, final=100_000)
            saved = save_simulation_result(session, result)
            # created_at を古い順に手動で書き換え (insert 順 = 古い順 にしたい)
            saved.created_at = (base + timedelta(minutes=i)).isoformat()
            session.commit()

        # 最終的に 50 件だけ残る
        runs = list_simulation_runs(session, limit=200)
        assert len(runs) == MAX_SAVED_RUNS
        # budget の小さい (= 古い) 5 件は消えていて、新しい 50 件が残る
        budgets = sorted(r.budget for r in runs)
        # 最初の 5 件 (budget 100000..100004) は削除されているはず
        assert budgets[0] >= 100_000 + 5


def test_list_simulation_runs_orders_newest_first(engine):
    with Session(engine) as session:
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i in range(3):
            saved = save_simulation_result(session, _make_result(budget=100 * (i + 1)))
            saved.created_at = (base + timedelta(days=i)).isoformat()
            session.commit()

        runs = list_simulation_runs(session)

    assert len(runs) == 3
    # 新しい順
    assert runs[0].budget == 300
    assert runs[1].budget == 200
    assert runs[2].budget == 100


def test_get_simulation_run_returns_row_or_none(engine):
    with Session(engine) as session:
        saved = save_simulation_result(session, _make_result())
        got = get_simulation_run(session, saved.id)
        assert got is not None
        assert got.id == saved.id

        missing = get_simulation_run(session, 99999)
        assert missing is None


def test_delete_simulation_run_removes_row(engine):
    with Session(engine) as session:
        saved = save_simulation_result(session, _make_result())
        assert delete_simulation_run(session, saved.id) is True
        assert get_simulation_run(session, saved.id) is None
        # 既に消えているので False
        assert delete_simulation_run(session, saved.id) is False


# ---------------------------------------------------------------------------
# API endpoints (list / detail / delete)
# ---------------------------------------------------------------------------


def _seed_run_via_db(api_client: TestClient) -> int:
    """api_client の app の engine を直接使って 1 件 seed、id を返す。"""
    app_engine = api_client.app.state.engine
    with Session(app_engine) as session:
        saved = save_simulation_result(session, _make_result())
        return saved.id


def test_api_list_runs_empty(api_client: TestClient):
    resp = api_client.get("/api/simulation/runs")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"runs": [], "total": 0}


def test_api_get_run_404(api_client: TestClient):
    resp = api_client.get("/api/simulation/runs/99999")
    assert resp.status_code == 404


def test_api_delete_run_404(api_client: TestClient):
    resp = api_client.delete("/api/simulation/runs/99999")
    assert resp.status_code == 404


def test_api_list_then_get_then_delete(api_client: TestClient):
    """直接 DB に 1 件 seed → list / get / delete が連動する。"""
    run_id = _seed_run_via_db(api_client)

    list_resp = api_client.get("/api/simulation/runs")
    assert list_resp.status_code == 200
    assert list_resp.json()["total"] == 1
    assert list_resp.json()["runs"][0]["id"] == run_id

    get_resp = api_client.get(f"/api/simulation/runs/{run_id}")
    assert get_resp.status_code == 200
    body = get_resp.json()
    assert body["run_id"] == run_id
    assert body["budget"] == 100_000
    assert body["final_bankroll"] == 105_000
    # bankroll_timeseries が JSON decode されて含まれる
    assert len(body["bankroll_timeseries"]) == 2

    del_resp = api_client.delete(f"/api/simulation/runs/{run_id}")
    assert del_resp.status_code == 200
    assert del_resp.json() == {"deleted": run_id}

    # 削除後は 404
    assert api_client.get(f"/api/simulation/runs/{run_id}").status_code == 404
