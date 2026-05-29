"""Tests for ai/simulation_persistence.py and saved-run API endpoints."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

import db.models  # noqa: F401
from ai.simulation import BankrollPoint, GroupStats, SimulationResult
from ai.simulation_persistence import (
    MAX_SAVED_RUNS,
    delete_simulation_run,
    get_simulation_run,
    list_simulation_runs,
    save_simulation_result,
)
from db.base import Base
from db.models.model_run import ModelRun
from db.models.simulation_run import SimulationRun


@pytest.fixture()
def engine():
    e = create_engine("sqlite:///:memory:", future=True)

    # simulation_runs.model_run_id の FK CASCADE / NOT NULL を効かせるため
    # 接続ごとに foreign_keys=ON にする (本番 db/session.py と同じ)。
    @event.listens_for(e, "connect")
    def _set_fk(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(e)
    yield e
    e.dispose()


def _seed_model(session: Session, model_path: str = "/tmp/dummy-model") -> int:
    """テスト用 ModelRun を 1 件作成し id を返す (simulation_runs の FK 親)。"""
    run = ModelRun(
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc).isoformat(),
        model_path=model_path,
    )
    session.add(run)
    session.flush()
    return run.id


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
        model_id = _seed_model(session)
        result = _make_result(budget=100_000, final=105_000)
        saved = save_simulation_result(session, result, model_id)
        # Session 内で属性をすべて読む (commit 後の lazy refresh を避ける)
        assert saved.id is not None
        assert saved.model_run_id == model_id
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
        model_id = _seed_model(session)
        # MAX_SAVED_RUNS + 5 件を保存
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i in range(MAX_SAVED_RUNS + 5):
            result = _make_result(budget=100_000 + i, final=100_000)
            saved = save_simulation_result(session, result, model_id)
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
        model_id = _seed_model(session)
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        for i in range(3):
            saved = save_simulation_result(
                session, _make_result(budget=100 * (i + 1)), model_id
            )
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
        model_id = _seed_model(session)
        saved = save_simulation_result(session, _make_result(), model_id)
        got = get_simulation_run(session, saved.id)
        assert got is not None
        assert got.id == saved.id

        missing = get_simulation_run(session, 99999)
        assert missing is None


def test_delete_simulation_run_removes_row(engine):
    with Session(engine) as session:
        model_id = _seed_model(session)
        saved = save_simulation_result(session, _make_result(), model_id)
        assert delete_simulation_run(session, saved.id) is True
        assert get_simulation_run(session, saved.id) is None
        # 既に消えているので False
        assert delete_simulation_run(session, saved.id) is False


# ---------------------------------------------------------------------------
# model_run_id FK: フィルタ / CASCADE 削除 / renumber 追従
# ---------------------------------------------------------------------------


def test_list_simulation_runs_filters_by_model(engine):
    """model_run_id を渡すとそのモデルの run のみ返る。"""
    with Session(engine) as session:
        model_a = _seed_model(session, model_path="/tmp/model-a")
        model_b = _seed_model(session, model_path="/tmp/model-b")
        save_simulation_result(session, _make_result(budget=100), model_a)
        save_simulation_result(session, _make_result(budget=200), model_a)
        save_simulation_result(session, _make_result(budget=300), model_b)
        session.commit()

        a_runs = list_simulation_runs(session, model_run_id=model_a)
        b_runs = list_simulation_runs(session, model_run_id=model_b)
        all_runs = list_simulation_runs(session)

    assert len(a_runs) == 2
    assert all(r.model_run_id == model_a for r in a_runs)
    assert len(b_runs) == 1
    assert b_runs[0].model_run_id == model_b
    assert len(all_runs) == 3


def test_delete_model_cascades_to_simulation_runs(engine):
    """ModelRun 削除で紐づく simulation_runs も消える (ON DELETE CASCADE)。"""
    with Session(engine) as session:
        model_a = _seed_model(session, model_path="/tmp/model-a")
        model_b = _seed_model(session, model_path="/tmp/model-b")
        sim_a = save_simulation_result(session, _make_result(), model_a)
        sim_b = save_simulation_result(session, _make_result(), model_b)
        sim_a_id, sim_b_id = sim_a.id, sim_b.id
        session.commit()

        # model_a を削除 → sim_a だけ消え、sim_b は残る
        session.delete(session.get(ModelRun, model_a))
        session.commit()

        assert get_simulation_run(session, sim_a_id) is None
        assert get_simulation_run(session, sim_b_id) is not None


def test_renumber_model_ids_cascades_to_simulation_runs(engine):
    """renumber_model_ids で model_runs.id が振り直されても FK が追従する
    (ON UPDATE CASCADE)。"""
    from ai.registry import renumber_model_ids

    with Session(engine) as session:
        # id を飛び番にするため 3 件作って真ん中を消す
        _seed_model(session, model_path="/tmp/m1")
        m2 = _seed_model(session, model_path="/tmp/m2")
        m3 = _seed_model(session, model_path="/tmp/m3")
        session.delete(session.get(ModelRun, m2))
        session.commit()

        # 残った m3 (= 最新) に sim を紐づける
        sim = save_simulation_result(session, _make_result(), m3)
        sim_id = sim.id
        session.commit()

        renumber_model_ids(session)
        session.commit()

        # m3 は created_at 順で 2 番目 → id=2 に詰められ、sim も追従する
        runs = list_simulation_runs(session)
        assert len(runs) == 1
        new_model_id = runs[0].model_run_id
        # FK 整合: 振り直された ModelRun が実在する
        assert session.get(ModelRun, new_model_id) is not None
        # m1 が id=1, m3 が id=2
        assert new_model_id == 2
        assert get_simulation_run(session, sim_id) is not None


# ---------------------------------------------------------------------------
# API endpoints (list / detail / delete)
# ---------------------------------------------------------------------------


def _seed_run_via_db(api_client: TestClient) -> int:
    """api_client の app の engine を直接使って 1 件 seed、id を返す。"""
    app_engine = api_client.app.state.engine
    with Session(app_engine) as session:
        model_id = _seed_model(session)
        saved = save_simulation_result(session, _make_result(), model_id)
        session.commit()
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
