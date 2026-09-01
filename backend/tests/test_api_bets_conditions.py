"""購入記録に実行条件を残すこと。

モデルを差し替えたり設定を変えたりすると、**過去の記録がどの条件で出たものか
分からなくなり実績を評価できない**。シミュレーションには 0013 で同じ仕組みを
入れており、実運用の記録にも要る。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.deps import get_settings_store
from core.settings_store import SettingsStore


@pytest.fixture
def client_with_settings(app_with_temp_db: FastAPI, tmp_path: Path):
    store = SettingsStore(tmp_path / "settings.json")
    store.save({
        "probability_model_path": "models/20260827T140017-nn",
        "place_min_hit_prob": 0.6,
        "win_min_odds": 1.1,
        "stake_units": {"単勝": 500},
    })
    app_with_temp_db.dependency_overrides[get_settings_store] = lambda: store
    try:
        with TestClient(app_with_temp_db) as client:
            yield client
    finally:
        app_with_temp_db.dependency_overrides.pop(get_settings_store, None)


def _seed_race(race_id: str = "COND_RACE") -> str:
    from core.paths import db_path
    from db.models.race import Race
    from db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        session.add(Race(race_id=race_id, date="2026-05-31", course="東京",
                         surface="芝", distance=1600, n_runners=8))
    return race_id


def test_single_bet_records_the_conditions(client_with_settings: TestClient):
    race_id = _seed_race("COND_ONE")
    resp = client_with_settings.post("/api/bets", json={
        "race_id": race_id, "bet_type": "単勝", "combo": "5",
        "stake": 500, "source": "recommendation",
    })
    assert resp.status_code == 201
    cond = resp.json()["conditions"]
    assert cond is not None
    assert cond["probability_model"] == "20260827T140017-nn"
    assert cond["place_min_hit_prob"] == 0.6
    assert cond["win_min_odds"] == 1.1


def test_bulk_bets_share_one_condition_snapshot(client_with_settings: TestClient):
    """同じ推奨から出た点は条件が同一なので 1 回だけ写す。"""
    race_id = _seed_race("COND_BULK")
    resp = client_with_settings.post("/api/bets/bulk", json={
        "race_id": race_id, "bet_type": "馬連", "source": "recommendation",
        "combos": [{"combo": "1-2", "stake": 100}, {"combo": "1-3", "stake": 100}],
    })
    assert resp.status_code == 201
    rows = resp.json()["items"]
    assert len(rows) == 2
    assert rows[0]["conditions"] == rows[1]["conditions"]
    assert rows[0]["conditions"]["probability_model"] == "20260827T140017-nn"


def test_probability_model_unset_records_none(app_with_temp_db: FastAPI, tmp_path: Path):
    """確率モデル未設定なら、しきい値も None にする。

    しきい値だけ残すと「0.30 で絞ったのか、絞っていないのか」が判別できない。
    """
    store = SettingsStore(tmp_path / "settings.json")
    store.save({"probability_model_path": None, "place_min_hit_prob": 0.6})
    app_with_temp_db.dependency_overrides[get_settings_store] = lambda: store
    try:
        with TestClient(app_with_temp_db) as client:
            race_id = _seed_race("COND_NONE")
            resp = client.post("/api/bets", json={
                "race_id": race_id, "bet_type": "単勝", "combo": "1",
                "stake": 500, "source": "manual",
            })
    finally:
        app_with_temp_db.dependency_overrides.pop(get_settings_store, None)

    cond = resp.json()["conditions"]
    assert cond["probability_model"] is None
    assert cond["place_min_hit_prob"] is None
