"""Tests for /api/races endpoints."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from keiba_ai.db.models.entry import Entry
from keiba_ai.db.models.horse import Horse
from keiba_ai.db.models.race import Race


def _insert_race(session: Session, race_id: str, race_date: str, n_horses: int = 3) -> None:
    """Helper: insert one race + n_horses entries into the DB."""
    session.add(Race(
        race_id=race_id,
        date=race_date,
        course="東京",
        surface="芝",
        distance=2000,
        race_class="G1",
        n_runners=n_horses,
        payout_win=None,
        payout_place=None,
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
            finish_position=i + 1,
        ))
    session.commit()


def test_upcoming_races_empty(api_client: TestClient) -> None:
    """No races in DB — empty list returned."""
    resp = api_client.get("/api/races/upcoming")
    assert resp.status_code == 200
    assert resp.json()["races"] == []


def test_upcoming_races_filters_past(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    """Past races are excluded; only future races appear."""
    from keiba_ai.core.paths import db_path
    from keiba_ai.db.session import make_engine, session_scope

    engine = make_engine(db_path())
    today = date.today()
    past = (today - timedelta(days=5)).isoformat()
    future = (today + timedelta(days=2)).isoformat()

    with session_scope(engine) as session:
        _insert_race(session, "PAST001", past)
        _insert_race(session, "FUTURE001", future)

    with TestClient(app_with_temp_db) as client:
        resp = client.get("/api/races/upcoming?days=7")
    assert resp.status_code == 200
    ids = [r["race_id"] for r in resp.json()["races"]]
    assert "FUTURE001" in ids
    assert "PAST001" not in ids


def test_race_detail_not_found(api_client: TestClient) -> None:
    resp = api_client.get("/api/races/NONEXISTENT")
    assert resp.status_code == 404


def test_race_detail_found(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    from keiba_ai.core.paths import db_path
    from keiba_ai.db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        _insert_race(session, "RACE001", date.today().isoformat(), n_horses=5)

    with TestClient(app_with_temp_db) as client:
        resp = client.get("/api/races/RACE001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["race_id"] == "RACE001"
    assert len(data["entries"]) == 5
