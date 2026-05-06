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


# ── /races/recent ─────────────────────────────────────────────────────────────

def test_recent_races_empty(api_client: TestClient) -> None:
    """No races in DB — empty list returned."""
    resp = api_client.get("/api/races/recent")
    assert resp.status_code == 200
    assert resp.json()["races"] == []


def test_recent_races_within_window(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    """Only races within the days window (exclusive of today) are returned."""
    from keiba_ai.core.paths import db_path
    from keiba_ai.db.session import make_engine, session_scope

    engine = make_engine(db_path())
    today = date.today()
    recent = (today - timedelta(days=10)).isoformat()
    old = (today - timedelta(days=60)).isoformat()
    future = (today + timedelta(days=1)).isoformat()

    with session_scope(engine) as session:
        _insert_race(session, "RECENT001", recent)
        _insert_race(session, "OLD001", old)
        _insert_race(session, "FUTURE001", future)
        # Today itself should be excluded (date < today)
        _insert_race(session, "TODAY001", today.isoformat())

    with TestClient(app_with_temp_db) as client:
        resp = client.get("/api/races/recent?days=30")

    assert resp.status_code == 200
    ids = [r["race_id"] for r in resp.json()["races"]]
    assert "RECENT001" in ids
    assert "OLD001" not in ids
    assert "FUTURE001" not in ids
    assert "TODAY001" not in ids


def test_recent_races_descending_order(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    """Results are sorted by date descending (most recent first)."""
    from keiba_ai.core.paths import db_path
    from keiba_ai.db.session import make_engine, session_scope

    engine = make_engine(db_path())
    today = date.today()
    d1 = (today - timedelta(days=5)).isoformat()
    d2 = (today - timedelta(days=15)).isoformat()
    d3 = (today - timedelta(days=25)).isoformat()

    with session_scope(engine) as session:
        # Insert out of order to confirm sorting is applied
        _insert_race(session, "R_D3", d3)
        _insert_race(session, "R_D1", d1)
        _insert_race(session, "R_D2", d2)

    with TestClient(app_with_temp_db) as client:
        resp = client.get("/api/races/recent?days=30")

    assert resp.status_code == 200
    dates = [r["date"] for r in resp.json()["races"]]
    assert dates == sorted(dates, reverse=True)


def test_recent_races_limit(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    """limit parameter caps the number of returned races."""
    from keiba_ai.core.paths import db_path
    from keiba_ai.db.session import make_engine, session_scope

    engine = make_engine(db_path())
    today = date.today()

    with session_scope(engine) as session:
        for i in range(5):
            d = (today - timedelta(days=i + 1)).isoformat()
            _insert_race(session, f"RLIMIT{i:03d}", d)

    with TestClient(app_with_temp_db) as client:
        resp = client.get("/api/races/recent?days=30&limit=3")

    assert resp.status_code == 200
    assert len(resp.json()["races"]) == 3


def test_recent_races_explicit_date_range(
    app_with_temp_db: FastAPI,
    tmp_path: Path,
) -> None:
    """from/to date range overrides days mode; bounds are inclusive."""
    from keiba_ai.core.paths import db_path
    from keiba_ai.db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        _insert_race(session, "RIN001", "2024-12-01")
        _insert_race(session, "RIN002", "2024-12-15")
        _insert_race(session, "RIN003", "2024-12-31")
        _insert_race(session, "ROUT001", "2024-11-30")  # before from
        _insert_race(session, "ROUT002", "2025-01-01")  # after to

    with TestClient(app_with_temp_db) as client:
        resp = client.get("/api/races/recent?from=2024-12-01&to=2024-12-31")

    assert resp.status_code == 200
    race_ids = {r["race_id"] for r in resp.json()["races"]}
    assert race_ids == {"RIN001", "RIN002", "RIN003"}


def test_recent_races_invalid_date_range_returns_422(
    app_with_temp_db: FastAPI,
) -> None:
    """from > to is rejected with 422."""
    with TestClient(app_with_temp_db) as client:
        resp = client.get("/api/races/recent?from=2024-12-31&to=2024-12-01")

    assert resp.status_code == 422


def test_recent_races_invalid_date_format_returns_422(
    app_with_temp_db: FastAPI,
) -> None:
    """Malformed date string is rejected with 422."""
    with TestClient(app_with_temp_db) as client:
        resp = client.get("/api/races/recent?from=2024/12/01&to=2024-12-31")

    assert resp.status_code == 422


def test_recent_races_date_range_too_long_returns_422(
    app_with_temp_db: FastAPI,
) -> None:
    """Date ranges over 365 days are rejected with 422."""
    with TestClient(app_with_temp_db) as client:
        resp = client.get("/api/races/recent?from=2023-01-01&to=2024-12-31")

    assert resp.status_code == 422


# ── /races/by_date ────────────────────────────────────────────────────────────

def test_by_date_empty(api_client: TestClient) -> None:
    """No races in DB for the given date — empty list returned (not 404)."""
    resp = api_client.get("/api/races/by_date?date=2024-12-01")
    assert resp.status_code == 200
    assert resp.json()["races"] == []


def test_by_date_returns_matching_races(
    app_with_temp_db: FastAPI,
) -> None:
    """Races on the target date are returned; other dates are excluded."""
    from keiba_ai.core.paths import db_path
    from keiba_ai.db.session import make_engine, session_scope

    engine = make_engine(db_path())
    target = "2024-06-01"

    with session_scope(engine) as session:
        _insert_race(session, "MATCH001", target)
        _insert_race(session, "MATCH002", target)
        _insert_race(session, "OTHER001", "2024-06-02")

    with TestClient(app_with_temp_db) as client:
        resp = client.get(f"/api/races/by_date?date={target}")

    assert resp.status_code == 200
    race_ids = {r["race_id"] for r in resp.json()["races"]}
    assert race_ids == {"MATCH001", "MATCH002"}
    assert "OTHER001" not in race_ids


# ── /races/this_weekend ───────────────────────────────────────────────────────


def test_this_weekend_empty(api_client: TestClient) -> None:
    """DB にレースがない場合は空リストで 200 を返す。"""
    resp = api_client.get("/api/races/this_weekend")
    assert resp.status_code == 200
    assert resp.json()["races"] == []


def test_this_weekend_returns_only_weekend_races(
    app_with_temp_db: FastAPI,
) -> None:
    """土・日のレースのみ返り、他日付のレースは除外される。"""
    from datetime import timedelta

    from keiba_ai.core.dates import this_weekend_dates
    from keiba_ai.core.paths import db_path
    from keiba_ai.db.session import make_engine, session_scope

    engine = make_engine(db_path())
    sat, sun = this_weekend_dates()
    other_day = (sat - timedelta(days=1)).isoformat()  # 先週日 = 除外対象

    with session_scope(engine) as session:
        _insert_race(session, "SAT001", sat.isoformat())
        _insert_race(session, "SUN001", sun.isoformat())
        _insert_race(session, "OTHER001", other_day)

    with TestClient(app_with_temp_db) as client:
        resp = client.get("/api/races/this_weekend")

    assert resp.status_code == 200
    ids = {r["race_id"] for r in resp.json()["races"]}
    assert "SAT001" in ids
    assert "SUN001" in ids
    assert "OTHER001" not in ids


def test_by_date_invalid_format_returns_422(api_client: TestClient) -> None:
    """Malformed date string is rejected with 422."""
    resp = api_client.get("/api/races/by_date?date=2024/06/01")
    assert resp.status_code == 422


def test_by_date_entry_summary_includes_horse_name(
    app_with_temp_db: FastAPI,
) -> None:
    """horse_name is populated via bulk Horse lookup in /races/{race_id}."""
    from keiba_ai.core.paths import db_path
    from keiba_ai.db.session import make_engine, session_scope
    from keiba_ai.db.models.horse import Horse as HorseModel

    engine = make_engine(db_path())
    target_date = "2024-06-15"

    with session_scope(engine) as session:
        # Insert race with 1 horse and give the horse a name
        race_id = "HORSENAME01"
        session.add(Race(
            race_id=race_id,
            date=target_date,
            course="東京",
            surface="芝",
            distance=2000,
            race_class="G1",
            n_runners=1,
            payout_win=None,
            payout_place=None,
        ))
        session.flush()
        hid = "HN_001"
        session.add(HorseModel(horse_id=hid, name="テストホース"))
        session.flush()
        session.add(Entry(
            race_id=race_id,
            horse_id=hid,
            post_position=1,
            finish_position=1,
        ))
        session.commit()

    with TestClient(app_with_temp_db) as client:
        resp = client.get(f"/api/races/{race_id}")

    assert resp.status_code == 200
    entries = resp.json()["entries"]
    assert len(entries) == 1
    assert entries[0]["horse_name"] == "テストホース"
