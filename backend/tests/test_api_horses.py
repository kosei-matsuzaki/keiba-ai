"""Tests for /api/horses/{horse_id}/history.

出走馬一覧から「この馬は前走どうだったか」を引く API。**当日以降は返さない** —
予想の根拠として読むものなので、特徴量側 (features/builder.py) と同じ制約を持つ。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session


def _seed(session: Session) -> None:
    from db.models.entry import Entry
    from db.models.horse import Horse
    from db.models.jockey import Jockey
    from db.models.race import Race

    session.add(Horse(horse_id="H1", name="テスト馬"))
    session.add(Jockey(jockey_id="J1", name="テスト騎手"))
    for i, date in enumerate(["2024-01-06", "2024-02-10", "2024-03-16"], start=1):
        session.add(
            Race(
                race_id=f"R{i}",
                date=date,
                course="東京",
                surface="芝",
                distance=1600 + i * 200,
                race_class="1勝クラス",
                name=f"レース{i}",
                n_runners=12,
                track_condition="良",
            )
        )
        session.flush()
        session.add(
            Entry(
                race_id=f"R{i}",
                horse_id="H1",
                jockey_id="J1",
                post_position=i,
                finish_position=i,
                odds_win=3.0 + i,
                popularity=i,
                agari_3f=34.0 + i,
                passing="3-3",
                finish_time=95.0 + i,
            )
        )
    session.commit()


def _client(app: FastAPI) -> TestClient:
    return TestClient(app)


def test_history_returns_runs_newest_first(app_with_temp_db: FastAPI, tmp_path: Path) -> None:
    from core.paths import db_path
    from db.session import make_engine, session_scope

    with session_scope(make_engine(db_path())) as session:
        _seed(session)

    with _client(app_with_temp_db) as client:
        resp = client.get("/api/horses/H1/history")

    assert resp.status_code == 200
    runs = resp.json()["runs"]
    assert [r["date"] for r in runs] == ["2024-03-16", "2024-02-10", "2024-01-06"]
    assert runs[0]["race_name"] == "レース3"
    assert runs[0]["jockey_name"] == "テスト騎手"
    assert runs[0]["agari_3f"] == 37.0


def test_history_excludes_the_race_day_itself(
    app_with_temp_db: FastAPI, tmp_path: Path
) -> None:
    """**同じ日は返さない。** 当日の結果は「前走までの成績」ではない。"""
    from core.paths import db_path
    from db.session import make_engine, session_scope

    with session_scope(make_engine(db_path())) as session:
        _seed(session)

    with _client(app_with_temp_db) as client:
        resp = client.get("/api/horses/H1/history?before=2024-02-10")

    dates = [r["date"] for r in resp.json()["runs"]]
    assert dates == ["2024-01-06"]


def test_history_limit(app_with_temp_db: FastAPI, tmp_path: Path) -> None:
    from core.paths import db_path
    from db.session import make_engine, session_scope

    with session_scope(make_engine(db_path())) as session:
        _seed(session)

    with _client(app_with_temp_db) as client:
        resp = client.get("/api/horses/H1/history?limit=2")

    assert len(resp.json()["runs"]) == 2


def test_history_unknown_horse_is_empty(app_with_temp_db: FastAPI, tmp_path: Path) -> None:
    """未知の馬でも 404 にしない (出走馬一覧から引くので、空で返す方が扱いやすい)。"""
    with _client(app_with_temp_db) as client:
        resp = client.get("/api/horses/NOPE/history")
    assert resp.status_code == 200
    assert resp.json()["runs"] == []
