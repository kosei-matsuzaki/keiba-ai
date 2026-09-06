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


def _seed_rivals(session: Session) -> None:
    """R1 に H1 以外の馬を足して、上がり順位が付く状態にする。

    H1 の R1 での上がりは 35.0（`_seed` が 34.0 + i で入れる）。
    ライバルを 34.5 / 35.0 / 36.0 で入れると、34.5 が 1 位、35.0 が 2 頭で
    同着 2 位、36.0 が 4 位になる。
    """
    from db.models.entry import Entry
    from db.models.horse import Horse

    for hid, agari in (("H2", 34.5), ("H3", 35.0), ("H4", 36.0)):
        session.add(Horse(horse_id=hid, name=hid))
        session.add(Entry(race_id="R1", horse_id=hid, post_position=9, agari_3f=agari))
    session.commit()


def test_agari_rank_is_within_the_race(app_with_temp_db: FastAPI, tmp_path: Path) -> None:
    """上がりは「そのレースで何番目か」で読むもの。値だけでは色を付けられない。

    同じ 33.8 でも高速馬場なら平凡、時計のかかる馬場なら最速なので、
    順位はレース内の他馬と比べて出す。
    """
    from core.paths import db_path
    from db.session import make_engine, session_scope

    with session_scope(make_engine(db_path())) as session:
        _seed(session)
        _seed_rivals(session)

    with _client(app_with_temp_db) as client:
        resp = client.get("/api/horses/H1/history")
    runs = {r["race_id"]: r for r in resp.json()["runs"]}

    # H1 の 35.0 は 34.5 に次ぐ 2 位（H3 と同着）
    assert runs["R1"]["agari_rank"] == 2
    # R2 / R3 は H1 しか走っていないので単独 1 位
    assert runs["R2"]["agari_rank"] == 1


def test_agari_rank_is_none_without_agari(app_with_temp_db: FastAPI, tmp_path: Path) -> None:
    """上がりが取れていない走りには順位を付けない（無彩色で出す側の入力）。"""
    from core.paths import db_path
    from db.models.entry import Entry
    from db.models.horse import Horse
    from db.models.race import Race
    from db.session import make_engine, session_scope

    with session_scope(make_engine(db_path())) as session:
        session.add(Horse(horse_id="H9", name="計測なし"))
        session.add(
            Race(
                race_id="RX",
                date="2024-05-05",
                course="中山",
                surface="ダ",
                distance=1200,
                n_runners=10,
            )
        )
        session.flush()
        session.add(Entry(race_id="RX", horse_id="H9", post_position=1, agari_3f=None))

    with _client(app_with_temp_db) as client:
        resp = client.get("/api/horses/H9/history")
    assert resp.json()["runs"][0]["agari_rank"] is None
