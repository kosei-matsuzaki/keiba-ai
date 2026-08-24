"""カレンダー / データ取込状況エンドポイントのテスト。

「どの日のデータが手元にあるか」を月表示で示すための API。
1 レースも無い日は返さない（呼び出し側は「返らない日 = 未取得」と扱う）。
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from db.models.entry import Entry
from db.models.horse import Horse
from db.models.race import Race


@pytest.fixture()
def seed(app_with_temp_db: FastAPI):
    """app_with_temp_db と同じ DB にレースを積むヘルパを返す。

    conftest の db_session は in-memory エンジンで api_client とは別 DB なので、
    API テストでは db_path() 経由で同じファイルに書く必要がある。
    """
    from core.paths import db_path
    from db.session import make_engine, session_scope

    engine = make_engine(db_path())

    def _add(
        race_id: str,
        day: str,
        *,
        course: str = "東京",
        race_class: str | None = None,
        name: str | None = None,
        finished: bool = False,
    ) -> None:
        with session_scope(engine) as session:
            session.add(
                Race(
                    race_id=race_id,
                    date=day,
                    course=course,
                    surface="芝",
                    distance=2000,
                    race_class=race_class,
                    name=name,
                    n_runners=2,
                )
            )
            session.flush()
            for i in range(2):
                hid = f"H_{race_id}_{i}"
                if not session.get(Horse, hid):
                    session.add(Horse(horse_id=hid, name=None))
                    session.flush()
                session.add(
                    Entry(
                        race_id=race_id,
                        horse_id=hid,
                        post_position=i + 1,
                        finish_position=(i + 1) if finished else None,
                    )
                )

    yield _add
    engine.dispose()


class TestRacesCalendar:
    def test_returns_only_days_that_have_races(self, app_with_temp_db, seed) -> None:
        seed("CAL0101", "2026-05-02")
        seed("CAL0102", "2026-05-02")
        seed("CAL0201", "2026-05-09")

        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/races/calendar?from=2026-05-01&to=2026-05-31")

        assert resp.status_code == 200
        days = resp.json()["days"]
        assert [d["date"] for d in days] == ["2026-05-02", "2026-05-09"]
        assert days[0]["race_count"] == 2
        assert days[1]["race_count"] == 1

    def test_result_count_distinguishes_shutuba_only(self, app_with_temp_db, seed) -> None:
        """出馬表だけの日は result_count = 0 になる（カレンダーで色を変えるため）。"""
        seed("CAL0301", "2026-05-16", finished=True)
        seed("CAL0302", "2026-05-16", finished=False)
        seed("CAL0401", "2026-05-23", finished=False)

        with TestClient(app_with_temp_db) as client:
            days = client.get("/api/races/calendar?from=2026-05-01&to=2026-05-31").json()["days"]
        by_date = {d["date"]: d for d in days}

        assert by_date["2026-05-16"]["race_count"] == 2
        assert by_date["2026-05-16"]["result_count"] == 1
        assert by_date["2026-05-23"]["result_count"] == 0

    def test_lists_courses_without_duplicates(self, app_with_temp_db, seed) -> None:
        seed("CAL0501", "2026-06-06", course="東京")
        seed("CAL0502", "2026-06-06", course="東京")
        seed("CAL0503", "2026-06-06", course="阪神")

        with TestClient(app_with_temp_db) as client:
            days = client.get("/api/races/calendar?from=2026-06-01&to=2026-06-30").json()["days"]

        assert days[0]["courses"] == ["東京", "阪神"]

    def test_highlight_picks_the_highest_grade(self, app_with_temp_db, seed) -> None:
        seed("CAL0601", "2026-06-13", race_class="未勝利", name="3歳未勝利")
        seed("CAL0602", "2026-06-13", race_class="G3", name="エプソムC")
        seed("CAL0603", "2026-06-13", race_class="OP", name="オープン")

        with TestClient(app_with_temp_db) as client:
            days = client.get("/api/races/calendar?from=2026-06-01&to=2026-06-30").json()["days"]

        assert days[0]["highlight_name"] == "エプソムC"
        assert days[0]["highlight_class"] == "G3"

    def test_no_highlight_when_only_flat_races(self, app_with_temp_db, seed) -> None:
        """平場しか無い日は名前を出さない（「未勝利」と出しても情報にならない）。"""
        seed("CAL0701", "2026-06-20", race_class="未勝利", name="3歳未勝利")

        with TestClient(app_with_temp_db) as client:
            days = client.get("/api/races/calendar?from=2026-06-01&to=2026-06-30").json()["days"]

        assert days[0]["highlight_name"] is None
        assert days[0]["highlight_race_id"] is None

    def test_empty_range_returns_empty_list(self, app_with_temp_db) -> None:
        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/races/calendar?from=2030-01-01&to=2030-01-31")
        assert resp.status_code == 200
        assert resp.json()["days"] == []

    def test_invalid_date_returns_422(self, app_with_temp_db) -> None:
        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/races/calendar?from=2026-13-01&to=2026-13-31")
        assert resp.status_code == 422

    def test_reversed_range_returns_422(self, app_with_temp_db) -> None:
        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/races/calendar?from=2026-05-31&to=2026-05-01")
        assert resp.status_code == 422


class TestDataCoverage:
    def test_reports_span_and_counts(self, app_with_temp_db, seed) -> None:
        seed("COV0101", "2026-01-10", finished=True)
        seed("COV0201", "2026-03-14", finished=False)

        with TestClient(app_with_temp_db) as client:
            data = client.get("/api/races/coverage").json()

        assert data["first_date"] == "2026-01-10"
        assert data["last_date"] == "2026-03-14"
        assert data["race_count"] == 2
        assert data["result_count"] == 1
        assert data["entry_count"] == 4

    def test_empty_db_returns_nulls(self, app_with_temp_db) -> None:
        with TestClient(app_with_temp_db) as client:
            data = client.get("/api/races/coverage").json()

        assert data["first_date"] is None
        assert data["last_date"] is None
        assert data["race_count"] == 0
        assert data["result_count"] == 0
