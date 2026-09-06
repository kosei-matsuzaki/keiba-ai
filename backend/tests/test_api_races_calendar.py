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


    def test_graded_returns_every_g3_and_above(self, app_with_temp_db, seed) -> None:
        """G3 以上は 1 日に何本あっても全部返す。

        1 本だけ返していたころは、重賞が 3〜4 本ある日でも残りがカレンダーから
        消えていた（どの日を開くかの判断材料が落ちる）。

        同格の並びは race_id 順 = 場コード順で、レース番号順ではない。
        """
        seed("202605010111", "2026-05-03", course="東京", race_class="G1", name="天皇賞(春)")
        seed("202605020211", "2026-05-03", course="京都", race_class="G3", name="テスト賞")
        seed("202605030311", "2026-05-03", course="新潟", race_class="G2", name="テストS")
        seed("202605010101", "2026-05-03", course="未勝利場", race_class="未勝利", name="3歳未勝利")

        with TestClient(app_with_temp_db) as client:
            days = client.get("/api/races/calendar?from=2026-05-01&to=2026-05-31").json()["days"]
        day = next(d for d in days if d["date"] == "2026-05-03")

        # 格上から順に並ぶ（G1 → G2 → G3）。平場は入らない。
        assert [(g["race_class"], g["name"]) for g in day["graded"]] == [
            ("G1", "天皇賞(春)"),
            ("G2", "テストS"),
            ("G3", "テスト賞"),
        ]
        # 開催場も返す（同じ日に別の場の重賞が並ぶため）
        assert [g["course"] for g in day["graded"]] == ["東京", "新潟", "京都"]

    def test_graded_orders_same_grade_by_race_id_not_race_number(
        self, app_with_temp_db, seed
    ) -> None:
        """同格が 2 場にあるときは場コード順。番号の小さいほうが先とは限らない。"""
        # 場コードは 05 (東京) < 08 (京都)。レース番号は東京 11R > 京都 09R。
        seed("202605010111", "2026-05-10", course="東京", race_class="G3", name="東京の重賞")
        seed("202608010209", "2026-05-10", course="京都", race_class="G3", name="京都の重賞")

        with TestClient(app_with_temp_db) as client:
            days = client.get("/api/races/calendar?from=2026-05-01&to=2026-05-31").json()["days"]
        day = next(d for d in days if d["date"] == "2026-05-10")

        assert [g["name"] for g in day["graded"]] == ["東京の重賞", "京都の重賞"]

    def test_graded_includes_jump_races_and_unlabelled_juusho(
        self, app_with_temp_db, seed
    ) -> None:
        """`_GRADED_CLASSES` の残り 2 系統。どちらも実 DB に存在する。

        障害の重賞は netkeiba が JGI/JGII/JGIII を race_class ではなく名前側に
        持たせるので、race_class は平地と同じ "G1"。DB に JG* という race_class は
        1 件も無く、名前に "(JG" を含むものが 113 件ある。

        格が特定できない "重賞" は 7 件あり、G3 の後ろ・OP の前に並ぶ
        （`_CLASS_PRIORITY` に入れ忘れると未知と同じ最下位に落ちる）。
        """
        seed("202606010111", "2026-06-14", race_class="G1", name="中山グランドジャンプ(JGI)")
        seed("202606010112", "2026-06-14", race_class="重賞", name="葵ステークス(重賞)")
        seed("202606010113", "2026-06-14", race_class="OP", name="オープン特別")

        with TestClient(app_with_temp_db) as client:
            days = client.get("/api/races/calendar?from=2026-06-01&to=2026-06-30").json()["days"]
        day = next(d for d in days if d["date"] == "2026-06-14")

        assert [(g["race_class"], g["name"]) for g in day["graded"]] == [
            ("G1", "中山グランドジャンプ(JGI)"),
            ("重賞", "葵ステークス(重賞)"),
        ]

    def test_unlabelled_juusho_outranks_open(self, app_with_temp_db, seed) -> None:
        """"重賞" を `_CLASS_PRIORITY` に入れ忘れると未知と同じ最下位に落ち、
        同じ日の OP がその日の主役に選ばれてしまう。

        graded の並びでは差が出ない（G1〜G3 の後ろという位置は順位表に無くても
        変わらない）ので、highlight 側でしか確かめられない。
        """
        seed("202606010111", "2026-06-21", race_class="重賞", name="葵ステークス(重賞)")
        seed("202606010112", "2026-06-21", race_class="OP", name="オープン特別")

        with TestClient(app_with_temp_db) as client:
            days = client.get("/api/races/calendar?from=2026-06-01&to=2026-06-30").json()["days"]
        day = next(d for d in days if d["date"] == "2026-06-21")

        assert day["highlight_name"] == "葵ステークス(重賞)"

    def test_graded_excludes_listed_and_open(self, app_with_temp_db, seed) -> None:
        """Listed と OP は重賞ではないので graded に入れない。"""
        seed("202606010111", "2026-06-07", race_class="Listed", name="谷川岳ステークス(L)")
        seed("202606010112", "2026-06-08", race_class="OP", name="オープン特別")

        with TestClient(app_with_temp_db) as client:
            days = client.get("/api/races/calendar?from=2026-06-01&to=2026-06-30").json()["days"]
        by_date = {d["date"]: d for d in days}

        assert by_date["2026-06-07"]["graded"] == []
        assert by_date["2026-06-08"]["graded"] == []
        # graded が空でも、その日の主役は highlight として残る
        assert by_date["2026-06-07"]["highlight_name"] == "谷川岳ステークス(L)"

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
