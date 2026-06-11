"""Tests for features/horse_history.py.

Critical: verifies that compute_horse_history never uses data from
races on or after before_date (leakage prevention).
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import db.models  # noqa: F401
from db.base import Base
from db.models.entry import Entry
from db.models.horse import Horse
from db.models.race import Race
from features.extractors.horse_history import (
    build_horse_history_cache,
    compute_horse_history,
    compute_horse_history_from_cache,
    is_high_class,
    parse_margin,
    parse_passing,
    race_class_weight,
)


@pytest.fixture()
def leakage_engine():
    """Create a DB with one horse that has 3 races: past, today, future."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)

    today = date(2024, 6, 15)
    past = today - timedelta(days=10)
    future = today + timedelta(days=10)

    with Session(engine) as session:
        session.add(Horse(horse_id="H001", name=None))
        for rid, _d, _pos in [
            ("R001", past, 1),
            ("R002", today, 2),
            ("R003", future, 3),
        ]:
            session.add(
                Race(
                    race_id=rid,
                    date=_d.isoformat(),
                    course="東京",
                    surface="芝",
                    distance=1600,
                    n_runners=8,
                )
            )
        session.flush()
        for rid, _d, pos in [
            ("R001", past, 1),
            ("R002", today, 2),
            ("R003", future, 3),
        ]:
            session.add(
                Entry(
                    race_id=rid,
                    horse_id="H001",
                    post_position=1,
                    finish_position=pos,
                )
            )
        session.commit()

    yield engine
    engine.dispose()


def test_leakage_before_date_only(leakage_engine):
    """Only the past race (finish=1) should contribute to the aggregate."""
    with Session(leakage_engine) as session:
        result = compute_horse_history(
            session,
            horse_id="H001",
            before_date=date(2024, 6, 15),  # today's race is NOT included
        )

    # Only R001 (past, pos=1) should be included
    assert result["recent_n_starts"] == 1
    assert result["recent_avg_finish"] == pytest.approx(1.0)


def test_leakage_future_not_included(leakage_engine):
    """With before_date = past, only zero races qualify."""
    with Session(leakage_engine) as session:
        result = compute_horse_history(
            session,
            horse_id="H001",
            before_date=date(2024, 6, 1),  # before all races
        )

    assert result["recent_n_starts"] == 0
    assert math.isnan(result["recent_avg_finish"])


def test_no_history_returns_nan(leakage_engine):
    with Session(leakage_engine) as session:
        result = compute_horse_history(
            session,
            horse_id="UNKNOWN",
            before_date=date(2024, 6, 15),
        )

    assert result["recent_n_starts"] == 0
    assert math.isnan(result["recent_avg_finish"])


def test_same_distance_filter(leakage_engine):
    with Session(leakage_engine) as session:
        result = compute_horse_history(
            session,
            horse_id="H001",
            before_date=date(2024, 6, 15),
            distance=1600,
        )
    # R001 has distance=1600 and is before before_date
    assert result["starts_same_distance"] == 1


def test_same_course_filter(leakage_engine):
    with Session(leakage_engine) as session:
        result = compute_horse_history(
            session,
            horse_id="H001",
            before_date=date(2024, 6, 15),
            course="東京",
        )
    assert result["starts_same_course"] == 1


# ── PR-C: new field tests ─────────────────────────────────────────────────────


@pytest.fixture()
def rich_engine():
    """DB with one horse, 4 past races with varied data for PR-C feature tests."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)

    base = date(2024, 6, 15)

    with Session(engine) as session:
        session.add(Horse(horse_id="H001", name=None))
        # Races at different dates (all before 2024-06-15)
        # R1: 30 days ago, finish=1, agari_3f=34.5, 東京1600
        # R2: 20 days ago, finish=2, agari_3f=35.0, 東京1600
        # R3: 10 days ago, finish=3, agari_3f=None, 中山1800
        # R4:  5 days ago, finish=1, agari_3f=33.8, 東京1600
        races = [
            ("R001", base - timedelta(days=30), 1,   34.5, "東京", 1600),
            ("R002", base - timedelta(days=20), 2,   35.0, "東京", 1600),
            ("R003", base - timedelta(days=10), 3,   None, "中山", 1800),
            ("R004", base - timedelta(days=5),  1,   33.8, "東京", 1600),
        ]
        for rid, d, _finish, _agari, course, dist in races:
            session.add(
                Race(
                    race_id=rid,
                    date=d.isoformat(),
                    course=course,
                    surface="芝",
                    distance=dist,
                    n_runners=8,
                )
            )
        session.flush()
        for rid, _d, finish, agari, _course, _dist in races:
            session.add(
                Entry(
                    race_id=rid,
                    horse_id="H001",
                    post_position=1,
                    finish_position=finish,
                    agari_3f=agari,
                )
            )
        session.commit()

    yield engine
    engine.dispose()


def test_compute_horse_history_includes_avg_agari_3f(rich_engine):
    """recent_avg_agari_3f averages only non-None agari_3f values in last 5 races."""
    with Session(rich_engine) as session:
        result = compute_horse_history(
            session,
            horse_id="H001",
            before_date=date(2024, 6, 15),
        )

    # Valid agari values in last 5 races: 34.5, 35.0, 33.8 (R3 is None)
    expected = (34.5 + 35.0 + 33.8) / 3
    assert result["recent_avg_agari_3f"] == pytest.approx(expected)


def test_compute_horse_history_includes_days_since_last_race(rich_engine):
    """days_since_last_race is the gap from the most recent race to before_date."""
    with Session(rich_engine) as session:
        result = compute_horse_history(
            session,
            horse_id="H001",
            before_date=date(2024, 6, 15),
        )

    # Most recent race is R4, 5 days before before_date
    assert result["days_since_last_race"] == pytest.approx(5.0)


def test_compute_horse_history_includes_wins_same_course(rich_engine):
    """wins_same_course counts finish_position==1 on the given course."""
    with Session(rich_engine) as session:
        result = compute_horse_history(
            session,
            horse_id="H001",
            before_date=date(2024, 6, 15),
            course="東京",
        )

    # R001 (finish=1) and R004 (finish=1) are both at 東京
    assert result["wins_same_course"] == 2


def test_compute_horse_history_includes_recent_finish_n(rich_engine):
    """recent_finish_1/2/3 are the last 3 finish positions in reverse-date order."""
    with Session(rich_engine) as session:
        result = compute_horse_history(
            session,
            horse_id="H001",
            before_date=date(2024, 6, 15),
        )

    # Sorted desc by date: R4(finish=1), R3(finish=3), R2(finish=2), R1(finish=1)
    assert result["recent_finish_1"] == pytest.approx(1.0)
    assert result["recent_finish_2"] == pytest.approx(3.0)
    assert result["recent_finish_3"] == pytest.approx(2.0)


def test_compute_horse_history_includes_course_place_rate(rich_engine):
    """horse_course_place_rate = (finishes <= 3 at course) / starts_same_course."""
    with Session(rich_engine) as session:
        result = compute_horse_history(
            session,
            horse_id="H001",
            before_date=date(2024, 6, 15),
            course="東京",
        )
    # At 東京: R1(finish=1), R2(finish=2), R4(finish=1) → 3 places / 3 starts
    assert result["starts_same_course"] == 3
    assert result["horse_course_place_rate"] == pytest.approx(1.0)


def test_horse_course_place_rate_nan_without_course(rich_engine):
    """horse_course_place_rate is NaN when course filter is not provided."""
    with Session(rich_engine) as session:
        result = compute_horse_history(
            session,
            horse_id="H001",
            before_date=date(2024, 6, 15),
        )
    assert math.isnan(result["horse_course_place_rate"])


def test_horse_history_excludes_races_after_before_date(rich_engine):
    """Leakage prevention: races on or after before_date must not appear.

    The rich_engine fixture has 4 races:
      R1: 30d ago (2024-05-16), R2: 20d ago (2024-05-26),
      R3: 10d ago (2024-06-05), R4:  5d ago (2024-06-10).

    Using cutoff = 2024-05-28 (17 days before base 2024-06-15) means only
    R1 (2024-05-16) and R2 (2024-05-26) qualify (both < 2024-05-28).
    R3 and R4 are on or after the cutoff and must be excluded.
    """
    base = date(2024, 6, 15)
    # cutoff = 2024-05-28; R1(5/16) and R2(5/26) are before it
    before = base - timedelta(days=18)  # = 2024-05-28

    with Session(rich_engine) as session:
        result = compute_horse_history(
            session,
            horse_id="H001",
            before_date=before,
        )

    assert result["recent_n_starts"] == 2
    # Most recent qualifying race is R2 = 2024-05-26
    r2_date = base - timedelta(days=20)  # 2024-05-26
    expected_days = float((before - r2_date).days)
    assert result["days_since_last_race"] == pytest.approx(expected_days)


# ---------------------------------------------------------------------------
# Cache parity tests (build_horse_history_cache + compute_horse_history_from_cache)
# ---------------------------------------------------------------------------


def _assert_dicts_equal(a: dict, b: dict, *, ctx: str = "") -> None:
    """Compare two horse_history result dicts allowing NaN == NaN."""
    assert a.keys() == b.keys(), f"keys differ {ctx}: {a.keys()} vs {b.keys()}"
    for k in a:
        va, vb = a[k], b[k]
        # NaN を等しく扱う
        if isinstance(va, float) and math.isnan(va):
            assert isinstance(vb, float) and math.isnan(vb), f"{k} {ctx}: {va!r} vs {vb!r}"
        else:
            assert va == vb, f"{k} {ctx}: {va!r} vs {vb!r}"


def test_cache_parity_with_sql_version(rich_engine):
    """compute_horse_history_from_cache の出力が compute_horse_history と
    bit-for-bit (NaN 含む) 一致することを多パターンで確認する。"""
    base = date(2024, 6, 15)

    test_cases = [
        # (before_date, distance, course)
        (base, None, None),
        (base, 1600, "東京"),
        (base, 1600, None),
        (base, None, "東京"),
        (base - timedelta(days=18), 1600, "東京"),  # 一部レースだけ含む cutoff
        (base + timedelta(days=365), 1600, "東京"),  # 全レース含む
    ]

    with Session(rich_engine) as session:
        cache = build_horse_history_cache(session)
        for before, distance, course in test_cases:
            ctx = f"(before={before}, dist={distance}, course={course})"
            sql_result = compute_horse_history(
                session, horse_id="H001", before_date=before,
                distance=distance, course=course,
            )
            cache_result = compute_horse_history_from_cache(
                cache, horse_id="H001", before_date=before,
                distance=distance, course=course,
            )
            _assert_dicts_equal(sql_result, cache_result, ctx=ctx)


def test_cache_parity_unknown_horse(rich_engine):
    """履歴のない horse_id でも SQL 版と同一 (NaN/0) を返す。"""
    with Session(rich_engine) as session:
        cache = build_horse_history_cache(session)
        sql_result = compute_horse_history(
            session, horse_id="UNKNOWN", before_date=date(2024, 6, 15),
        )
        cache_result = compute_horse_history_from_cache(
            cache, horse_id="UNKNOWN", before_date=date(2024, 6, 15),
        )
        _assert_dicts_equal(sql_result, cache_result, ctx="(unknown horse)")


def test_cache_single_sql_query(rich_engine):
    """build_horse_history_cache が 1 SQL で全行を取得することを確認。

    sqlalchemy event listener で session 経由のクエリ数を数える。
    """
    from sqlalchemy import event

    queries: list[str] = []

    def _capture(conn, cursor, statement, params, context, executemany):  # noqa: ARG001
        queries.append(statement)

    event.listen(rich_engine, "before_cursor_execute", _capture)
    try:
        with Session(rich_engine) as session:
            cache = build_horse_history_cache(session)
        assert cache.df is not None
        # Building the cache should emit at most 1 SELECT against entries (the
        # join with races counts as the same statement).
        select_queries = [q for q in queries if q.strip().lower().startswith("select")]
        assert len(select_queries) == 1, (
            f"expected 1 SELECT, got {len(select_queries)}: {select_queries}"
        )
    finally:
        event.remove(rich_engine, "before_cursor_execute", _capture)


# ---------------------------------------------------------------------------
# Phase C: 脚質 / 瞬発ピーク / 前走 class・斤量
# ---------------------------------------------------------------------------


@pytest.fixture()
def phase_c_engine():
    """passing / weight_carried / race_class を埋めた 2 戦の履歴。"""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    base = date(2024, 6, 15)

    with Session(engine) as session:
        session.add(Horse(horse_id="H001", name=None))
        # R1 (古い): passing 2-3 (前付け), agari 34.0, 斤量 54, G3
        # R2 (新しい=前走): passing 8-7 (後方), agari 35.5, 斤量 55, G1
        races = [
            ("R001", base - timedelta(days=30), "2-3", 34.0, 54.0, "G3", 1, 10),
            ("R002", base - timedelta(days=5),  "8-7", 35.5, 55.0, "G1", 4, 10),
        ]
        for rid, d, _pass, _ag, _wc, rc, _fin, nr in races:
            session.add(Race(
                race_id=rid, date=d.isoformat(), course="東京", surface="芝",
                distance=1600, race_class=rc, n_runners=nr,
            ))
        session.flush()
        for rid, _d, passing, agari, wc, _rc, fin, _nr in races:
            session.add(Entry(
                race_id=rid, horse_id="H001", post_position=1,
                finish_position=fin, agari_3f=agari, passing=passing,
                weight_carried=wc,
            ))
        session.commit()
    yield engine
    engine.dispose()


def test_phase_c_position_ratio_and_best_agari(phase_c_engine):
    """early/late position ratio, best agari, 前走 class/斤量 が正しく計算される。"""
    with Session(phase_c_engine) as session:
        r = compute_horse_history(
            session, horse_id="H001", before_date=date(2024, 6, 15),
            distance=1600, course="東京",
        )
    # early = mean(2/10, 8/10) = 0.5 ; late = mean(3/10, 7/10) = 0.5
    assert r["recent_early_position_ratio"] == pytest.approx(0.5)
    assert r["recent_late_position_ratio"] == pytest.approx(0.5)
    # best agari = min(34.0, 35.5) = 34.0
    assert r["recent_best_agari_3f"] == pytest.approx(34.0)
    # 前走 = R002 (most recent): G1 weight, 斤量 55
    assert r["last_class_weight"] == pytest.approx(float(race_class_weight("G1")))
    assert r["last_weight_carried"] == pytest.approx(55.0)


def test_phase_c_cache_parity(phase_c_engine):
    """新規 Phase C キーも SQL/cache で一致する。"""
    with Session(phase_c_engine) as session:
        cache = build_horse_history_cache(session)
        sql = compute_horse_history(
            session, horse_id="H001", before_date=date(2024, 6, 15),
            distance=1600, course="東京",
        )
        cac = compute_horse_history_from_cache(
            cache, horse_id="H001", before_date=date(2024, 6, 15),
            distance=1600, course="東京",
        )
    _assert_dicts_equal(sql, cac, ctx="(phase C)")


def test_phase_c_clip_when_passing_exceeds_runners(phase_c_engine):
    """passing 位置 > 頭数 の異常データでも ratio は [0,1] にクリップされる。"""
    # n_runners を 5 に上書きして passing 8-7 が 8/5=1.6 → 1.0 にクリップされるか
    with Session(phase_c_engine) as session:
        session.query(Race).filter(Race.race_id == "R002").update({"n_runners": 5})
        session.commit()
        r = compute_horse_history(
            session, horse_id="H001", before_date=date(2024, 6, 15),
        )
    # R001: early 2/10=0.2, late 3/10=0.3 ; R002: early min(8/5,1)=1.0, late min(7/5,1)=1.0
    assert r["recent_early_position_ratio"] == pytest.approx((0.2 + 1.0) / 2)
    assert r["recent_late_position_ratio"] == pytest.approx((0.3 + 1.0) / 2)


# ---------------------------------------------------------------------------
# Q4: race-level features
# ---------------------------------------------------------------------------


@pytest.fixture()
def class_engine():
    """Horse with mixed-class history: G1 win, G3 place, OP runs, 1勝 race."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)

    base = date(2024, 6, 15)
    races = [
        # (race_id, days_ago, race_class, finish_position)
        ("RC1", 30, "1勝クラス", 5),
        ("RC2", 20, "OP", 3),
        ("RC3", 15, "G1", 1),
        ("RC4", 10, "G3", 2),
        ("RC5", 5,  "OP", 8),
    ]
    with Session(engine) as session:
        session.add(Horse(horse_id="H1", name=None))
        for rid, days, rc, _pos in races:
            session.add(Race(
                race_id=rid,
                date=(base - timedelta(days=days)).isoformat(),
                course="東京",
                surface="芝",
                distance=1600,
                n_runners=10,
                race_class=rc,
            ))
        session.flush()
        for rid, _days, _rc, pos in races:
            session.add(Entry(
                race_id=rid,
                horse_id="H1",
                post_position=1,
                finish_position=pos,
            ))
        session.commit()
    yield engine
    engine.dispose()


def test_race_class_weight_lookup():
    assert race_class_weight("G1") == 8
    assert race_class_weight("GI") == 8
    assert race_class_weight("G3") == 6
    assert race_class_weight("Listed") == 5
    assert race_class_weight("OP") == 5
    assert race_class_weight("3勝クラス") == 4
    assert race_class_weight("未勝利") == 1
    assert race_class_weight("新馬") == 1
    assert race_class_weight(None) == 1
    assert race_class_weight("謎クラス") == 1


def test_is_high_class():
    assert is_high_class("G1") is True
    assert is_high_class("GIII") is True
    assert is_high_class("Listed") is False
    assert is_high_class("OP") is False
    assert is_high_class("1勝クラス") is False
    assert is_high_class(None) is False


def test_horse_history_includes_class_features_sql(class_engine):
    """SQL 版で recent_avg_class_weight / high_class_* が正しい。"""
    with Session(class_engine) as session:
        result = compute_horse_history(
            session, horse_id="H1", before_date=date(2024, 6, 15)
        )
    # last 5: RC5(OP=5), RC4(G3=6), RC3(G1=8), RC2(OP=5), RC1(1勝=2) → mean=5.2
    assert result["recent_avg_class_weight"] == pytest.approx(5.2)
    assert result["high_class_starts"] == 2  # G1 + G3
    assert result["high_class_places"] == 2  # G1 1着 + G3 2着


def test_horse_history_class_features_cache_parity(class_engine):
    """SQL 版と cache 版の出力が完全一致 (Q4 fields 含む)。"""
    base = date(2024, 6, 15)
    with Session(class_engine) as session:
        cache = build_horse_history_cache(session)
        sql_r = compute_horse_history(session, "H1", before_date=base)
        cache_r = compute_horse_history_from_cache(cache, "H1", before_date=base)
    for k in sql_r:
        sv, cv = sql_r[k], cache_r[k]
        if isinstance(sv, float) and math.isnan(sv):
            assert isinstance(cv, float) and math.isnan(cv), f"{k}: {sv} vs {cv}"
        else:
            assert sv == cv, f"{k}: {sv} vs {cv}"


def test_class_features_zero_for_unknown_horse(class_engine):
    with Session(class_engine) as session:
        result = compute_horse_history(
            session, horse_id="UNKNOWN", before_date=date(2024, 6, 15)
        )
    assert math.isnan(result["recent_avg_class_weight"])
    assert result["high_class_starts"] == 0
    assert result["high_class_places"] == 0


def test_class_features_cutoff_excludes_future(class_engine):
    """before_date で過去だけに絞る → G1/G3 は除外される。"""
    with Session(class_engine) as session:
        cache = build_horse_history_cache(session)
        cutoff = date(2024, 5, 28)  # RC1, RC2 のみ該当
        cr = compute_horse_history_from_cache(cache, "H1", before_date=cutoff)
    assert cr["recent_n_starts"] == 2
    assert cr["recent_avg_class_weight"] == pytest.approx(3.5)  # (5+2)/2
    assert cr["high_class_starts"] == 0
    assert cr["high_class_places"] == 0


# ---------------------------------------------------------------------------
# Phase B: margin / finish_time / passing 由来 features
# ---------------------------------------------------------------------------


def test_parse_margin_literals():
    assert parse_margin("ハナ") == pytest.approx(0.05)
    assert parse_margin("アタマ") == pytest.approx(0.10)
    assert parse_margin("クビ") == pytest.approx(0.15)
    assert parse_margin("同着") == 0.0
    assert parse_margin("大") == pytest.approx(12.0)
    assert parse_margin("大差") == pytest.approx(12.0)


def test_parse_margin_fractions():
    # 純粋分数
    assert parse_margin("1/2") == pytest.approx(0.5)
    assert parse_margin("3/4") == pytest.approx(0.75)
    # 整数 + 分数
    assert parse_margin("1.1/4") == pytest.approx(1.25)
    assert parse_margin("1.1/2") == pytest.approx(1.5)
    assert parse_margin("2.3/4") == pytest.approx(2.75)
    assert parse_margin("3.1/2") == pytest.approx(3.5)


def test_parse_margin_integers():
    assert parse_margin("1") == 1.0
    assert parse_margin("10") == 10.0


def test_parse_margin_invalid():
    assert parse_margin(None) is None
    assert parse_margin("") is None
    assert parse_margin("3+ハナ") is None  # 稀少 hybrid 表記
    assert parse_margin("謎") is None


def test_parse_passing_basic():
    assert parse_passing("2-2") == [2, 2]
    assert parse_passing("11-12-11-10") == [11, 12, 11, 10]
    assert parse_passing("1-1-1-1") == [1, 1, 1, 1]


def test_parse_passing_invalid():
    assert parse_passing(None) is None
    assert parse_passing("") is None
    assert parse_passing("abc") is None


@pytest.fixture()
def phase_b_engine():
    """Horse with rich margin / finish_time / passing data over 5 races.

    R1 (30d ago): 1着 / margin=None (winner) / time=80.0 / 1200m / passing=2-2
    R2 (20d ago): 2着 / margin=クビ / time=85.0 / 1400m / passing=3-3
    R3 (15d ago): 5着 / margin=2 / time=120.0 / 1800m / passing=8-7-6-5
    R4 (10d ago): 1着 / margin=同着 / time=100.0 / 1600m / passing=1-1
    R5 (5d ago):  3着 / margin=1/2 / time=95.0 / 1600m / passing=4-3-2
    """
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)

    base = date(2024, 6, 15)
    races = [
        # (race_id, days_ago, finish, margin, time, distance, passing)
        ("PB1", 30, 1, None,    80.0,  1200, "2-2"),
        ("PB2", 20, 2, "クビ",  85.0,  1400, "3-3"),
        ("PB3", 15, 5, "2",     120.0, 1800, "8-7-6-5"),
        ("PB4", 10, 1, "同着",  100.0, 1600, "1-1"),
        ("PB5", 5,  3, "1/2",   95.0,  1600, "4-3-2"),
    ]
    with Session(engine) as session:
        session.add(Horse(horse_id="HB", name=None))
        for rid, days, _f, _m, _t, dist, _p in races:
            session.add(Race(
                race_id=rid,
                date=(base - timedelta(days=days)).isoformat(),
                course="東京",
                surface="芝",
                distance=dist,
                n_runners=10,
            ))
        session.flush()
        for rid, _days, finish, margin, ftime, _dist, passing in races:
            session.add(Entry(
                race_id=rid,
                horse_id="HB",
                post_position=1,
                finish_position=finish,
                finish_time=ftime,
                margin=margin,
                passing=passing,
            ))
        session.commit()
    yield engine
    engine.dispose()


def test_phase_b_recent_avg_margin(phase_b_engine):
    """勝ち馬 (R1=None, R4=同着) は 0 馬身として扱う。"""
    with Session(phase_b_engine) as session:
        result = compute_horse_history(session, "HB", before_date=date(2024, 6, 15))
    # R1=0 (winner), R2=0.15 (クビ), R3=2.0, R4=0 (同着→0), R5=0.5
    expected = (0.0 + 0.15 + 2.0 + 0.0 + 0.5) / 5
    assert result["recent_avg_margin"] == pytest.approx(expected)


def test_phase_b_recent_avg_finish_time_norm(phase_b_engine):
    """finish_time / distance の平均。"""
    with Session(phase_b_engine) as session:
        result = compute_horse_history(session, "HB", before_date=date(2024, 6, 15))
    expected = (
        80.0 / 1200 + 85.0 / 1400 + 120.0 / 1800 + 100.0 / 1600 + 95.0 / 1600
    ) / 5
    assert result["recent_avg_finish_time_norm"] == pytest.approx(expected)


def test_phase_b_recent_best_margin_in_top3(phase_b_engine):
    """top-3 のうち最小着差: R1(0)/R2(0.15)/R4(0)/R5(0.5) → min=0"""
    with Session(phase_b_engine) as session:
        result = compute_horse_history(session, "HB", before_date=date(2024, 6, 15))
    assert result["recent_best_margin_in_top3"] == pytest.approx(0.0)


def test_phase_b_recent_avg_position_change(phase_b_engine):
    """passing 末尾と finish_position の差の平均 (正なら追い込み)。

    R1: last=2, finish=1 → +1
    R2: last=3, finish=2 → +1
    R3: last=5, finish=5 → 0
    R4: last=1, finish=1 → 0
    R5: last=2, finish=3 → -1
    avg = (1+1+0+0-1)/5 = 0.2
    """
    with Session(phase_b_engine) as session:
        result = compute_horse_history(session, "HB", before_date=date(2024, 6, 15))
    assert result["recent_avg_position_change"] == pytest.approx(0.2)


def test_phase_b_recent_passing_volatility(phase_b_engine):
    """各 race の通過順位 std の平均 (population std)。

    R1: [2,2] std=0
    R2: [3,3] std=0
    R3: [8,7,6,5] std≈1.118
    R4: [1,1] std=0
    R5: [4,3,2] std≈0.816
    """
    import statistics as _stats

    with Session(phase_b_engine) as session:
        result = compute_horse_history(session, "HB", before_date=date(2024, 6, 15))
    expected = (
        _stats.pstdev([2, 2])
        + _stats.pstdev([3, 3])
        + _stats.pstdev([8, 7, 6, 5])
        + _stats.pstdev([1, 1])
        + _stats.pstdev([4, 3, 2])
    ) / 5
    assert result["recent_passing_volatility"] == pytest.approx(expected)


def test_phase_b_cache_parity(phase_b_engine):
    """SQL 版と cache 版が完全一致 (Phase B fields 含む)。"""
    base = date(2024, 6, 15)
    with Session(phase_b_engine) as session:
        cache = build_horse_history_cache(session)
        sql_r = compute_horse_history(session, "HB", before_date=base)
        cache_r = compute_horse_history_from_cache(cache, "HB", before_date=base)
    for k in sql_r:
        sv, cv = sql_r[k], cache_r[k]
        if isinstance(sv, float) and math.isnan(sv):
            assert isinstance(cv, float) and math.isnan(cv), f"{k}: {sv!r} vs {cv!r}"
        else:
            assert sv == pytest.approx(cv), f"{k}: {sv!r} vs {cv!r}"


def test_phase_b_features_nan_for_unknown_horse(phase_b_engine):
    """履歴 0 の馬は Phase B fields も NaN を返す。"""
    with Session(phase_b_engine) as session:
        result = compute_horse_history(
            session, "UNKNOWN", before_date=date(2024, 6, 15)
        )
    assert math.isnan(result["recent_avg_margin"])
    assert math.isnan(result["recent_avg_finish_time_norm"])
    assert math.isnan(result["recent_best_margin_in_top3"])
    assert math.isnan(result["recent_avg_position_change"])
    assert math.isnan(result["recent_passing_volatility"])


def test_phase_b_no_top3_returns_nan():
    """top-3 履歴が無いとき recent_best_margin_in_top3 は NaN。"""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    base = date(2024, 6, 15)
    with Session(engine) as session:
        session.add(Horse(horse_id="HX", name=None))
        # finish=10 の race を 1 件だけ
        session.add(Race(
            race_id="RX1",
            date=(base - timedelta(days=10)).isoformat(),
            course="東京", surface="芝", distance=1600, n_runners=18,
        ))
        session.flush()
        session.add(Entry(
            race_id="RX1", horse_id="HX", post_position=1,
            finish_position=10, margin="5", finish_time=98.0, passing="15-12-10",
        ))
        session.commit()
        result = compute_horse_history(session, "HX", before_date=base)
    assert result["recent_best_margin_in_top3"] is not None
    assert math.isnan(result["recent_best_margin_in_top3"])
    # 他の Phase B feature はちゃんと値が入る
    assert result["recent_avg_margin"] == pytest.approx(5.0)
    engine.dispose()


def test_phase_b_partial_missing_data():
    """一部 race で margin/passing/finish_time が None のとき、
    その race だけスキップして残りで計算する。"""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    base = date(2024, 6, 15)
    with Session(engine) as session:
        session.add(Horse(horse_id="HM", name=None))
        races = [
            # (race_id, days, finish, margin, time, passing)
            ("RM1", 30, 3, "クビ", 95.0, "5-4-3"),    # 全部 OK
            ("RM2", 20, 7, None,   None, None),       # 全 None (winner じゃない)
            ("RM3", 10, 1, None,   90.0, "1-1"),      # winner: margin=None → 0
        ]
        for rid, d, _f, _m, _t, _p in races:
            session.add(Race(
                race_id=rid,
                date=(base - timedelta(days=d)).isoformat(),
                course="東京", surface="芝", distance=1600, n_runners=10,
            ))
        session.flush()
        for rid, _d, f, m, t, p in races:
            session.add(Entry(
                race_id=rid, horse_id="HM", post_position=1,
                finish_position=f, margin=m, finish_time=t, passing=p,
            ))
        session.commit()
        sql_r = compute_horse_history(session, "HM", before_date=base)
        cache = build_horse_history_cache(session)
        cache_r = compute_horse_history_from_cache(cache, "HM", before_date=base)

    # margin: RM1=0.15, RM2=None (skipped), RM3=0 (winner) → avg=(0.15+0)/2=0.075
    assert sql_r["recent_avg_margin"] == pytest.approx(0.075)
    # finish_time_norm: RM1=95/1600, RM2=skipped, RM3=90/1600 → avg
    assert sql_r["recent_avg_finish_time_norm"] == pytest.approx(
        (95.0 / 1600 + 90.0 / 1600) / 2
    )
    # parity
    for k in sql_r:
        sv, cv = sql_r[k], cache_r[k]
        if isinstance(sv, float) and math.isnan(sv):
            assert isinstance(cv, float) and math.isnan(cv), f"{k}: {sv!r} vs {cv!r}"
        else:
            assert sv == pytest.approx(cv), f"{k}: {sv!r} vs {cv!r}"
    engine.dispose()
