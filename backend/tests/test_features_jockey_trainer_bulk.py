"""Parity tests for jockey / trainer bulk preload caches.

Verifies build_*_history_cache + compute_*_stats_from_cache produce
bit-for-bit identical output (NaN included) to the SQL-per-call versions.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

import db.models  # noqa: F401
from db.base import Base
from db.models.entry import Entry
from db.models.horse import Horse
from db.models.jockey import Jockey
from db.models.race import Race
from db.models.trainer import Trainer
from features.extractors.jockey import (
    build_jockey_history_cache,
    compute_jockey_stats,
    compute_jockey_stats_from_cache,
)
from features.extractors.trainer import (
    build_trainer_history_cache,
    compute_trainer_stats,
    compute_trainer_stats_from_cache,
)


def _assert_dicts_equal(a: dict, b: dict, *, ctx: str = "") -> None:
    assert a.keys() == b.keys(), f"keys differ {ctx}"
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, float) and math.isnan(va):
            assert isinstance(vb, float) and math.isnan(vb), f"{k} {ctx}: {va!r} vs {vb!r}"
        else:
            assert va == vb, f"{k} {ctx}: {va!r} vs {vb!r}"


@pytest.fixture()
def engine_with_history():
    """DB with 1 jockey, 1 trainer, multiple races spread over time."""
    eng = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(eng)

    base = date(2024, 6, 15)
    races = [
        # (race_id, days_ago, course, finish_position)
        ("R1", 60, "東京", 1),
        ("R2", 45, "中山", 3),
        ("R3", 25, "東京", 2),
        ("R4", 20, "東京", 4),
        ("R5", 10, "中山", 1),  # within 30-day recent window
        ("R6", 5,  "東京", 5),  # within 30-day recent window
    ]

    with Session(eng) as session:
        session.add(Horse(horse_id="H1", name="horse1"))
        session.add(Jockey(jockey_id="J1", name="jockey1"))
        session.add(Trainer(trainer_id="T1", name="trainer1"))
        for rid, days, course, _pos in races:
            session.add(Race(
                race_id=rid,
                date=(base - timedelta(days=days)).isoformat(),
                course=course,
                surface="芝",
                distance=1600,
                n_runners=10,
            ))
        session.flush()
        for rid, _days, _course, pos in races:
            session.add(Entry(
                race_id=rid,
                horse_id="H1",
                jockey_id="J1",
                trainer_id="T1",
                post_position=1,
                finish_position=pos,
            ))
        session.commit()

    yield eng
    eng.dispose()


# ── Jockey ────────────────────────────────────────────────────────────────────


def test_jockey_cache_parity_no_course(engine_with_history):
    base = date(2024, 6, 15)
    with Session(engine_with_history) as session:
        cache = build_jockey_history_cache(session)
        sql_r = compute_jockey_stats(session, "J1", before_date=base)
        cache_r = compute_jockey_stats_from_cache(cache, "J1", before_date=base)
    _assert_dicts_equal(sql_r, cache_r, ctx="(no course)")


def test_jockey_cache_parity_with_course(engine_with_history):
    base = date(2024, 6, 15)
    with Session(engine_with_history) as session:
        cache = build_jockey_history_cache(session)
        for course in [None, "東京", "中山"]:
            sql_r = compute_jockey_stats(session, "J1", before_date=base, course=course)
            cache_r = compute_jockey_stats_from_cache(cache, "J1", before_date=base, course=course)
            _assert_dicts_equal(sql_r, cache_r, ctx=f"(course={course})")


def test_jockey_cache_parity_unknown(engine_with_history):
    base = date(2024, 6, 15)
    with Session(engine_with_history) as session:
        cache = build_jockey_history_cache(session)
        sql_r = compute_jockey_stats(session, "UNKNOWN", before_date=base, course="東京")
        cache_r = compute_jockey_stats_from_cache(cache, "UNKNOWN", before_date=base, course="東京")
    _assert_dicts_equal(sql_r, cache_r, ctx="(unknown)")


def test_jockey_cache_window_boundary(engine_with_history):
    """30-day recent window cutoff inclusion / exclusion is identical."""
    base = date(2024, 6, 15)
    # before_date が past の方にズレてるシナリオも試す
    for delta in [0, -5, 5, 100]:
        target = base + timedelta(days=delta)
        with Session(engine_with_history) as session:
            cache = build_jockey_history_cache(session)
            sql_r = compute_jockey_stats(session, "J1", before_date=target, course="東京")
            cache_r = compute_jockey_stats_from_cache(cache, "J1", before_date=target, course="東京")
        _assert_dicts_equal(sql_r, cache_r, ctx=f"(delta={delta})")


# ── Trainer ───────────────────────────────────────────────────────────────────


def test_trainer_cache_parity(engine_with_history):
    base = date(2024, 6, 15)
    with Session(engine_with_history) as session:
        cache = build_trainer_history_cache(session)
        for course in [None, "東京", "中山"]:
            sql_r = compute_trainer_stats(session, "T1", before_date=base, course=course)
            cache_r = compute_trainer_stats_from_cache(cache, "T1", before_date=base, course=course)
            _assert_dicts_equal(sql_r, cache_r, ctx=f"(course={course})")


def test_trainer_cache_parity_unknown(engine_with_history):
    with Session(engine_with_history) as session:
        cache = build_trainer_history_cache(session)
        sql_r = compute_trainer_stats(session, "UNKNOWN", before_date=date(2024, 6, 15))
        cache_r = compute_trainer_stats_from_cache(cache, "UNKNOWN", before_date=date(2024, 6, 15))
    _assert_dicts_equal(sql_r, cache_r, ctx="(unknown trainer)")


# ── Single-SQL guarantees ─────────────────────────────────────────────────────


def test_jockey_cache_single_sql_query(engine_with_history):
    queries: list[str] = []

    def _capture(conn, cursor, statement, params, context, executemany):  # noqa: ARG001
        queries.append(statement)

    event.listen(engine_with_history, "before_cursor_execute", _capture)
    try:
        with Session(engine_with_history) as session:
            build_jockey_history_cache(session)
        select_count = sum(1 for q in queries if q.strip().lower().startswith("select"))
        assert select_count == 1, f"expected 1 SELECT, got {select_count}"
    finally:
        event.remove(engine_with_history, "before_cursor_execute", _capture)


def test_trainer_cache_single_sql_query(engine_with_history):
    queries: list[str] = []

    def _capture(conn, cursor, statement, params, context, executemany):  # noqa: ARG001
        queries.append(statement)

    event.listen(engine_with_history, "before_cursor_execute", _capture)
    try:
        with Session(engine_with_history) as session:
            build_trainer_history_cache(session)
        select_count = sum(1 for q in queries if q.strip().lower().startswith("select"))
        assert select_count == 1
    finally:
        event.remove(engine_with_history, "before_cursor_execute", _capture)
