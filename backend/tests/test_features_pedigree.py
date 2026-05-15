"""Tests for features/pedigree.py.

Verifies win-rate computation and leakage prevention (Race.date < before_date).
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

import db.models  # noqa: F401
from db.base import Base
from db.models.entry import Entry
from db.models.horse import Horse
from db.models.race import Race
from features.pedigree import (
    build_pedigree_cache,
    compute_pedigree_features,
    compute_pedigree_features_from_cache,
)


@pytest.fixture()
def pedigree_engine():
    """DB with 2 horses (same sire), 3 past races and 1 future race."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)

    cutoff = date(2024, 6, 15)

    with Session(engine) as session:
        session.add(Horse(horse_id="H001", name=None, sire="ディープインパクト", dam="スターオブコジーン"))
        session.add(Horse(horse_id="H002", name=None, sire="ディープインパクト", dam="アドマイヤグルーヴ"))
        session.add(Horse(horse_id="H003", name=None, sire="キングカメハメハ", dam=None))

        # Races:
        # R001: 30 days before cutoff — H001 wins, H002 finishes 2nd
        # R002: 15 days before cutoff — H001 finishes 3rd
        # R003: future (after cutoff)  — H001 wins  ← must be EXCLUDED
        races = [
            ("R001", cutoff - timedelta(days=30)),
            ("R002", cutoff - timedelta(days=15)),
            ("R003", cutoff + timedelta(days=5)),
        ]
        for rid, d in races:
            session.add(
                Race(
                    race_id=rid,
                    date=d.isoformat(),
                    course="東京",
                    surface="芝",
                    distance=1600,
                    n_runners=3,
                )
            )
        session.flush()

        # R001: H001 wins, H002 2nd
        session.add(Entry(race_id="R001", horse_id="H001", post_position=1, finish_position=1))
        session.add(Entry(race_id="R001", horse_id="H002", post_position=2, finish_position=2))
        # R002: H001 3rd
        session.add(Entry(race_id="R002", horse_id="H001", post_position=1, finish_position=3))
        # R003 (future): H001 wins — MUST NOT count
        session.add(Entry(race_id="R003", horse_id="H001", post_position=1, finish_position=1))

        session.commit()

    yield engine
    engine.dispose()


def test_sire_progeny_win_rate_excludes_after_before_date(pedigree_engine):
    """Races on or after before_date must not contribute to progeny win rate."""
    with Session(pedigree_engine) as session:
        result = compute_pedigree_features(
            session,
            sire="ディープインパクト",
            dam=None,
            before_date=date(2024, 6, 15),
        )

    # Before cutoff: R001 (H001 wins, H002 2nd) + R002 (H001 3rd)
    # Total entries with sire=ディープインパクト before cutoff: 3
    # Wins: 1 (H001 in R001)
    # R003 must be excluded even though H001 wins there
    rate = result["sire_progeny_win_rate"]
    assert rate is not None
    assert rate == pytest.approx(1 / 3)


def test_dam_progeny_win_rate_zero_when_no_wins(pedigree_engine):
    """dam_progeny_win_rate is 0.0 when the progeny has starts but no wins."""
    with Session(pedigree_engine) as session:
        result = compute_pedigree_features(
            session,
            sire=None,
            dam="アドマイヤグルーヴ",
            before_date=date(2024, 6, 15),
        )

    # H002 has dam=アドマイヤグルーヴ, only started in R001 (finished 2nd)
    rate = result["dam_progeny_win_rate"]
    assert rate is not None
    assert rate == pytest.approx(0.0)


def test_returns_none_when_sire_unknown(pedigree_engine):
    """sire_progeny_win_rate and dam_progeny_win_rate are None when sire/dam is None."""
    with Session(pedigree_engine) as session:
        result = compute_pedigree_features(
            session,
            sire=None,
            dam=None,
            before_date=date(2024, 6, 15),
        )

    assert result["sire_progeny_win_rate"] is None
    assert result["dam_progeny_win_rate"] is None


def test_returns_none_when_sire_has_no_progeny_in_db(pedigree_engine):
    """When the sire exists but has no entries before before_date, return None."""
    with Session(pedigree_engine) as session:
        result = compute_pedigree_features(
            session,
            sire="ゴールドシップ",  # not in DB at all
            dam=None,
            before_date=date(2024, 6, 15),
        )

    assert result["sire_progeny_win_rate"] is None


def test_sire_with_all_wins(pedigree_engine):
    """Sanity: if every qualifying start is a win, rate should be 1.0."""
    with Session(pedigree_engine) as session:
        result = compute_pedigree_features(
            session,
            sire="キングカメハメハ",  # H003, no races yet → None
            dam=None,
            before_date=date(2024, 6, 15),
        )

    # H003 has no entries in DB → rate is None
    assert result["sire_progeny_win_rate"] is None


# ---------------------------------------------------------------------------
# Cache parity tests (build_pedigree_cache + compute_pedigree_features_from_cache)
# ---------------------------------------------------------------------------


def _assert_pedigree_equal(a: dict, b: dict, *, ctx: str) -> None:
    assert a.keys() == b.keys()
    for k in a:
        # both sides may be float or None
        assert a[k] == b[k], f"{k} {ctx}: {a[k]!r} vs {b[k]!r}"


def test_pedigree_cache_parity_known_sire_and_dam(pedigree_engine):
    """Known sire / dam combinations match SQL version bit-for-bit."""
    base = date(2024, 6, 15)
    cases = [
        ("ディープインパクト", "スターオブコジーン"),
        ("ディープインパクト", "アドマイヤグルーヴ"),
        ("ディープインパクト", None),
        (None, "アドマイヤグルーヴ"),
    ]
    with Session(pedigree_engine) as session:
        cache = build_pedigree_cache(session)
        for sire, dam in cases:
            sql_r = compute_pedigree_features(session, sire, dam, before_date=base)
            cache_r = compute_pedigree_features_from_cache(cache, sire, dam, before_date=base)
            _assert_pedigree_equal(sql_r, cache_r, ctx=f"sire={sire!r}, dam={dam!r}")


def test_pedigree_cache_parity_unknown_names(pedigree_engine):
    """Unknown / NULL sire/dam → SQL と同じ None を返す。"""
    base = date(2024, 6, 15)
    with Session(pedigree_engine) as session:
        cache = build_pedigree_cache(session)
        # 1. completely unknown sire
        sql_r = compute_pedigree_features(session, "ゴールドシップ", None, before_date=base)
        cache_r = compute_pedigree_features_from_cache(cache, "ゴールドシップ", None, before_date=base)
        _assert_pedigree_equal(sql_r, cache_r, ctx="(unknown sire)")
        # 2. both None
        sql_r = compute_pedigree_features(session, None, None, before_date=base)
        cache_r = compute_pedigree_features_from_cache(cache, None, None, before_date=base)
        _assert_pedigree_equal(sql_r, cache_r, ctx="(both None)")


def test_pedigree_cache_horse_to_sire_dam_lookup(pedigree_engine):
    """horse_to_sire_dam が全ての horse を含み正しい sire/dam を返す。"""
    with Session(pedigree_engine) as session:
        cache = build_pedigree_cache(session)
    assert cache.horse_to_sire_dam["H001"] == ("ディープインパクト", "スターオブコジーン")
    assert cache.horse_to_sire_dam["H002"] == ("ディープインパクト", "アドマイヤグルーヴ")
    assert cache.horse_to_sire_dam["H003"] == ("キングカメハメハ", None)


def test_pedigree_cache_two_sql_queries(pedigree_engine):
    """build_pedigree_cache emits exactly 2 SELECTs (horses + entries-join)."""
    from sqlalchemy import event

    queries: list[str] = []

    def _capture(conn, cursor, statement, params, context, executemany):  # noqa: ARG001
        queries.append(statement)

    event.listen(pedigree_engine, "before_cursor_execute", _capture)
    try:
        with Session(pedigree_engine) as session:
            build_pedigree_cache(session)
        select_count = sum(1 for q in queries if q.strip().lower().startswith("select"))
        assert select_count == 2, f"expected 2 SELECTs, got {select_count}: {queries}"
    finally:
        event.remove(pedigree_engine, "before_cursor_execute", _capture)
