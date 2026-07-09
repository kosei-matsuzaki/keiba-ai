"""Tests for features.speed_figure (par-time + track-variant speed figure)."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from sqlalchemy import create_engine

from db.base import Base
from db.models.entry import Entry
from db.models.horse import Horse
from db.models.race import Race
from db.session import session_scope
from features.speed_figure import (
    SpeedFigureModel,
    add_speed_figure_column,
    build_speed_figure_model,
)


def _add_race(session, race_id, date, course, distance, surface, times):
    """1 レース + 着順つき entries。times[0] が 1 着 (勝ち馬 finish_time)。"""
    session.add(Race(
        race_id=race_id, date=date, course=course, surface=surface,
        distance=distance, n_runners=len(times),
    ))
    for i, t in enumerate(times):
        hid = f"H{race_id}_{i}"
        if not session.get(Horse, hid):
            session.add(Horse(horse_id=hid, name=None))
        session.add(Entry(
            race_id=race_id, horse_id=hid, post_position=i + 1,
            finish_position=i + 1, finish_time=t,
        ))


@pytest.fixture()
def eng():
    e = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(e)
    with session_scope(e) as s:
        # train: 東京芝1600 winner times 95, 97, 99 -> median 97 (par)
        _add_race(s, "TR1", "2024-01-06", "東京", 1600, "芝", [95.0, 96, 97, 98])
        _add_race(s, "TR2", "2024-02-03", "東京", 1600, "芝", [97.0, 98, 99, 100])
        _add_race(s, "TR3", "2024-03-02", "東京", 1600, "芝", [99.0, 100, 101, 102])
        # test-period race on a SLOW day (winner 105 vs par 97 -> variant +8)
        _add_race(s, "TE1", "2025-05-03", "東京", 1600, "芝", [105.0, 106, 107])
    yield e
    e.dispose()


def test_par_is_median_winner_time_train_only(eng):
    with session_scope(eng) as s:
        train_ids = {"TR1", "TR2", "TR3"}  # TE1 excluded
        model = build_speed_figure_model(s, train_ids)
    # median of winner times 95, 97, 99 = 97
    assert model.par[("東京", 1600, "芝")] == pytest.approx(97.0)
    # global / surf_dist fallbacks present
    assert model.par_global == pytest.approx(97.0)
    assert model.par_by_surf_dist[("芝", 1600)] == pytest.approx(97.0)


def test_par_excludes_test_period_no_leak(eng):
    """test レース (TE1, winner 105) を train に含めなければ par に影響しない。"""
    with session_scope(eng) as s:
        with_test = build_speed_figure_model(s, {"TR1", "TR2", "TR3", "TE1"})
        without_test = build_speed_figure_model(s, {"TR1", "TR2", "TR3"})
    # par shifts when TE1 leaks into the fit, stays 97 when excluded
    assert without_test.par[("東京", 1600, "芝")] == pytest.approx(97.0)
    assert with_test.par[("東京", 1600, "芝")] != pytest.approx(97.0)


def test_track_variant_captures_slow_day(eng):
    with session_scope(eng) as s:
        model = build_speed_figure_model(s, {"TR1", "TR2", "TR3"})
    # TE1 day: winner 105 vs par 97 -> variant ~ +8 (slow track)
    assert model.variants[("東京", "2025-05-03")] == pytest.approx(8.0, abs=1e-6)
    # a par-matching train day has variant ~0 (TR2 winner 97 == par)
    assert model.variants[("東京", "2024-02-03")] == pytest.approx(0.0, abs=1e-6)


def test_speed_figure_sign_and_variant_adjustment(eng):
    with session_scope(eng) as s:
        model = build_speed_figure_model(s, {"TR1", "TR2", "TR3"})
    # A horse running 97s on a normal day (variant 0) == par -> speed_fig ~ 0
    df = pd.DataFrame({
        "course": ["東京", "東京", "東京"],
        "distance": [1600, 1600, 1600],
        "surface": ["芝", "芝", "芝"],
        "date": ["2024-02-03", "2024-02-03", "2025-05-03"],
        "finish_time": [97.0, 91.0, 105.0],
    })
    sp = add_speed_figure_column(df, model)
    # 97 on par day -> ~0
    assert sp[0] == pytest.approx(0.0, abs=1e-6)
    # faster (91 < 97) -> positive (good)
    assert sp[1] > 0
    # 105 on slow day (variant +8): par+variant = 105 -> speed_fig ~ 0 (not penalised)
    assert sp[2] == pytest.approx(0.0, abs=1e-6)


def test_speed_figure_nan_when_par_missing():
    """par テーブルに無い組 + global も無いとき NaN。"""
    model = SpeedFigureModel()  # empty, par_global = nan
    df = pd.DataFrame({
        "course": ["札幌"], "distance": [2000], "surface": ["ダ"],
        "date": ["2024-01-01"], "finish_time": [120.0],
    })
    sp = add_speed_figure_column(df, model)
    assert math.isnan(sp[0])


def test_speed_figure_nan_when_finish_time_missing(eng):
    with session_scope(eng) as s:
        model = build_speed_figure_model(s, {"TR1", "TR2", "TR3"})
    df = pd.DataFrame({
        "course": ["東京"], "distance": [1600], "surface": ["芝"],
        "date": ["2024-02-03"], "finish_time": [None],
    })
    sp = add_speed_figure_column(df, model)
    assert math.isnan(sp[0])


def test_empty_train_falls_back_to_all(eng):
    """train_race_ids が DB と交わらないとき全体で fit して退行を避ける。"""
    with session_scope(eng) as s:
        model = build_speed_figure_model(s, {"NONEXISTENT"})
    # median of all winners 95, 97, 99, 105 = 98
    assert model.par_global == pytest.approx(98.0)
