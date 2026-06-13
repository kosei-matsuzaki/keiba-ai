"""Tests for features/history_sequence.py — leak-safe per-past-race token sequences."""

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
from features.history_sequence import (
    TOKEN_FEATURE_NAMES,
    H,
    _margin_num,
    _passing_first,
    build_history_sequences,
    fit_history_normalizer,
)


def test_margin_passing_handle_nan_float():
    """DB NULL は pandas で float NaN になりうる → strip/split で落ちないこと。"""
    for bad in (float("nan"), None, 3.5):
        assert math.isnan(_margin_num(bad))
        assert math.isnan(_passing_first(bad))
    assert _margin_num("クビ") == 0.2
    assert _passing_first("3-3-2-1") == 3.0

FIELD = 4  # horses per race


@pytest.fixture()
def seq_engine():
    """1 target horse (HC) with 4 chronological races + filler horses per race."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    base = date(2024, 6, 15)
    # (race_id, days_ago, HC finish, surface, distance, agari)
    races = [
        ("S1", 40, 1, "芝", 1600, 34.0),
        ("S2", 30, 2, "ダ", 1200, 35.0),
        ("S3", 20, 3, "芝", 2000, 36.0),
        ("S4", 10, 1, "芝", 1600, 34.5),  # target race for history of HC
    ]
    with Session(engine) as session:
        session.add(Horse(horse_id="HC", name=None))
        for k in range(FIELD - 1):
            session.add(Horse(horse_id=f"F{k}", name=None))
        for rid, days, _hcfin, surf, dist, _ag in races:
            session.add(Race(
                race_id=rid, date=(base - timedelta(days=days)).isoformat(),
                course="東京", surface=surf, track_condition="良",
                distance=dist, n_runners=FIELD,
            ))
        session.flush()
        for rid, _days, hcfin, _surf, _dist, ag in races:
            session.add(Entry(
                race_id=rid, horse_id="HC", post_position=1,
                finish_position=hcfin, agari_3f=ag, finish_time=95.0,
            ))
            # filler horses fill the remaining finish positions
            others = [p for p in range(1, FIELD + 1) if p != hcfin][: FIELD - 1]
            for k, pos in enumerate(others):
                session.add(Entry(
                    race_id=rid, horse_id=f"F{k}", post_position=k + 2,
                    finish_position=pos, agari_3f=ag + 1.0, finish_time=96.0,
                ))
        session.commit()
    yield engine
    engine.dispose()


def _idx(name: str) -> int:
    return TOKEN_FEATURE_NAMES.index(name)


def test_token_feature_dim():
    assert H == len(TOKEN_FEATURE_NAMES) == 16


def test_leak_safe_history_length(seq_engine):
    """S4 の HC 履歴は厳密に過去 (S1,S2,S3) の 3 走のみ。"""
    with Session(seq_engine) as session:
        cache = build_history_sequences(session)
    seq = cache.seqs[("S4", "HC")]
    assert seq.shape == (3, H)  # S1,S2,S3 のみ (S4 自身・未来は含まない)


def test_debut_race_has_no_key(seq_engine):
    """最初のレース (S1) は過去走なし → キーを持たない。"""
    with Session(seq_engine) as session:
        cache = build_history_sequences(session)
    assert ("S1", "HC") not in cache.seqs
    # S2 は S1 の 1 走だけを履歴に持つ
    assert cache.seqs[("S2", "HC")].shape == (1, H)


def test_token_values(seq_engine):
    """finish_norm / won / surface one-hot / race_avg_agari がトークンに正しく入る。"""
    with Session(seq_engine) as session:
        cache = build_history_sequences(session)
    seq = cache.seqs[("S4", "HC")]  # rows = S1, S2, S3
    # S1: finish=1, field=4 → finish_norm=0.25, won=1, surface=芝
    assert seq[0, _idx("finish_norm")] == pytest.approx(0.25)
    assert seq[0, _idx("won")] == pytest.approx(1.0)
    assert seq[0, _idx("surface_芝")] == pytest.approx(1.0)
    assert seq[0, _idx("surface_ダ")] == pytest.approx(0.0)
    # S2: finish=2 → won=0, surface=ダ
    assert seq[1, _idx("won")] == pytest.approx(0.0)
    assert seq[1, _idx("surface_ダ")] == pytest.approx(1.0)
    # race_avg_agari_3f for S1: HC=34.0, fillers=35.0×3 → (34+35*3)/4 = 34.75
    assert seq[0, _idx("race_avg_agari_3f")] == pytest.approx((34.0 + 35.0 * 3) / 4)


def test_max_len_truncation(seq_engine):
    """max_len=2 のとき S4 の履歴は直近 2 走 (S2,S3) に切られる。"""
    with Session(seq_engine) as session:
        cache = build_history_sequences(session, max_len=2)
    seq = cache.seqs[("S4", "HC")]
    assert seq.shape == (2, H)
    # 直近2走 = S2(ダ), S3(芝) → 先頭は S2 (surface=ダ)
    assert seq[0, _idx("surface_ダ")] == pytest.approx(1.0)


def test_normalizer_shape_and_train_only(seq_engine):
    with Session(seq_engine) as session:
        cache = build_history_sequences(session)
    mean, std = fit_history_normalizer(cache, train_race_ids={"S4", "S3", "S2"})
    assert mean.shape == (H,)
    assert std.shape == (H,)
    assert (std > 0).all()
