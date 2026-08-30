"""レース内相対の走破指標。

既存の recent_avg_agari_3f / recent_avg_finish_time_norm は**生の秒数**で、
「その日の時計水準」が値に混ざる。重馬場の 34.0 と良馬場の 34.0 が同じ値になり、
条件の違うレースをまたいで比較できない。同じレースを走った馬どうしは条件を
共有するので、レース内平均からの差を取ると水準が落ちる。
"""

from __future__ import annotations

import os
from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db.base import Base
from db.models.entry import Entry
from db.models.horse import Horse
from db.models.race import Race
from features.builder import RELATIVE_FORM_COLS, get_active_features
from features.extractors.horse_history import (
    build_horse_history_cache,
    compute_horse_history,
    compute_horse_history_from_cache,
)


@pytest.fixture
def two_races():
    """同じ馬が「速い決着」と「遅い決着」を 1 回ずつ走った DB。

    生の上がりは同じ 34.0 でも、レース内の位置づけは正反対になる。
    """
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        for i in range(1, 4):
            s.add(Horse(horse_id=f"H{i}", name=f"H{i}"))
        # 速い決着: 他馬が 33.0 → 34.0 は「最も遅い」
        s.add(Race(race_id="FAST", date="2024-01-06", course="東京", surface="芝",
                   distance=1600, n_runners=3))
        # 遅い決着: 他馬が 35.0 → 34.0 は「最も速い」
        s.add(Race(race_id="SLOW", date="2024-02-03", course="東京", surface="芝",
                   distance=1600, n_runners=3))
        s.flush()
        for rid, others in (("FAST", 33.0), ("SLOW", 35.0)):
            s.add(Entry(race_id=rid, horse_id="H1", post_position=1, finish_position=2,
                        agari_3f=34.0, finish_time=95.0))
            for j, hid in enumerate(("H2", "H3"), start=2):
                s.add(Entry(race_id=rid, horse_id=hid, post_position=j,
                            finish_position=j, agari_3f=others,
                            finish_time=95.0 + (others - 34.0)))
        s.commit()
    return engine


class TestRelativeForm:
    def test_raw_agari_cannot_tell_the_two_races_apart(self, two_races):
        """生の上がりは 2 走とも 34.0 で、区別できない（これが動機）。"""
        with Session(two_races) as s:
            cache = build_horse_history_cache(s)
            feats = compute_horse_history_from_cache(
                cache, "H1", before_date=date(2024, 3, 1), distance=1600, course="東京"
            )
        assert feats["recent_avg_agari_3f"] == pytest.approx(34.0)

    def test_relative_agari_separates_them(self, two_races):
        """相対にすると +1.0 と −1.0 になり、平均 0.0 に戻る。"""
        with Session(two_races) as s:
            cache = build_horse_history_cache(s)
            feats = compute_horse_history_from_cache(
                cache, "H1", before_date=date(2024, 3, 1), distance=1600, course="東京"
            )
        # FAST: 34.0 - mean(34,33,33)=33.33 → +0.67 / SLOW: 34.0 - 34.67 → -0.67
        assert feats["recent_avg_agari_rel"] == pytest.approx(0.0, abs=1e-9)

    def test_rank_pct_reflects_position_in_the_field(self, two_races):
        """FAST では最下位 (1.0)、SLOW では最上位 (1/3) → 平均 2/3。"""
        with Session(two_races) as s:
            cache = build_horse_history_cache(s)
            feats = compute_horse_history_from_cache(
                cache, "H1", before_date=date(2024, 3, 1), distance=1600, course="東京"
            )
        assert feats["recent_avg_agari_rank_pct"] == pytest.approx(2 / 3, abs=1e-6)

    def test_sql_and_cache_paths_agree(self, two_races):
        """**学習は cache 経路、推論は SQL 経路**を通るので、値が一致しないと
        学習時と本番でモデルへの入力が変わる（履歴 GRU で踏んだ型の不具合）。"""
        with Session(two_races) as s:
            cache = build_horse_history_cache(s)
            a = compute_horse_history_from_cache(
                cache, "H1", before_date=date(2024, 3, 1), distance=1600, course="東京"
            )
            b = compute_horse_history(
                s, "H1", before_date=date(2024, 3, 1), distance=1600, course="東京"
            )
        for key in RELATIVE_FORM_COLS:
            assert a[key] == pytest.approx(b[key], abs=1e-9), key


class TestFeatureFlag:
    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv("KEIBA_RELATIVE_FORM", raising=False)
        assert not set(RELATIVE_FORM_COLS) & set(get_active_features())

    def test_on_when_set(self, monkeypatch):
        monkeypatch.setenv("KEIBA_RELATIVE_FORM", "1")
        assert set(RELATIVE_FORM_COLS) <= set(get_active_features())


def test_env_isolation():
    """他テストへ漏らさない。"""
    os.environ.pop("KEIBA_RELATIVE_FORM", None)
