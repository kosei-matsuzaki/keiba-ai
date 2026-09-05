"""シミュレーションの実行条件を結果に残す。

シミュレーションは Settings の値（確率モデル・確信度のしきい値・券種・1 点あたり
の金額）を**実行時に読む**ので、設定を変えれば同じ画面の同じボタンでも別の条件で
走る。条件を残さないと、保存済みの run どうしを比べられない。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import ai.simulation.engine as sim_mod


@pytest.fixture
def stub_engine(monkeypatch):
    """1 レースだけの合成 DB と、推論まわりの stub。"""
    from datetime import date

    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from db.base import Base
    from db.models.entry import Entry
    from db.models.horse import Horse
    from db.models.race import Race

    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        for i in range(1, 5):
            session.add(Horse(horse_id=f"H{i}", name=f"H{i}"))
        session.add(Race(race_id="R1", date=date(2024, 6, 15).isoformat(),
                         course="東京", surface="芝", distance=1600, n_runners=4))
        session.flush()
        for i in range(1, 5):
            session.add(Entry(race_id="R1", horse_id=f"H{i}", post_position=i,
                              finish_position=i, odds_win=3.0, popularity=i))
        session.commit()

    monkeypatch.setattr(sim_mod, "load_model_full", lambda _p: SimpleNamespace())
    monkeypatch.setattr(sim_mod, "predict_race", lambda _m, f, **_kw: pd.DataFrame({
        "horse_id": f["horse_id"].values, "score": [1.0] * len(f),
        "win_prob": [0.25] * len(f), "place_prob": [0.5] * len(f),
    }))
    monkeypatch.setattr(sim_mod, "predict_race_with_combinations", lambda *a, **kw: {})
    monkeypatch.setattr(sim_mod, "compute_race_odds_with_sources", lambda *a, **kw: ({}, {}))
    monkeypatch.setattr(sim_mod, "compute_past_race_odds", lambda *a, **kw: {})
    monkeypatch.setattr(sim_mod, "recommend_for_race",
                        lambda **kw: SimpleNamespace(candidates=[]))
    return engine


def _run(engine, **kwargs):
    from sqlalchemy.orm import Session
    with Session(engine) as session:
        return sim_mod.simulate_active_model(
            session=session, model_path=Path("/tmp/dummy"),
            start=None, end=None, race_budget=5_000, **kwargs,
        )


class TestConditionsAreRecorded:
    def test_probability_model_is_recorded_by_name(self, stub_engine):
        """パスではなくディレクトリ名で残す (環境で表記が違うため)。"""
        r = _run(stub_engine, probability_model_path=Path("/a/b/models/20260101T000000-nn"),
                 place_min_confidence=0.4)
        assert r.conditions["probability_model"] == "20260101T000000-nn"
        assert r.conditions["place_min_confidence"] == 0.4

    def test_unused_probability_model_records_none(self, stub_engine):
        """確率モデル未使用なら、しきい値も None にする。

        しきい値だけ残すと「0.30 で絞ったのか、絞っていないのか」が判別できない。
        """
        r = _run(stub_engine)
        assert r.conditions["probability_model"] is None
        assert r.conditions["place_min_confidence"] is None

    def test_records_the_options_that_change_the_result(self, stub_engine):
        r = _run(stub_engine, enabled_bet_types=["単勝", "複勝"],
                 min_hit_prob_by_bet_type={"馬連": 0.05})
        c = r.conditions
        assert c["enabled_bet_types"] == ["単勝", "複勝"]
        assert c["race_budget"] == 5_000
        # 連系の点数は下限だけで決まる (券種ごとの上限は持たない)
        assert "max_points_per_bet_type" not in c
        assert c["combo_min_hit_prob"] == {"馬連": 0.05}

    def test_records_win_min_odds_and_top_k(self, stub_engine):
        """RACE 画面と揃えた 2 つも残す。

        どちらも「同じ設定で回し直したのか」の判別に要る。渡し忘れると既定で
        走るが、結果からは見分けられなかった (win_min_odds が実際にそうなって
        いて、Settings を変えてもシミュレーションだけ追従していなかった)。
        """
        r = _run(stub_engine, win_min_odds=2.5, top_k_combinations=20)
        assert r.conditions["win_min_odds"] == 2.5
        assert r.conditions["top_k_combinations"] == 20

    def test_conditions_survive_as_dict(self, stub_engine):
        """API / 永続化は as_dict 経由なので、そこに載ることを固定する。"""
        r = _run(stub_engine)
        assert "conditions" in r.as_dict()
        json.dumps(r.as_dict()["conditions"])   # JSON 化できること
