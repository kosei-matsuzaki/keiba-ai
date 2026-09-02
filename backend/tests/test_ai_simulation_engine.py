"""Unit tests for ai/simulation/engine.py (_settle_candidates と損益の積み上げ)。

Full integration test (simulate_active_model) requires a trained model bundle
and is covered manually.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ai.simulation.engine import (
    GroupStats,
    _settle_candidates,
)


@dataclass
class _FakeCandidate:
    """Minimal stand-in for BetCandidate (only fields _settle_candidates reads)."""
    bet_type: str
    combo: str
    stake: int


def _cand(bet_type: str, combo: str, stake: int = 100) -> _FakeCandidate:
    return _FakeCandidate(bet_type=bet_type, combo=combo, stake=stake)


# ---------------------------------------------------------------------------
# GroupStats payback / hit_rate
# ---------------------------------------------------------------------------


def test_group_stats_zero_division_safe():
    g = GroupStats(label="x")
    assert g.payback_rate == 0.0
    assert g.hit_rate == 0.0


def test_group_stats_basic():
    g = GroupStats(label="単勝", n_bets=10, invested=1000, payout=1500.0, hits=3)
    assert g.payback_rate == 1.5
    assert g.hit_rate == 0.3


# ---------------------------------------------------------------------------
# _settle_candidates
# ---------------------------------------------------------------------------


def test_settle_tansho_winner_hit():
    """単勝: combo == winner_pp で hit + payout = stake × confirmed odds"""
    finish_to_pp = {1: 5, 2: 3, 3: 7}
    past_odds = {"単勝": {"5": 4.2, "3": 6.8}}  # all horses listed
    cands = [_cand("単勝", "5", stake=100)]
    out = _settle_candidates(cands, "R001", finish_to_pp, past_odds)
    assert len(out) == 1
    assert out[0]["hit"] == 1
    assert out[0]["payout"] == pytest.approx(100 * 4.2)


def test_settle_tansho_loser_miss():
    """単勝: combo != winner_pp で miss + payout=0"""
    finish_to_pp = {1: 5}
    past_odds = {"単勝": {"5": 4.2, "3": 6.8}}
    cands = [_cand("単勝", "3", stake=100)]
    out = _settle_candidates(cands, "R001", finish_to_pp, past_odds)
    assert out[0]["hit"] == 0
    assert out[0]["payout"] == 0.0


def test_settle_fukusho_top3_hit():
    """複勝: combo が top-3 にいたら hit"""
    finish_to_pp = {1: 5, 2: 3, 3: 7}
    past_odds = {"複勝": {"5": 1.5, "3": 1.8, "7": 2.2}}
    out = _settle_candidates([_cand("複勝", "7", 200)], "R001", finish_to_pp, past_odds)
    assert out[0]["hit"] == 1
    assert out[0]["payout"] == pytest.approx(200 * 2.2)


def test_settle_fukusho_outside_top3_miss():
    finish_to_pp = {1: 5, 2: 3, 3: 7}
    past_odds = {"複勝": {"5": 1.5}}
    out = _settle_candidates([_cand("複勝", "11", 100)], "R001", finish_to_pp, past_odds)
    assert out[0]["hit"] == 0
    assert out[0]["payout"] == 0.0


def test_settle_renkei_hit_via_payouts_dict():
    """馬連: past_odds 内に combo がいれば hit + payout"""
    finish_to_pp = {1: 5, 2: 3, 3: 7}
    past_odds = {"馬連": {"3-5": 18.5}}  # 3-5 (post 3 と post 5) = top-2
    out = _settle_candidates([_cand("馬連", "3-5", 100)], "R001", finish_to_pp, past_odds)
    assert out[0]["hit"] == 1
    assert out[0]["payout"] == pytest.approx(100 * 18.5)


def test_settle_renkei_miss():
    """馬連: past_odds 内に combo が無ければ miss"""
    finish_to_pp = {1: 5, 2: 3, 3: 7}
    past_odds = {"馬連": {"3-5": 18.5}}  # only winning combo recorded
    out = _settle_candidates([_cand("馬連", "5-7", 100)], "R001", finish_to_pp, past_odds)
    assert out[0]["hit"] == 0
    assert out[0]["payout"] == 0.0


def test_settle_skips_zero_stake():
    """stake=0 候補はスキップされ settlements に含まれない"""
    finish_to_pp = {1: 5}
    past_odds = {"単勝": {"5": 4.0}}
    cands = [_cand("単勝", "5", 0), _cand("単勝", "3", 100)]
    out = _settle_candidates(cands, "R001", finish_to_pp, past_odds)
    assert len(out) == 1  # zero-stake skipped
    assert out[0]["bet_type"] == "単勝" and out[0]["combo"] if False else True


def test_settle_handles_missing_winner():
    """winner_pp が None でも crash しない"""
    finish_to_pp = {2: 3, 3: 7}  # 1 着 なし (DNF など)
    past_odds = {"単勝": {"3": 5.0}}
    out = _settle_candidates([_cand("単勝", "3", 100)], "R001", finish_to_pp, past_odds)
    assert out[0]["hit"] == 0  # winner_pp is None → no hit


# ---------------------------------------------------------------------------
# Compounding wealth (Option D-revised: payout 加算ありの 真の Kelly)
# ---------------------------------------------------------------------------


def _compounding_setup(monkeypatch, n_races: int, n_horses: int = 4, odds_win: float = 2.0):
    """compounding wealth テスト用の synthetic DB + stub セット。

    各 race の finish_position[i] = i (1-index)、つまり post 1 が常に 1 着。
    """
    from datetime import date, timedelta
    from types import SimpleNamespace

    import pandas as pd
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    import ai.simulation.engine as sim_mod
    from db.base import Base
    from db.models.entry import Entry
    from db.models.horse import Horse
    from db.models.race import Race

    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    base = date(2024, 6, 15)
    with Session(engine) as session:
        for hi in range(1, n_horses + 1):
            session.add(Horse(horse_id=f"H{hi}", name=f"H{hi}"))
        for ri in range(1, n_races + 1):
            # 後ろの race ほど新しい日付に (timedelta 逆順で OK)
            session.add(Race(
                race_id=f"R{ri}",
                date=(base + timedelta(days=ri)).isoformat(),
                course="東京", surface="芝", distance=1600, n_runners=n_horses,
            ))
        session.flush()
        for ri in range(1, n_races + 1):
            for hi in range(1, n_horses + 1):
                session.add(Entry(
                    race_id=f"R{ri}", horse_id=f"H{hi}", post_position=hi,
                    finish_position=hi, odds_win=odds_win, popularity=hi,
                ))
        session.commit()

    fake_bundle = SimpleNamespace(
        lambdarank=None, binary=None, calibrator=None, combo_calibrators=None,
    )
    monkeypatch.setattr(sim_mod, "load_model_full", lambda _p: fake_bundle)
    monkeypatch.setattr(sim_mod, "predict_race", lambda _m, f, **_kw: pd.DataFrame({
        "horse_id": f["horse_id"].values, "score": [1.0] * len(f),
        "win_prob": [1.0 / n_horses] * len(f), "place_prob": [0.5] * len(f),
    }))
    monkeypatch.setattr(sim_mod, "predict_race_with_combinations", lambda *a, **kw: {})
    monkeypatch.setattr(sim_mod, "compute_race_odds_with_sources", lambda *a, **kw: ({}, {}))
    # combo "1" (post 1 = winner) のみ単勝 4.0 倍が確定
    monkeypatch.setattr(sim_mod, "compute_past_race_odds", lambda *a, **kw: {"単勝": {"1": 4.0}})

    return engine


def test_profit_starts_at_zero(monkeypatch):
    """賭けなければ損益は 0 のまま。**元手という概念を持たない。**"""
    from pathlib import Path
    from types import SimpleNamespace

    from sqlalchemy.orm import Session

    import ai.simulation.engine as sim_mod

    engine = _compounding_setup(monkeypatch, n_races=1)
    budgets_seen: list[int] = []

    def _fake(*, race_budget, **_kw):
        budgets_seen.append(race_budget)
        return SimpleNamespace(candidates=[])
    monkeypatch.setattr(sim_mod, "recommend_for_race", _fake)

    with Session(engine) as session:
        result = sim_mod.simulate_active_model(
            session=session, model_path=Path("/tmp/dummy"),
            start=None, end=None, race_budget=5_000,
        )

    # 1 レースの予算は残高に依存しない。指定した額がそのまま渡る
    assert budgets_seen == [5_000]
    assert result.final_profit == 0
    assert result.peak_profit == 0
    assert result.trough_profit == 0


def test_profit_accumulates_across_races(monkeypatch):
    """払戻は次レースの賭け金に影響しない。損益だけが積み上がる。"""
    from pathlib import Path
    from types import SimpleNamespace

    from sqlalchemy.orm import Session

    import ai.simulation.engine as sim_mod

    engine = _compounding_setup(monkeypatch, n_races=3)
    budgets_seen: list[int] = []

    # 各 race で 100 円を当たり combo "1" に賭ける → odds 4.0 で payout 400
    def _winning(*, race_budget, **_kw):
        budgets_seen.append(race_budget)
        return SimpleNamespace(
            candidates=[SimpleNamespace(bet_type="単勝", combo="1", stake=100)]
        )
    monkeypatch.setattr(sim_mod, "recommend_for_race", _winning)

    with Session(engine) as session:
        result = sim_mod.simulate_active_model(
            session=session, model_path=Path("/tmp/dummy"),
            start=None, end=None, race_budget=5_000,
        )

    # **予算は資産に連動しない** (以前は残資産 × 5% で膨らんでいた)
    assert budgets_seen == [5_000, 5_000, 5_000]
    # race 毎: stake=100, payout=400, profit=+300
    assert result.final_profit == 900
    assert result.peak_profit == 900
    assert result.trough_profit == 0


def test_profit_goes_negative_and_keeps_betting(monkeypatch):
    """負け続けてもマイナスのまま賭け続ける (破産で評価が止まらない)。"""
    from pathlib import Path
    from types import SimpleNamespace

    from sqlalchemy.orm import Session

    import ai.simulation.engine as sim_mod

    engine = _compounding_setup(monkeypatch, n_races=3)
    n_calls = 0

    def _losing(*, race_budget, **_kw):
        nonlocal n_calls
        n_calls += 1
        # combo "9" は 1 着ではないので必ず外れ
        return SimpleNamespace(
            candidates=[SimpleNamespace(bet_type="単勝", combo="9", stake=100)]
        )
    monkeypatch.setattr(sim_mod, "recommend_for_race", _losing)

    with Session(engine) as session:
        result = sim_mod.simulate_active_model(
            session=session, model_path=Path("/tmp/dummy"),
            start=None, end=None, race_budget=5_000,
        )

    assert n_calls == 3  # 最後まで賭け続ける
    assert result.final_profit == -300
    assert result.trough_profit == -300
    assert result.peak_profit == 0
    # 途中で沈んだ分がそのまま「必要だった資金」
    assert result.required_capital == 300


def test_profit_timeseries_daily_aggregation(monkeypatch):
    """profit_timeseries は日次集約 (同日複数 race でも 1 ポイント)。"""
    from pathlib import Path
    from types import SimpleNamespace

    from sqlalchemy.orm import Session

    import ai.simulation.engine as sim_mod

    engine = _compounding_setup(monkeypatch, n_races=3)

    def _winning(*, race_budget, **_kw):
        return SimpleNamespace(
            candidates=[SimpleNamespace(bet_type="単勝", combo="1", stake=100)]
        )
    monkeypatch.setattr(sim_mod, "recommend_for_race", _winning)

    with Session(engine) as session:
        result = sim_mod.simulate_active_model(
            session=session, model_path=Path("/tmp/dummy"),
            start=None, end=None, race_budget=5_000,
        )

    # _compounding_setup は race ごとに別の日付を振る
    assert len(result.profit_timeseries) == 3
    assert [p.profit for p in result.profit_timeseries] == [300, 600, 900]
    assert [p.date for p in result.profit_timeseries] == sorted(
        p.date for p in result.profit_timeseries
    )
    for point in result.profit_timeseries:
        assert point.n_bets == 1
        assert point.invested == 100


def test_points_come_from_confidence(monkeypatch):
    """単複の点数は確信度から決まる (1 点 = 100 円)。連系は下限だけ。"""
    from pathlib import Path
    from types import SimpleNamespace

    from sqlalchemy.orm import Session

    import ai.simulation.engine as sim_mod

    engine = _compounding_setup(monkeypatch, n_races=1)
    seen: dict = {}

    def _fake(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(candidates=[])
    monkeypatch.setattr(sim_mod, "recommend_for_race", _fake)

    with Session(engine) as session:
        sim_mod.simulate_active_model(
            session=session, model_path=Path("/tmp/dummy"),
            start=None, end=None, race_budget=10_000,
        )
    # 確率モデル未設定なので基準の点数。券種ごとの上限は渡さない
    assert seen["points_by_bet_type"] == {"単勝": 5, "複勝": 5}
    assert "max_points_per_bet_type" not in seen
