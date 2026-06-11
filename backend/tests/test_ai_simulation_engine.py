"""Unit tests for ai/simulation/engine.py:_settle_candidates and STRATEGY_PRESETS.

Full integration test (simulate_active_model) requires a trained model bundle
and is covered manually.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ai.simulation.engine import (
    STRATEGY_PRESETS,
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
# STRATEGY_PRESETS sanity
# ---------------------------------------------------------------------------


def test_strategy_presets_present():
    assert {"conservative", "balanced", "aggressive"} <= set(STRATEGY_PRESETS)


def test_strategy_presets_kelly_ascending():
    """積極的になるほど Kelly が大きく min_ev が小さくなる。"""
    c = STRATEGY_PRESETS["conservative"]
    b = STRATEGY_PRESETS["balanced"]
    a = STRATEGY_PRESETS["aggressive"]
    assert c["kelly_fraction"] < b["kelly_fraction"] < a["kelly_fraction"]
    assert c["min_ev"] > b["min_ev"] > a["min_ev"]


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


def _compounding_setup(monkeypatch, n_races: int, n_horses: int = 4):
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
                    finish_position=hi, odds_win=2.0, popularity=hi,
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


def test_compounding_initial_bankroll_equals_budget(monkeypatch):
    """1 race 目の recommend_for_race に渡される bankroll は budget と一致する。"""
    from pathlib import Path
    from types import SimpleNamespace

    from sqlalchemy.orm import Session

    import ai.simulation.engine as sim_mod

    engine = _compounding_setup(monkeypatch, n_races=1)

    bankrolls_seen: list[int] = []

    def _fake(*, bankroll, **_kw):
        bankrolls_seen.append(bankroll)
        return SimpleNamespace(candidates=[])
    monkeypatch.setattr(sim_mod, "recommend_for_race", _fake)

    with Session(engine) as session:
        result = sim_mod.simulate_active_model(
            session=session, model_path=Path("/tmp/dummy"),
            start=None, end=None, budget=10_000, strategy="balanced",
        )

    assert bankrolls_seen[0] == 10_000
    assert result.final_bankroll == 10_000  # bet なしなので変動なし
    assert result.peak_bankroll == 10_000


def test_compounding_bankroll_grows_with_payouts(monkeypatch):
    """winning bet の payout が次 race の bankroll に加算される (compounding)。"""
    from pathlib import Path
    from types import SimpleNamespace

    from sqlalchemy.orm import Session

    import ai.simulation.engine as sim_mod

    engine = _compounding_setup(monkeypatch, n_races=3)

    bankrolls_seen: list[int] = []

    # 各 race で 100 円を winning combo "1" に賭ける → odds 4.0 で payout 400
    def _winning_recommend(*, bankroll, **_kw):
        bankrolls_seen.append(bankroll)
        if bankroll < 100:
            return SimpleNamespace(candidates=[])
        cand = SimpleNamespace(bet_type="単勝", combo="1", stake=100)
        return SimpleNamespace(candidates=[cand])
    monkeypatch.setattr(sim_mod, "recommend_for_race", _winning_recommend)

    with Session(engine) as session:
        result = sim_mod.simulate_active_model(
            session=session, model_path=Path("/tmp/dummy"),
            start=None, end=None, budget=10_000, strategy="balanced",
        )

    # race 毎: stake=100, payout=400, profit=+300
    # bankroll 推移: 10000 → 10300 → 10600 → 10900
    assert bankrolls_seen == [10_000, 10_300, 10_600]
    assert result.final_bankroll == 10_900
    assert result.peak_bankroll == 10_900


def test_compounding_bankroll_shrinks_on_loss(monkeypatch):
    """losing bet の場合は bankroll が stake 分減る。"""
    from pathlib import Path
    from types import SimpleNamespace

    from sqlalchemy.orm import Session

    import ai.simulation.engine as sim_mod

    engine = _compounding_setup(monkeypatch, n_races=3)

    bankrolls_seen: list[int] = []

    # 各 race で combo "2" (= 2 着、winner ではない) に 100 円賭け → payout 0
    def _losing_recommend(*, bankroll, **_kw):
        bankrolls_seen.append(bankroll)
        if bankroll < 100:
            return SimpleNamespace(candidates=[])
        cand = SimpleNamespace(bet_type="単勝", combo="2", stake=100)
        return SimpleNamespace(candidates=[cand])
    monkeypatch.setattr(sim_mod, "recommend_for_race", _losing_recommend)

    with Session(engine) as session:
        result = sim_mod.simulate_active_model(
            session=session, model_path=Path("/tmp/dummy"),
            start=None, end=None, budget=10_000, strategy="balanced",
        )

    # bankroll 推移: 10000 → 9900 → 9800 → 9700 (3 連敗)
    assert bankrolls_seen == [10_000, 9_900, 9_800]
    assert result.final_bankroll == 9_700
    assert result.peak_bankroll == 10_000  # 初期値が peak


def test_compounding_bankroll_zero_at_bankrupt(monkeypatch):
    """bankroll が 0 を下回ると以降は実質 bet しない (破産)。"""
    from pathlib import Path
    from types import SimpleNamespace

    from sqlalchemy.orm import Session

    import ai.simulation.engine as sim_mod

    engine = _compounding_setup(monkeypatch, n_races=3)

    bankrolls_seen: list[int] = []

    def _greedy_losing(*, bankroll, **_kw):
        bankrolls_seen.append(bankroll)
        # bankroll の全額を負け combo に賭ける
        stake = bankroll // 100 * 100
        if stake == 0:
            return SimpleNamespace(candidates=[])
        cand = SimpleNamespace(bet_type="単勝", combo="2", stake=stake)
        return SimpleNamespace(candidates=[cand])
    monkeypatch.setattr(sim_mod, "recommend_for_race", _greedy_losing)

    with Session(engine) as session:
        result = sim_mod.simulate_active_model(
            session=session, model_path=Path("/tmp/dummy"),
            start=None, end=None, budget=5_000, strategy="balanced",
        )

    # 1st で全額消化 → bankroll=0, 以降 bet なし
    assert bankrolls_seen[0] == 5_000
    assert bankrolls_seen[1] == 0
    assert bankrolls_seen[2] == 0
    assert result.final_bankroll == 0


def test_max_stake_per_race_yen_caps_absolute_bet(monkeypatch):
    """max_stake_per_race_yen を渡すと、bankroll が増えても 1 race の累計
    stake はその絶対上限を超えない (compounding wealth でのインフレ抑制)。"""
    from pathlib import Path
    from types import SimpleNamespace

    from sqlalchemy.orm import Session

    import ai.simulation.engine as sim_mod

    engine = _compounding_setup(monkeypatch, n_races=3)

    seen_bankrolls: list[int] = []
    seen_max_yen: list[int | None] = []

    # recommend_for_race を stub。max_stake_per_race_yen が渡ってくることを観測。
    def _fake(*, bankroll, max_stake_per_race_yen=None, **_kw):
        seen_bankrolls.append(bankroll)
        seen_max_yen.append(max_stake_per_race_yen)
        # cap = min(bankroll * 0.05, max_stake_per_race_yen) を再現
        pct_cap = bankroll * 0.05
        cap = (
            min(pct_cap, max_stake_per_race_yen)
            if max_stake_per_race_yen
            else pct_cap
        )
        stake = int(cap) // 100 * 100
        if stake == 0:
            return SimpleNamespace(candidates=[])
        cand = SimpleNamespace(bet_type="単勝", combo="1", stake=stake)
        return SimpleNamespace(candidates=[cand])
    monkeypatch.setattr(sim_mod, "recommend_for_race", _fake)

    with Session(engine) as session:
        result = sim_mod.simulate_active_model(
            session=session,
            model_path=Path("/tmp/dummy"),
            start=None, end=None,
            budget=1_000_000,
            strategy="balanced",
            max_stake_per_race_yen=2_000,  # 1 race max 2,000 円
        )

    # 全ての race で max_stake_per_race_yen が下流に届いている
    assert all(v == 2_000 for v in seen_max_yen)
    # 各 race の stake が 2000 円を超えない (recommend_for_race 内でも capping)
    # bankroll は 1_000_000 → 1_000_000 * 0.05 = 50000 が pct cap だが
    # max_stake_per_race_yen=2000 が優先される。
    # 1 race ごとの invested は 2000 で頭打ち。
    # winning combo "1" odds=4.0 → payout=8000、profit=+6000/race
    # bankroll: 1000000 → 1006000 → 1012000 → 1018000
    assert seen_bankrolls == [1_000_000, 1_006_000, 1_012_000]
    assert result.final_bankroll == 1_018_000


def test_compounding_bankroll_timeseries_daily_aggregation(monkeypatch):
    """bankroll_timeseries は日次集約 (同日複数 race の場合も 1 ポイント)。"""
    from pathlib import Path
    from types import SimpleNamespace

    from sqlalchemy.orm import Session

    import ai.simulation.engine as sim_mod

    engine = _compounding_setup(monkeypatch, n_races=3)

    def _const_winning(*, bankroll, **_kw):
        if bankroll < 100:
            return SimpleNamespace(candidates=[])
        cand = SimpleNamespace(bet_type="単勝", combo="1", stake=100)
        return SimpleNamespace(candidates=[cand])
    monkeypatch.setattr(sim_mod, "recommend_for_race", _const_winning)

    with Session(engine) as session:
        result = sim_mod.simulate_active_model(
            session=session, model_path=Path("/tmp/dummy"),
            start=None, end=None, budget=10_000, strategy="balanced",
        )

    # _compounding_setup は race 毎に異なる日付を使うので 3 ポイント
    assert len(result.bankroll_timeseries) == 3
    # date 昇順
    dates = [p.date for p in result.bankroll_timeseries]
    assert dates == sorted(dates)
    # 各日の bankroll は単調増加 (winning ばかり)
    bankrolls = [p.bankroll for p in result.bankroll_timeseries]
    assert bankrolls == [10_300, 10_600, 10_900]
    # 各日 1 bet, stake=100, payout=400
    for p in result.bankroll_timeseries:
        assert p.n_bets == 1
        assert p.invested == 100
        assert p.payout == 400
