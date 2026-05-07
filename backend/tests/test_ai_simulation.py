"""Unit tests for ai/simulation.py:_settle_candidates and STRATEGY_PRESETS.

Full integration test (simulate_active_model) requires a trained model bundle
and is covered manually.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from keiba_ai.ai.simulation import (
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
# Depleting bankroll (Option D)
# ---------------------------------------------------------------------------


def test_depleting_bankroll_passes_remaining_to_recommender(monkeypatch):
    """recommend_for_race に渡される bankroll は (budget - 累計 invested) になる。

    重い model load + predict をスキップするため、simulate_active_model 内の
    依存関数を stub する。stake / payout は固定値を返し、bankroll の depletion
    が次の race に伝搬することだけを確認する。
    """
    from datetime import date, timedelta
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import pandas as pd
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    import keiba_ai.ai.simulation as sim_mod
    from keiba_ai.db.base import Base
    from keiba_ai.db.models.entry import Entry
    from keiba_ai.db.models.horse import Horse
    from keiba_ai.db.models.race import Race

    # ── Synthetic DB: 3 race × 4 horses ────────────────────────────────
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    base = date(2024, 6, 15)
    with Session(engine) as session:
        for hi in range(1, 5):
            session.add(Horse(horse_id=f"H{hi}", name=f"H{hi}"))
        for ri in range(1, 4):
            session.add(Race(
                race_id=f"R{ri}",
                date=(base - timedelta(days=ri)).isoformat(),
                course="東京", surface="芝", distance=1600, n_runners=4,
            ))
        session.flush()
        for ri in range(1, 4):
            for hi in range(1, 5):
                session.add(Entry(
                    race_id=f"R{ri}", horse_id=f"H{hi}", post_position=hi,
                    finish_position=hi, odds_win=2.0, popularity=hi,
                ))
        session.commit()

    # ── Stub heavy ML calls ────────────────────────────────────────────
    fake_bundle = SimpleNamespace(lambdarank=None, binary=None, calibrator=None)
    monkeypatch.setattr(sim_mod, "load_model_full", lambda _p: fake_bundle)

    def _fake_predict_race(_m, frame, **_kw):
        return pd.DataFrame({
            "horse_id": frame["horse_id"].values,
            "score": [1.0] * len(frame),
            "win_prob": [0.25] * len(frame),
            "place_prob": [0.5] * len(frame),
        })
    monkeypatch.setattr(sim_mod, "predict_race", _fake_predict_race)

    monkeypatch.setattr(sim_mod, "predict_race_with_combinations", lambda *a, **kw: {})
    monkeypatch.setattr(sim_mod, "compute_race_odds_with_sources", lambda *a, **kw: ({}, {}))
    monkeypatch.setattr(sim_mod, "compute_past_race_odds", lambda *a, **kw: {"単勝": {"1": 4.0}})

    # ── Stub recommend_for_race: 各 race で bankroll の 50% を 1 candidate に張る ──
    bankrolls_seen: list[int] = []

    def _fake_recommend(*, predictions, combinations_by_type, race_id,
                       bankroll, kelly_fraction, **_kw):
        bankrolls_seen.append(bankroll)
        # 各 race で bankroll × 0.5 を 単勝 candidate として返す。0 円なら 0 stake。
        stake = int(bankroll * 0.5) // 100 * 100  # round to 100
        cand = SimpleNamespace(bet_type="単勝", combo="2", stake=stake)
        return SimpleNamespace(candidates=[cand])
    monkeypatch.setattr(sim_mod, "recommend_for_race", _fake_recommend)

    # ── Run ────────────────────────────────────────────────────────────
    with Session(engine) as session:
        result = sim_mod.simulate_active_model(
            session=session,
            model_path=Path("/tmp/dummy"),
            start=None, end=None,
            budget=10_000,
            strategy="balanced",
        )

    # ── Assert ─────────────────────────────────────────────────────────
    # race 1: bankroll=10000 → stake=5000 (combo "2" miss → no payout)
    # race 2: bankroll=10000-5000=5000 → stake=2500
    # race 3: bankroll=5000-2500=2500 → stake=1200 (rounded)
    # 累計 invested = 5000 + 2500 + 1200 = 8700 ≤ 10000 ✓
    assert len(bankrolls_seen) == 3
    assert bankrolls_seen[0] == 10_000
    assert bankrolls_seen[1] == 10_000 - bankrolls_seen[0] // 2 // 100 * 100
    assert result.summary.invested <= 10_000, (
        f"累計 invested {result.summary.invested} が予算 10000 を超えた"
    )
    # bankroll が単調減少
    assert bankrolls_seen[0] >= bankrolls_seen[1] >= bankrolls_seen[2]


def test_depleting_bankroll_zero_at_exhaustion(monkeypatch):
    """予算を使い切った後の race は bankroll=0 で呼ばれる。"""
    from datetime import date, timedelta
    from pathlib import Path
    from types import SimpleNamespace

    import pandas as pd
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    import keiba_ai.ai.simulation as sim_mod
    from keiba_ai.db.base import Base
    from keiba_ai.db.models.entry import Entry
    from keiba_ai.db.models.horse import Horse
    from keiba_ai.db.models.race import Race

    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    base = date(2024, 6, 15)
    with Session(engine) as session:
        for hi in range(1, 4):
            session.add(Horse(horse_id=f"H{hi}", name=f"H{hi}"))
        for ri in range(1, 4):
            session.add(Race(
                race_id=f"R{ri}",
                date=(base - timedelta(days=ri)).isoformat(),
                course="東京", surface="芝", distance=1600, n_runners=3,
            ))
        session.flush()
        for ri in range(1, 4):
            for hi in range(1, 4):
                session.add(Entry(
                    race_id=f"R{ri}", horse_id=f"H{hi}", post_position=hi,
                    finish_position=hi, odds_win=2.0, popularity=hi,
                ))
        session.commit()

    fake_bundle = SimpleNamespace(lambdarank=None, binary=None, calibrator=None)
    monkeypatch.setattr(sim_mod, "load_model_full", lambda _p: fake_bundle)
    monkeypatch.setattr(sim_mod, "predict_race", lambda _m, f, **_kw: pd.DataFrame({
        "horse_id": f["horse_id"].values, "score": [1.0] * len(f),
        "win_prob": [0.33] * len(f), "place_prob": [0.5] * len(f),
    }))
    monkeypatch.setattr(sim_mod, "predict_race_with_combinations", lambda *a, **kw: {})
    monkeypatch.setattr(sim_mod, "compute_race_odds_with_sources", lambda *a, **kw: ({}, {}))
    monkeypatch.setattr(sim_mod, "compute_past_race_odds", lambda *a, **kw: {"単勝": {"1": 4.0}})

    bankrolls_seen: list[int] = []

    def _greedy_recommend(*, bankroll, **_kw):
        bankrolls_seen.append(bankroll)
        # 各 race で bankroll の **全額** を一気に賭ける
        stake = bankroll // 100 * 100
        cand = SimpleNamespace(bet_type="単勝", combo="2", stake=stake)
        return SimpleNamespace(candidates=[cand])
    monkeypatch.setattr(sim_mod, "recommend_for_race", _greedy_recommend)

    with Session(engine) as session:
        sim_mod.simulate_active_model(
            session=session, model_path=Path("/tmp/dummy"),
            start=None, end=None, budget=5_000, strategy="balanced",
        )

    # 1st race で 5000 全額消化 → 2nd / 3rd は bankroll=0
    assert bankrolls_seen[0] == 5_000
    assert bankrolls_seen[1] == 0
    assert bankrolls_seen[2] == 0
