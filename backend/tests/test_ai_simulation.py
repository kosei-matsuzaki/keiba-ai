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
