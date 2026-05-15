"""Tests for ai/bet_strategy.py — pattern generation, Kelly stake, assign_stakes,
and recommend_for_race."""

from __future__ import annotations

import math
from itertools import combinations, permutations

import pandas as pd
import pytest

from ai.bet_strategy import (
    assign_stakes,
    generate_box,
    generate_formation,
    generate_nagashi,
    kelly_stake,
    recommend_for_race,
)
from ai.types import BetCandidate, CombinationPrediction

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cp(combo: str, prob: float, est_odds: float, post_positions: tuple[int, ...]) -> CombinationPrediction:
    return CombinationPrediction(
        combo=combo,
        prob=prob,
        est_odds=est_odds,
        ev=prob * est_odds,
        post_positions=post_positions,
    )


def _umaren_combos(n: int, odds: float = 50.0) -> list[CombinationPrediction]:
    """Generate all C(n,2) 馬連 combinations for horses 1..n."""
    total = n * (n - 1) / 2
    prob = 1.0 / total
    result = []
    for a, b in combinations(range(1, n + 1), 2):
        result.append(_cp(f"{a}-{b}", prob, odds, (a, b)))
    return result


def _wide_combos(n: int, odds: float = 15.0) -> list[CombinationPrediction]:
    total = n * (n - 1) / 2
    prob = 1.0 / total
    result = []
    for a, b in combinations(range(1, n + 1), 2):
        result.append(_cp(f"{a}-{b}", prob, odds, (a, b)))
    return result


def _umatan_combos(n: int, odds: float = 100.0) -> list[CombinationPrediction]:
    total = n * (n - 1)
    prob = 1.0 / total
    result = []
    for a, b in permutations(range(1, n + 1), 2):
        result.append(_cp(f"{a}→{b}", prob, odds, (a, b)))
    return result


def _sanrenpuku_combos(n: int, odds: float = 100.0) -> list[CombinationPrediction]:
    total = math.comb(n, 3)
    prob = 1.0 / total
    result = []
    for a, b, c in combinations(range(1, n + 1), 3):
        result.append(_cp(f"{a}-{b}-{c}", prob, odds, (a, b, c)))
    return result


def _sanrentan_combos(n: int, odds: float = 500.0) -> list[CombinationPrediction]:
    total = math.perm(n, 3)
    prob = 1.0 / total
    result = []
    for a, b, c in permutations(range(1, n + 1), 3):
        result.append(_cp(f"{a}→{b}→{c}", prob, odds, (a, b, c)))
    return result


def _tansho_combos(n: int, odds: float = 10.0) -> list[CombinationPrediction]:
    prob = 1.0 / n
    return [_cp(str(i), prob, odds, (i,)) for i in range(1, n + 1)]


def _fukusho_combos(n: int, odds: float = 2.0) -> list[CombinationPrediction]:
    prob = 3.0 / n  # approx: top-3 / n horses
    return [_cp(str(i), prob, odds, (i,)) for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
# kelly_stake
# ---------------------------------------------------------------------------

class TestKellyStake:
    def test_textbook_value(self):
        # prob=0.5, odds=3.0 (net 2x), bankroll=10000, kelly_fraction=1.0
        # edge = 0.5*3 - 1 = 0.5
        # fraction = 1.0 * 0.5 / (3.0 - 1.0) = 0.25
        # raw = 10000 * 0.25 = 2500
        # rounded to 100 → 2500
        assert kelly_stake(0.5, 3.0, 10_000, 1.0, 100) == 2500

    def test_quarter_kelly(self):
        # Same as above but kelly_fraction=0.25
        # fraction = 0.25 * 0.25 = 0.0625
        # raw = 10000 * 0.0625 = 625 → 600 (floor to 100)
        assert kelly_stake(0.5, 3.0, 10_000, 0.25, 100) == 600

    def test_negative_edge_returns_zero(self):
        # prob=0.1, odds=5.0 → edge = 0.5 - 1 = -0.5 ≤ 0
        assert kelly_stake(0.1, 5.0, 100_000, 0.25) == 0

    def test_zero_prob_returns_zero(self):
        assert kelly_stake(0.0, 10.0, 100_000, 0.25) == 0

    def test_odds_le_1_returns_zero(self):
        # odds = 1.0 → denominator = 0, should return 0
        assert kelly_stake(0.5, 1.0, 100_000, 0.25) == 0
        assert kelly_stake(0.5, 0.9, 100_000, 0.25) == 0

    def test_rounding_floor(self):
        # prob=0.4, odds=4.0, bankroll=10000, fraction=1.0
        # edge = 1.6 - 1 = 0.6
        # fraction = 0.6 / 3.0 = 0.2
        # raw = 2000 → 2000 (exact multiple)
        assert kelly_stake(0.4, 4.0, 10_000, 1.0, 100) == 2000

    def test_round_to_500(self):
        # prob=0.5, odds=3.0, bankroll=10000, fraction=1.0, round_to=500
        # raw = 2500 → 2500 (exact multiple of 500)
        assert kelly_stake(0.5, 3.0, 10_000, 1.0, 500) == 2500

    def test_fractional_round_down(self):
        # prob=0.5, odds=2.5, bankroll=10000, fraction=1.0
        # edge = 1.25 - 1 = 0.25
        # fraction = 0.25 / 1.5 = 0.1667
        # raw = 1666.7 → 1600
        result = kelly_stake(0.5, 2.5, 10_000, 1.0, 100)
        assert result == 1600

    def test_high_prob_positive_edge(self):
        # prob=0.8, odds=2.0, bankroll=100000, fraction=0.25
        # edge = 1.6 - 1 = 0.6
        # fraction = 0.25 * 0.6 / 1.0 = 0.15
        # raw = 15000 → 15000
        assert kelly_stake(0.8, 2.0, 100_000, 0.25, 100) == 15000


# ---------------------------------------------------------------------------
# generate_nagashi
# ---------------------------------------------------------------------------

class TestGenerateNagashi:
    def test_umaren_includes_axis(self):
        n = 6
        combos = _umaren_combos(n)
        # Axis = horse 1: should match combos (1,2),(1,3),(1,4),(1,5),(1,6) = n-1 items
        result = generate_nagashi(combos, axis_post_position=1, bet_type="馬連")
        assert len(result) == n - 1
        for c in result:
            assert 1 in c.post_positions

    def test_wide_includes_axis(self):
        n = 5
        combos = _wide_combos(n)
        result = generate_nagashi(combos, axis_post_position=3, bet_type="ワイド")
        assert len(result) == n - 1
        for c in result:
            assert 3 in c.post_positions

    def test_umatan_axis_1st(self):
        n = 5
        combos = _umatan_combos(n)
        # axis=2, 1st fixed: should be (2,1),(2,3),(2,4),(2,5) = n-1 items
        result = generate_nagashi(combos, axis_post_position=2, bet_type="馬単", axis_position=1)
        assert len(result) == n - 1
        for c in result:
            assert c.post_positions[0] == 2

    def test_umatan_axis_2nd(self):
        n = 5
        combos = _umatan_combos(n)
        result = generate_nagashi(combos, axis_post_position=2, bet_type="馬単", axis_position=2)
        assert len(result) == n - 1
        for c in result:
            assert c.post_positions[1] == 2

    def test_sanrenpuku_axis(self):
        n = 6
        combos = _sanrenpuku_combos(n)
        # axis=1: all triples containing horse 1 = C(5,2) = 10
        result = generate_nagashi(combos, axis_post_position=1, bet_type="三連複")
        assert len(result) == math.comb(n - 1, 2)
        for c in result:
            assert 1 in c.post_positions

    def test_unsupported_bet_type_returns_empty(self):
        combos = _tansho_combos(8)
        assert generate_nagashi(combos, axis_post_position=1, bet_type="単勝") == []

    def test_sanrentan_returns_empty(self):
        combos = _sanrentan_combos(5)
        assert generate_nagashi(combos, axis_post_position=1, bet_type="三連単") == []

    def test_pattern_label(self):
        combos = _umaren_combos(5)
        result = generate_nagashi(combos, axis_post_position=1, bet_type="馬連")
        assert all(c.pattern == "nagashi" for c in result)

    def test_stake_initialized_to_zero(self):
        combos = _umaren_combos(5)
        result = generate_nagashi(combos, axis_post_position=1, bet_type="馬連")
        assert all(c.stake == 0 for c in result)


# ---------------------------------------------------------------------------
# generate_box
# ---------------------------------------------------------------------------

class TestGenerateBox:
    def test_umaren_count(self):
        # C(4,2) = 6
        n_box = 4
        combos = _umaren_combos(8)
        result = generate_box(combos, list(range(1, n_box + 1)), "馬連")
        assert len(result) == math.comb(n_box, 2)

    def test_wide_count(self):
        n_box = 5
        combos = _wide_combos(8)
        result = generate_box(combos, list(range(1, n_box + 1)), "ワイド")
        assert len(result) == math.comb(n_box, 2)

    def test_umatan_count(self):
        # P(4,2) = 12
        n_box = 4
        combos = _umatan_combos(8)
        result = generate_box(combos, list(range(1, n_box + 1)), "馬単")
        assert len(result) == math.perm(n_box, 2)

    def test_sanrenpuku_count(self):
        # C(4,3) = 4
        n_box = 4
        combos = _sanrenpuku_combos(8)
        result = generate_box(combos, list(range(1, n_box + 1)), "三連複")
        assert len(result) == math.comb(n_box, 3)

    def test_sanrentan_count(self):
        # P(4,3) = 24
        n_box = 4
        combos = _sanrentan_combos(8)
        result = generate_box(combos, list(range(1, n_box + 1)), "三連単")
        assert len(result) == math.perm(n_box, 3)

    def test_tansho_count(self):
        combos = _tansho_combos(8)
        result = generate_box(combos, [1, 2, 3], "単勝")
        assert len(result) == 3

    def test_fukusho_count(self):
        combos = _fukusho_combos(8)
        result = generate_box(combos, [2, 4, 6], "複勝")
        assert len(result) == 3

    def test_pattern_label(self):
        combos = _umaren_combos(6)
        result = generate_box(combos, [1, 2, 3], "馬連")
        assert all(c.pattern == "box" for c in result)

    def test_stake_initialized_to_zero(self):
        combos = _umaren_combos(6)
        result = generate_box(combos, [1, 2, 3], "馬連")
        assert all(c.stake == 0 for c in result)

    def test_all_post_positions_in_box_set(self):
        box_pps = {1, 2, 3}
        combos = _umatan_combos(6)
        result = generate_box(combos, list(box_pps), "馬単")
        for c in result:
            assert set(c.post_positions).issubset(box_pps)


# ---------------------------------------------------------------------------
# generate_formation
# ---------------------------------------------------------------------------

class TestGenerateFormation:
    def test_umatan_count(self):
        # first=[1], second=[1,2,3] → 1*2 (exclude 1→1) = 2 valid combos
        combos = _umatan_combos(5)
        result = generate_formation(combos, [1], [1, 2, 3], None, "馬単")
        assert len(result) == 2  # (1,2) and (1,3)

    def test_umatan_excludes_same_horse(self):
        combos = _umatan_combos(5)
        result = generate_formation(combos, [1, 2], [2, 3], None, "馬単")
        for c in result:
            assert c.post_positions[0] != c.post_positions[1]

    def test_umatan_pattern_label(self):
        combos = _umatan_combos(5)
        result = generate_formation(combos, [1], [2, 3], None, "馬単")
        assert all(c.pattern == "formation" for c in result)

    def test_sanrentan_count(self):
        # first=[1], second=[2,3], third=[3,4] → (1,2,3),(1,2,4),(1,3,4) = 3 valid
        combos = _sanrentan_combos(5)
        result = generate_formation(combos, [1], [2, 3], [3, 4], "三連単")
        expected = {(1, 2, 3), (1, 2, 4), (1, 3, 4)}
        result_pps = {c.post_positions for c in result}
        assert result_pps == expected

    def test_sanrentan_all_distinct(self):
        combos = _sanrentan_combos(6)
        result = generate_formation(combos, [1, 2], [2, 3], [3, 4], "三連単")
        for c in result:
            assert len(set(c.post_positions)) == 3

    def test_sanrenpuku_delegates_to_box(self):
        combos = _sanrenpuku_combos(8)
        # union of {1},{2,3},{4} → box of {1,2,3,4} = C(4,3)=4
        result = generate_formation(combos, [1], [2, 3], [4], "三連複")
        assert len(result) == math.comb(4, 3)
        assert all(c.pattern == "box" for c in result)

    def test_umaren_returns_empty(self):
        combos = _umaren_combos(6)
        assert generate_formation(combos, [1], [2, 3], None, "馬連") == []

    def test_wide_returns_empty(self):
        combos = _wide_combos(6)
        assert generate_formation(combos, [1], [2, 3], None, "ワイド") == []

    def test_unsupported_returns_empty(self):
        combos = _tansho_combos(6)
        assert generate_formation(combos, [1], [2], None, "単勝") == []

    def test_sanrentan_missing_third_returns_empty(self):
        combos = _sanrentan_combos(5)
        assert generate_formation(combos, [1], [2], None, "三連単") == []


# ---------------------------------------------------------------------------
# assign_stakes
# ---------------------------------------------------------------------------

class TestAssignStakes:
    def _make_candidates(self, n: int, prob: float, odds: float) -> list[BetCandidate]:
        return [
            BetCandidate(
                bet_type="馬連",
                combo=f"{i}-{i+1}",
                pattern="box",
                prob=prob,
                est_odds=odds,
                ev=prob * odds,
                stake=0,
                post_positions=(i, i + 1),
            )
            for i in range(1, n + 1)
        ]

    def test_ev_le_1_excluded(self):
        cands = self._make_candidates(3, prob=0.05, odds=10.0)  # ev=0.5 < 1
        # Force ev to be below 1 for all
        low_ev = [c.model_copy(update={"ev": 0.8}) for c in cands]
        result = assign_stakes(low_ev, bankroll=100_000, kelly_fraction=0.25,
                               max_stake_per_race_pct=0.10)
        assert result == []

    def test_basic_kelly_applied(self):
        # Single candidate with ev > 1 → stake should be non-zero
        cand = BetCandidate(
            bet_type="単勝",
            combo="1",
            pattern="box",
            prob=0.5,
            est_odds=3.0,
            ev=1.5,
            stake=0,
            post_positions=(1,),
        )
        result = assign_stakes([cand], bankroll=10_000, kelly_fraction=1.0,
                               max_stake_per_race_pct=1.0)
        assert len(result) == 1
        assert result[0].stake == 2500  # matches kelly_stake(0.5,3.0,10000,1.0,100)

    def test_proportional_cap_applied(self):
        # 5 candidates each with kelly stake 2500 → total 12500
        # cap = 10000 * 0.05 = 500 → all stakes scaled down
        cands = [
            BetCandidate(
                bet_type="馬連",
                combo=f"{i}-{i+1}",
                pattern="box",
                prob=0.5,
                est_odds=3.0,
                ev=1.5,
                stake=0,
                post_positions=(i, i + 1),
            )
            for i in range(1, 6)
        ]
        result = assign_stakes(cands, bankroll=10_000, kelly_fraction=1.0,
                               max_stake_per_race_pct=0.05)
        total = sum(c.stake for c in result)
        # Total must not exceed cap = 500; all stakes multiples of 100
        assert total <= 500
        for c in result:
            assert c.stake % 100 == 0

    def test_cap_not_exceeded(self):
        # Large bankroll, many candidates
        cands = self._make_candidates(10, prob=0.5, odds=5.0)  # ev=2.5 > 1
        bankroll = 100_000
        pct = 0.03
        result = assign_stakes(cands, bankroll=bankroll, kelly_fraction=0.25,
                               max_stake_per_race_pct=pct)
        total = sum(c.stake for c in result)
        assert total <= bankroll * pct + 100  # +100 for rounding tolerance

    def test_zero_stake_after_cap_excluded(self):
        # Many candidates → after scaling some may become 0 → excluded
        cands = self._make_candidates(50, prob=0.5, odds=2.0)  # ev=1.0 exactly
        # ev == 1.0 is NOT > 1.0, so all should be excluded
        result = assign_stakes(cands, bankroll=100_000, kelly_fraction=0.25,
                               max_stake_per_race_pct=0.05)
        assert result == []

    def test_stake_multiples_of_100(self):
        cands = [
            BetCandidate(
                bet_type="馬連",
                combo="1-3",
                pattern="nagashi",
                prob=0.35,
                est_odds=4.0,
                ev=1.4,
                stake=0,
                post_positions=(1, 3),
            )
        ]
        result = assign_stakes(cands, bankroll=77_777, kelly_fraction=0.25,
                               max_stake_per_race_pct=1.0)
        for c in result:
            assert c.stake % 100 == 0

    def test_original_candidates_not_mutated(self):
        cand = BetCandidate(
            bet_type="単勝",
            combo="1",
            pattern="box",
            prob=0.5,
            est_odds=3.0,
            ev=1.5,
            stake=0,
            post_positions=(1,),
        )
        assign_stakes([cand], bankroll=10_000, kelly_fraction=1.0,
                      max_stake_per_race_pct=1.0)
        assert cand.stake == 0  # original unchanged

    def test_keep_zero_stake_includes_low_ev_candidates(self):
        """keep_zero_stake=True retains ev<=1.0 candidates with stake=0."""
        high_ev = BetCandidate(
            bet_type="馬連", combo="1-2", pattern="box",
            prob=0.5, est_odds=3.0, ev=1.5, stake=0, post_positions=(1, 2),
        )
        low_ev = BetCandidate(
            bet_type="馬連", combo="1-3", pattern="box",
            prob=0.1, est_odds=5.0, ev=0.5, stake=0, post_positions=(1, 3),
        )
        result = assign_stakes(
            [high_ev, low_ev],
            bankroll=10_000,
            kelly_fraction=1.0,
            max_stake_per_race_pct=1.0,
            keep_zero_stake=True,
        )
        combos = {c.combo for c in result}
        # high-ev candidate appears with positive stake
        high_result = next(c for c in result if c.combo == "1-2")
        assert high_result.stake > 0
        # low-ev candidate is present with stake=0
        assert "1-3" in combos
        low_result = next(c for c in result if c.combo == "1-3")
        assert low_result.stake == 0

    def test_keep_zero_stake_false_excludes_zero_stake(self):
        """Default keep_zero_stake=False excludes ev<=1.0 candidates (backward compat)."""
        low_ev = BetCandidate(
            bet_type="単勝", combo="1", pattern="box",
            prob=0.05, est_odds=5.0, ev=0.25, stake=0, post_positions=(1,),
        )
        result = assign_stakes(
            [low_ev],
            bankroll=100_000,
            kelly_fraction=0.25,
            max_stake_per_race_pct=0.05,
        )
        assert result == []


# ---------------------------------------------------------------------------
# recommend_for_race (integration)
# ---------------------------------------------------------------------------

class TestRecommendForRace:
    def _build_predictions(self, n: int) -> pd.DataFrame:
        """Simple predictions DataFrame: horse i has win_prob proportional to (n-i+1)."""
        rows = []
        total = sum(range(1, n + 1))
        for i in range(1, n + 1):
            rows.append({
                "horse_id": str(i),
                "score": float(n - i + 1),
                "win_prob": (n - i + 1) / total,
                "post_position": i,
            })
        return pd.DataFrame(rows)

    def test_returns_recommendation_result(self):
        from ai.types import RecommendationResult
        preds = self._build_predictions(8)
        combos = {
            "馬連": _umaren_combos(8, odds=50.0),
            "ワイド": _wide_combos(8, odds=15.0),
        }
        result = recommend_for_race(
            predictions=preds,
            combinations_by_type=combos,
            race_id="test_race_001",
            bankroll=100_000,
            kelly_fraction=0.25,
            max_stake_per_race_pct=0.05,
        )
        assert isinstance(result, RecommendationResult)
        assert result.race_id == "test_race_001"
        assert result.bankroll_at_decision == 100_000

    def test_candidates_have_positive_stake(self):
        preds = self._build_predictions(8)
        combos = {
            "単勝": _tansho_combos(8, odds=10.0),
        }
        result = recommend_for_race(
            predictions=preds,
            combinations_by_type=combos,
            race_id="r1",
            bankroll=100_000,
            kelly_fraction=0.25,
            max_stake_per_race_pct=0.10,
        )
        for c in result.candidates:
            assert c.stake > 0

    def test_total_stake_within_cap(self):
        preds = self._build_predictions(8)
        bankroll = 100_000
        pct = 0.05
        combos = {
            "馬連": _umaren_combos(8, odds=50.0),
            "三連複": _sanrenpuku_combos(8, odds=100.0),
        }
        result = recommend_for_race(
            predictions=preds,
            combinations_by_type=combos,
            race_id="r2",
            bankroll=bankroll,
            kelly_fraction=0.25,
            max_stake_per_race_pct=pct,
        )
        total = sum(c.stake for c in result.candidates)
        assert total <= bankroll * pct + 100  # +100 rounding tolerance

    def test_enabled_bet_types_filter(self):
        preds = self._build_predictions(6)
        combos = {
            "馬連": _umaren_combos(6, odds=50.0),
            "三連単": _sanrentan_combos(6, odds=500.0),
        }
        result = recommend_for_race(
            predictions=preds,
            combinations_by_type=combos,
            race_id="r3",
            bankroll=100_000,
            kelly_fraction=0.25,
            max_stake_per_race_pct=0.10,
            enabled_bet_types=["馬連"],
        )
        for c in result.candidates:
            assert c.bet_type == "馬連"

    def test_empty_combinations_returns_empty_candidates(self):
        preds = self._build_predictions(5)
        result = recommend_for_race(
            predictions=preds,
            combinations_by_type={},
            race_id="r4",
            bankroll=100_000,
            kelly_fraction=0.25,
            max_stake_per_race_pct=0.05,
        )
        assert result.candidates == []

    def test_no_duplicate_combo_per_bet_type(self):
        preds = self._build_predictions(6)
        combos = {
            "馬連": _umaren_combos(6, odds=50.0),
            "ワイド": _wide_combos(6, odds=15.0),
        }
        result = recommend_for_race(
            predictions=preds,
            combinations_by_type=combos,
            race_id="r5",
            bankroll=200_000,
            kelly_fraction=0.25,
            max_stake_per_race_pct=0.20,
        )
        seen = set()
        for c in result.candidates:
            key = (c.bet_type, c.combo)
            assert key not in seen, f"Duplicate candidate: {key}"
            seen.add(key)

    def test_top_n_horses_parameter(self):
        """top_n_horses=2 should generate smaller box than default 3."""
        preds = self._build_predictions(8)
        combos = {"馬連": _umaren_combos(8, odds=50.0)}

        result_n2 = recommend_for_race(
            predictions=preds,
            combinations_by_type=combos,
            race_id="r6a",
            bankroll=200_000,
            kelly_fraction=1.0,
            max_stake_per_race_pct=0.50,
            top_n_horses=2,
        )
        result_n4 = recommend_for_race(
            predictions=preds,
            combinations_by_type=combos,
            race_id="r6b",
            bankroll=200_000,
            kelly_fraction=1.0,
            max_stake_per_race_pct=0.50,
            top_n_horses=4,
        )
        # n=2 box has C(2,2)=1 combo; n=4 box has C(4,2)=6 combos
        # nagashi with n=2 would have 1 combo; with n=4 the pattern selection
        # may pick differently.  At minimum, larger top_n should not reduce output.
        assert len(result_n4.candidates) >= len(result_n2.candidates)
