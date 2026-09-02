"""Tests for ai/betting/strategy.py — pattern generation, Kelly stake, assign_stakes,
and recommend_for_race."""

from __future__ import annotations

import math
from itertools import combinations, permutations

import pandas as pd

from ai.betting.strategy import (
    assign_flat_stakes,
    generate_box,
    generate_formation,
    generate_nagashi,
    recommend_for_race,
)
from ai.core.types import BetCandidate, CombinationPrediction
from core.bet_types import DEFAULT_COMBO_MIN_HIT_PROB

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
# assign_flat_stakes
# ---------------------------------------------------------------------------

class TestAssignFlatStakes:
    """定額配分: 単勝→複勝→連系の順、同券種内は確率の高い順に 1 点ずつ。"""

    def _c(self, combo: str, ev: float | None, prob: float = 0.1,
           bet_type: str = "単勝") -> BetCandidate:
        return BetCandidate(
            bet_type=bet_type,
            combo=combo,
            pattern="box",
            prob=prob,
            est_odds=(ev / prob) if ev is not None else None,
            ev=ev,
            stake=0,
            post_positions=(int(combo) if combo.isdigit() else 1,),
        )

    def test_each_eligible_candidate_gets_one_unit(self):
        cands = [self._c("1", 1.5), self._c("2", 1.3), self._c("3", 1.2)]
        # 点数上限は別のテストで見る。ここは「対象は 1 点ずつ」だけを確かめる
        out = assign_flat_stakes(
            cands, race_budget=1000
        )
        assert [c.stake for c in out] == [100, 100, 100]

    def test_orders_by_prob_desc_within_a_bet_type(self):
        # 同じ券種内は的中確率の高い順。EV 順にはしない。
        cands = [
            self._c("1", 1.2, prob=0.10),
            self._c("2", 1.9, prob=0.30),
            self._c("3", 1.5, prob=0.20),
        ]
        out = assign_flat_stakes(
            cands, race_budget=1000
        )
        assert [c.combo for c in out] == ["2", "3", "1"]

    def test_win_and_place_come_before_exotics(self):
        """予算が足りないときに切るのは連系。単複の方が回収率の推定が確か。

        較正後は単勝の EV が 0.6 前後で連系 (EV 5〜9) より低く出るので、EV 順に
        並べると単複が最後尾に回って真っ先に切り捨てられていた (実測: 2,034
        レースで単勝 3 点・複勝 1 点)。
        """
        cands = [
            self._c("A", 9.0, prob=0.01, bet_type="三連単"),
            self._c("B", 5.0, prob=0.05, bet_type="馬連"),
            self._c("1", 0.6, prob=0.25, bet_type="単勝"),
            self._c("2", 0.8, prob=0.55, bet_type="複勝"),
        ]
        out = assign_flat_stakes(
            cands, race_budget=200,
        )
        assert [c.bet_type for c in out] == ["単勝", "複勝"]

    def test_points_can_differ_by_bet_type(self):
        """厚みは「1 点いくら」ではなく「何点買うか」で表す (1 点 = 100 円)。"""
        cands = [
            self._c("1", 0.6, prob=0.25, bet_type="単勝"),
            self._c("A", 9.0, prob=0.01, bet_type="三連単"),
        ]
        out = assign_flat_stakes(
            cands, race_budget=5000,
            points_by_bet_type={"単勝": 5, "三連単": 1},
            # ここで見たいのは 1 点あたりの金額なので、的中確率の下限は外す
            min_hit_prob_by_bet_type={},
        )
        assert {(c.bet_type, c.stake) for c in out} == {("単勝", 500), ("三連単", 100)}

    def test_budget_caps_the_number_of_bets(self):
        cands = [self._c(str(i), 2.0 - i * 0.1) for i in range(1, 11)]
        out = assign_flat_stakes(
            cands, race_budget=300
        )
        assert len(out) == 3
        assert sum(c.stake for c in out) == 300

    def test_budget_need_not_be_spent(self):
        """対象が少なければ使い切らない (これが定額運用の要点)。"""
        cands = [self._c("1", 1.5), self._c("2", 0.9)]
        out = assign_flat_stakes(
            cands, race_budget=10_000)
        assert sum(c.stake for c in out) == 200

    def test_ev_no_longer_filters(self):
        """**EV は買う / 買わないの判定に使わない** (2026-08-28)。

        単勝・複勝では EV 条件を捨てて回収率が 0.698→0.931 / 0.654→0.887 と改善し、
        連系だけ残っていた閾値 1.1 にも根拠が無かった。EV が 1.0 を割る買い目も
        確率順で予算に収まる限り買う。
        """
        cands = [self._c("1", 0.6, prob=0.30), self._c("2", 0.5, prob=0.20)]
        out = assign_flat_stakes(
            cands, race_budget=1000)
        assert [c.combo for c in out] == ["1", "2"]  # 確率の高い順

    def test_candidates_without_odds_are_still_excluded(self):
        """値段が分からない買い目は買えない (EV 条件とは別の理由)。"""
        cands = [self._c("1", None), self._c("2", 0.5)]
        out = assign_flat_stakes(
            cands, race_budget=1000)
        assert [c.combo for c in out] == ["2"]

    def test_none_ev_is_not_bet(self):
        cands = [self._c("1", None), self._c("2", 1.5)]
        out = assign_flat_stakes(
            cands, race_budget=1000)
        assert [c.combo for c in out] == ["2"]

    def test_more_points_means_more_money(self):
        """点数がそのまま金額になる (1 点 = 100 円)。"""
        cands = [self._c("1", 1.5), self._c("2", 1.4)]
        out = assign_flat_stakes(
            cands, race_budget=1000, points_by_bet_type={"単勝": 5}
        )
        assert [c.stake for c in out] == [500, 500]

    def test_budget_smaller_than_unit_bets_nothing(self):
        cands = [self._c("1", 1.5)]
        assert assign_flat_stakes(
            cands, race_budget=50) == []

    def test_keep_zero_stake_returns_all(self):
        cands = [self._c("1", 1.5), self._c("2", 0.8)]
        out = assign_flat_stakes(
            cands, race_budget=100, keep_zero_stake=True
        )
        assert len(out) == 2
        assert {c.combo: c.stake for c in out} == {"1": 100, "2": 0}

    def test_does_not_mutate_input(self):
        cands = [self._c("1", 1.5)]
        assign_flat_stakes(
            cands, race_budget=1000)
        assert cands[0].stake == 0

    def test_deterministic_on_ev_ties(self):
        a = self._c("1", 1.5, prob=0.10)
        b = self._c("2", 1.5, prob=0.30)
        out = assign_flat_stakes([a, b], race_budget=100)
        # 同 ev なら確率の高いほうを先に買う
        assert [c.combo for c in out] == ["2"]


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
        from ai.core.types import RecommendationResult
        preds = self._build_predictions(8)
        combos = {
            "馬連": _umaren_combos(8, odds=50.0),
            "ワイド": _wide_combos(8, odds=15.0),
        }
        result = recommend_for_race(
            predictions=preds,
            combinations_by_type=combos,
            race_id="test_race_001",
            race_budget=5_000,
        )
        assert isinstance(result, RecommendationResult)
        assert result.race_id == "test_race_001"
        assert result.race_budget == 5_000

    def test_candidates_have_positive_stake(self):
        preds = self._build_predictions(8)
        combos = {
            "単勝": _tansho_combos(8, odds=10.0),
        }
        result = recommend_for_race(
            predictions=preds,
            combinations_by_type=combos,
            race_id="r1",
            race_budget=5_000,
        )
        for c in result.candidates:
            assert c.stake > 0

    def test_total_stake_within_budget(self):
        """合計は必ず 1 レース予算以下 (使い切らないこともある)。"""
        preds = self._build_predictions(8)
        budget = 1_000
        combos = {
            "馬連": _umaren_combos(8, odds=50.0),
            "三連複": _sanrenpuku_combos(8, odds=100.0),
        }
        result = recommend_for_race(
            predictions=preds,
            combinations_by_type=combos,
            race_id="r2",
            race_budget=budget,
        )
        total = sum(c.stake for c in result.candidates)
        assert total <= budget

    def test_every_bet_is_one_point_by_default(self):
        """点数の指定が無ければどの買い目も 1 点 = 100 円。"""
        preds = self._build_predictions(8)
        combos = {"馬連": _umaren_combos(8, odds=50.0)}
        result = recommend_for_race(
            predictions=preds,
            combinations_by_type=combos,
            race_id="r2b",
            race_budget=1_000,
            min_hit_prob_by_bet_type={},
        )
        staked = [c.stake for c in result.candidates if c.stake > 0]
        assert staked and set(staked) == {100}

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
            race_budget=5_000,
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
            race_budget=5_000,
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
            race_budget=5_000,
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
            race_budget=5_000,
            top_n_horses=2,
        )
        result_n4 = recommend_for_race(
            predictions=preds,
            combinations_by_type=combos,
            race_id="r6b",
            race_budget=5_000,
            top_n_horses=4,
        )
        # n=2 box has C(2,2)=1 combo; n=4 box has C(4,2)=6 combos
        # nagashi with n=2 would have 1 combo; with n=4 the pattern selection
        # may pick differently.  At minimum, larger top_n should not reduce output.
        assert len(result_n4.candidates) >= len(result_n2.candidates)



class TestComboMinHitProb:
    """連系の点数は固定ではなく、**的中確率の下限**でレースごとに変わる。"""

    @staticmethod
    def _c(combo: str, prob: float, bet_type: str = "馬連") -> BetCandidate:
        return BetCandidate(
            bet_type=bet_type,
            combo=combo,
            pattern="box",
            prob=prob,
            est_odds=20.0,
            ev=prob * 20.0,
            stake=0,
            post_positions=[int(x) for x in combo.split("-")],
        )

    def test_points_vary_with_confidence(self):
        """線を超えた買い目の数だけ買う = 堅いレースほど点数が増える。"""
        floors = {"馬連": 0.05}
        confident = [self._c("1-2", 0.20), self._c("1-3", 0.09), self._c("2-3", 0.06)]
        thin = [self._c("1-2", 0.09), self._c("1-3", 0.03), self._c("2-3", 0.02)]

        # 上限は外して、点数の差が下限だけで付くことを見る
        many = assign_flat_stakes(
            confident, race_budget=10_000,
            min_hit_prob_by_bet_type=floors,
        )
        few = assign_flat_stakes(
            thin, race_budget=10_000,
            min_hit_prob_by_bet_type=floors,
        )
        assert len(many) == 3
        assert len(few) == 1

    def test_no_points_when_nothing_clears_the_floor(self):
        """確信度が足りないレースでは連系を 1 点も買わない (予算は残す)。"""
        out = assign_flat_stakes(
            [self._c("1-2", 0.01), self._c("1-3", 0.005)],
            race_budget=10_000,
            min_hit_prob_by_bet_type={"馬連": 0.05},
        )
        assert out == []

    def test_floor_does_not_apply_to_win_and_place(self):
        """単勝・複勝は本命買いのルールで決めるので、連系の下限は掛けない。"""
        cands = [
            self._c("1", 0.02, bet_type="単勝"),
            self._c("1", 0.02, bet_type="複勝"),
        ]
        out = assign_flat_stakes(
            cands, race_budget=10_000,
            min_hit_prob_by_bet_type=DEFAULT_COMBO_MIN_HIT_PROB,
        )
        assert {c.bet_type for c in out} == {"単勝", "複勝"}


    def test_zero_stake_rows_are_kept_for_display(self):
        """画面は買わない買い目も薄く出すので、keep_zero_stake の形は保つ。"""
        out = assign_flat_stakes(
            [self._c("1-2", 0.20), self._c("1-3", 0.01)],
            race_budget=10_000,
            keep_zero_stake=True,
            min_hit_prob_by_bet_type={"馬連": 0.05},
        )
        assert sorted(c.stake for c in out) == [0, 100]



class TestPointsFromConfidence:
    """厚みは「1 点いくら」ではなく **何点買うか** で表す (1 点 = 100 円)。"""

    def test_points_scale_with_confidence(self):
        from ai.inference.confidence import points_for_confidence

        # 複勝は基準 0.50 で 5 点
        assert points_for_confidence("複勝", 0.50) == 5
        assert points_for_confidence("複勝", 0.70) > 5
        assert points_for_confidence("複勝", 0.30) < 5
        # 単勝は確信度の桁が違う (1 着確率) ので基準も違う
        assert points_for_confidence("単勝", 0.25) == 5
        assert points_for_confidence("単勝", 0.50) > 5

    def test_points_are_clamped(self):
        from ai.inference.confidence import MAX_POINTS, points_for_confidence

        assert points_for_confidence("複勝", 0.01) == 1
        assert points_for_confidence("複勝", 1.0) == MAX_POINTS

    def test_missing_confidence_falls_back_to_base(self):
        """確率モデル未設定でも動く。壊れたときに賭け金が動かないこと。"""
        from ai.inference.confidence import BASE_POINTS, points_for_confidence

        assert points_for_confidence("複勝", None) == BASE_POINTS
        assert points_for_confidence("単勝", None) == BASE_POINTS

    def test_combos_are_one_point_each(self):
        """連系は 1 組合せ = 1 点。何点買うかは的中確率の下限が決める。"""
        from ai.inference.confidence import points_for_confidence

        # 券種が基準表に無いので base のまま返る (呼び出し側は渡さない)
        assert points_for_confidence("馬連", 0.9) == points_for_confidence("馬連", 0.01)

    def test_stake_is_points_times_100(self):
        cands = [
            BetCandidate(
                bet_type="複勝", combo="1", pattern="box", prob=0.6,
                est_odds=1.5, ev=0.9, stake=0, post_positions=[1],
            )
        ]
        out = assign_flat_stakes(
            cands, race_budget=10_000, points_by_bet_type={"複勝": 7}
        )
        assert [c.stake for c in out] == [700]
