"""Tests for ai/temperature.py — TemperatureScaler."""

from __future__ import annotations

import numpy as np

from ai.core.temperature import TemperatureScaler


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max()
    exp_s = np.exp(shifted)
    return exp_s / exp_s.sum()


class TestTransformWin:
    def test_identity_at_T_one(self):
        scores = np.array([1.0, 2.0, 3.0, 0.5])
        ts = TemperatureScaler(T_win=1.0, T_place=1.0)
        result = ts.transform_win(scores)
        expected = _softmax(scores)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_T_gt_one_flattens_distribution(self):
        """T > 1 makes max probability smaller (distribution more uniform)."""
        scores = np.array([3.0, 1.0, 0.5, 0.1])
        ts_base = TemperatureScaler(T_win=1.0)
        ts_high = TemperatureScaler(T_win=5.0)
        result_base = ts_base.transform_win(scores)
        result_high = ts_high.transform_win(scores)
        # Max prob should decrease when T increases
        assert result_high.max() < result_base.max()

    def test_T_lt_one_sharpens_distribution(self):
        """T < 1 makes max probability larger (distribution more peaked)."""
        scores = np.array([3.0, 1.0, 0.5, 0.1])
        ts_base = TemperatureScaler(T_win=1.0)
        ts_low = TemperatureScaler(T_win=0.5)
        result_base = ts_base.transform_win(scores)
        result_low = ts_low.transform_win(scores)
        assert result_low.max() > result_base.max()

    def test_output_sums_to_one(self):
        scores = np.random.default_rng(42).standard_normal(10)
        for T in [0.3, 1.0, 3.0]:
            ts = TemperatureScaler(T_win=T)
            result = ts.transform_win(scores)
            assert abs(result.sum() - 1.0) < 1e-9

    def test_output_non_negative(self):
        scores = np.array([5.0, -2.0, 0.0, 1.0])
        ts = TemperatureScaler(T_win=2.0)
        result = ts.transform_win(scores)
        assert (result >= 0).all()


class TestTransformPlaceScores:
    def test_identity_at_T_one(self):
        scores = np.array([1.0, 2.0, 3.0])
        ts = TemperatureScaler(T_place=1.0)
        result = ts.transform_place_scores(scores)
        np.testing.assert_array_equal(result, scores)

    def test_scaling_by_temperature(self):
        scores = np.array([2.0, 4.0, 6.0])
        ts = TemperatureScaler(T_place=2.0)
        result = ts.transform_place_scores(scores)
        expected = scores / 2.0
        np.testing.assert_allclose(result, expected)


class TestFit:
    def _make_synthetic_races(self, n_races: int = 20, n_horses: int = 8, seed: int = 0):
        """Generate synthetic races with a clear favourite to make payback non-trivial."""
        rng = np.random.default_rng(seed)
        scores_list = []
        positions_list = []
        odds_list = []
        payout_list = []

        for _ in range(n_races):
            # Scores: first horse is strongest
            scores = rng.standard_normal(n_horses)
            scores[0] += 2.0  # give first horse a boost
            scores_list.append(scores)

            # Assign finish positions by rank of scores + noise
            rank_scores = scores + rng.standard_normal(n_horses) * 0.5
            order = np.argsort(-rank_scores)
            positions = np.empty(n_horses)
            for pos, idx in enumerate(order):
                positions[idx] = pos + 1
            positions_list.append(positions)

            # Odds: inversely related to softmax probs (simplified)
            probs = _softmax(scores)
            # Simple odds: 1/prob * 0.75 (track take)
            odds = np.clip(0.75 / probs, 1.1, 100.0)
            odds_list.append(odds)

            # Payout place: based on top-3 finishers
            payout = {1: 120, 2: 150, 3: 180}
            payout_list.append(payout)

        return scores_list, positions_list, odds_list, payout_list

    def test_fit_runs_without_error(self):
        scores, positions, odds, payouts = self._make_synthetic_races(seed=0)
        ts = TemperatureScaler()
        ts.fit(
            scores_per_race=scores,
            finish_positions_per_race=positions,
            odds_win_per_race=odds,
            payout_place_per_race=payouts,
            T_candidates=np.geomspace(0.5, 5.0, 10),
        )
        assert ts.T_win > 0
        assert ts.T_place > 0

    def test_fit_selects_temperature_from_candidates(self):
        """Fitted T must come from the supplied candidate set."""
        scores, positions, odds, payouts = self._make_synthetic_races(seed=1)
        candidates = np.geomspace(0.5, 5.0, 5)
        ts = TemperatureScaler()
        ts.fit(
            scores_per_race=scores,
            finish_positions_per_race=positions,
            odds_win_per_race=odds,
            payout_place_per_race=payouts,
            T_candidates=candidates,
        )
        assert any(abs(ts.T_win - c) < 1e-9 for c in candidates), (
            f"T_win={ts.T_win} not in candidates {candidates}"
        )
        assert any(abs(ts.T_place - c) < 1e-9 for c in candidates), (
            f"T_place={ts.T_place} not in candidates {candidates}"
        )

    def test_fit_with_empty_payout_maps(self):
        """Fitting should not crash when all payout_place entries are None."""
        scores, positions, odds, _ = self._make_synthetic_races(seed=2)
        payouts = [None] * len(scores)
        ts = TemperatureScaler()
        ts.fit(
            scores_per_race=scores,
            finish_positions_per_race=positions,
            odds_win_per_race=odds,
            payout_place_per_race=payouts,
            T_candidates=np.geomspace(0.5, 5.0, 5),
        )
        # T_win should still be selected; T_place defaults to 1.0 (no valid races)
        assert ts.T_win > 0

    def test_fit_random_scores_does_not_crash(self):
        """Even with fully random scores (no predictive signal), fit should complete."""
        rng = np.random.default_rng(99)
        n_races = 15
        n_horses = 8
        scores = [rng.standard_normal(n_horses) for _ in range(n_races)]
        positions = []
        for _ in range(n_races):
            pos = rng.permutation(n_horses) + 1
            positions.append(pos.astype(float))
        odds = [np.full(n_horses, 5.0) for _ in range(n_races)]
        payouts = [None] * n_races

        ts = TemperatureScaler()
        ts.fit(
            scores_per_race=scores,
            finish_positions_per_race=positions,
            odds_win_per_race=odds,
            payout_place_per_race=payouts,
            T_candidates=np.array([0.5, 1.0, 2.0]),
        )
        assert ts.T_win in (0.5, 1.0, 2.0)

    def test_default_T_values(self):
        ts = TemperatureScaler()
        assert ts.T_win == 1.0
        assert ts.T_place == 1.0
