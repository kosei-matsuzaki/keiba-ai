"""Per-bet-type temperature scaling for post-hoc probability calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TemperatureScaler:
    """馬券種別に softmax 温度を保持する 1 パラメータの確率補正器。

    T_win: 単勝確率の温度 (softmax(score / T_win) が win_prob になる)
    T_place: 複勝確率計算時に PL モンテカルロに渡すスコアの温度
            (softmax(score / T_place) を分布として扱い、PL サンプリングのシャープさを制御)

    T > 1 → 分布が平坦化 (賭けすぎ抑制)
    T < 1 → 分布が鋭利化 (max 確率が上がる)
    T = 1 → 補正なし (恒等)
    """

    T_win: float = 1.0
    T_place: float = 1.0

    def fit(
        self,
        scores_per_race: list[np.ndarray],
        finish_positions_per_race: list[np.ndarray],
        odds_win_per_race: list[np.ndarray],
        payout_place_per_race: list[dict[int, int] | None],
        ev_threshold_win: float = 1.1,
        ev_threshold_place: float = 1.05,
        T_candidates: np.ndarray | None = None,
    ) -> None:
        """馬券種別に payback を最大化する T を 1D grid search で選ぶ。

        Args:
            scores_per_race: 1 レース 1 配列のスコア (順位モデル出力)
            finish_positions_per_race: 同形状の着順 (1-based int, NaN 可)
            odds_win_per_race: 同形状の単勝オッズ (NaN 可)
            payout_place_per_race: 各レースの {finish_position: payout_yen} 辞書 (None 可)
            ev_threshold_win: 単勝 EV 閾値 (賭け判定に使う、evaluate.py と同じ)
            ev_threshold_place: 複勝 EV 閾値
            T_candidates: 探索する温度値。default は np.geomspace(0.1, 10.0, 50)
        """
        if T_candidates is None:
            T_candidates = np.geomspace(0.1, 10.0, 50)

        # Grid search for T_win: maximize payback_win across all races
        best_T_win = 1.0
        best_payback_win = -1.0
        for T in T_candidates:
            payback = _eval_payback_win(
                scores_per_race, finish_positions_per_race, odds_win_per_race,
                T=T, ev_threshold=ev_threshold_win,
            )
            if payback > best_payback_win:
                best_payback_win = payback
                best_T_win = float(T)

        # Grid search for T_place: maximize payback_place across all races
        best_T_place = 1.0
        best_payback_place = -1.0
        for T in T_candidates:
            payback = _eval_payback_place(
                scores_per_race, finish_positions_per_race, payout_place_per_race,
                T=T, ev_threshold=ev_threshold_place,
            )
            if payback > best_payback_place:
                best_payback_place = payback
                best_T_place = float(T)

        self.T_win = best_T_win
        self.T_place = best_T_place

    def transform_win(self, scores: np.ndarray) -> np.ndarray:
        """softmax(scores / T_win) を返す (1 レース内)."""
        scaled = scores / self.T_win
        shifted = scaled - scaled.max()
        exp_s = np.exp(shifted)
        return exp_s / exp_s.sum()

    def transform_place_scores(self, scores: np.ndarray) -> np.ndarray:
        """scores / T_place を返す (PL モンテカルロに渡す用)."""
        return scores / self.T_place


def _softmax_with_temperature(scores: np.ndarray, T: float) -> np.ndarray:
    """softmax(scores / T) を数値安定に計算する。"""
    scaled = scores / T
    shifted = scaled - scaled.max()
    exp_s = np.exp(shifted)
    return exp_s / exp_s.sum()


def _eval_payback_win(
    scores_per_race: list[np.ndarray],
    finish_positions_per_race: list[np.ndarray],
    odds_win_per_race: list[np.ndarray],
    T: float,
    ev_threshold: float,
) -> float:
    """温度 T での単勝 payback を計算して返す。ベット 0 件なら -1.0 を返す。"""
    invested = 0.0
    gross = 0.0
    for scores, positions, odds in zip(scores_per_race, finish_positions_per_race, odds_win_per_race):
        win_probs = _softmax_with_temperature(scores, T)
        for i in range(len(scores)):
            o = float(odds[i]) if i < len(odds) else float("nan")
            if np.isnan(o):
                continue
            ev = win_probs[i] * o
            if ev > ev_threshold:
                invested += 1.0
                pos = float(positions[i]) if i < len(positions) else float("nan")
                if not np.isnan(pos) and int(pos) == 1:
                    gross += o
    if invested <= 0:
        return -1.0
    return gross / invested


def _eval_payback_place(
    scores_per_race: list[np.ndarray],
    finish_positions_per_race: list[np.ndarray],
    payout_place_per_race: list[dict[int, int] | None],
    T: float,
    ev_threshold: float,
) -> float:
    """温度 T での複勝 payback を計算して返す。ベット 0 件なら -1.0 を返す。

    PL 複勝 prob は temperature 済みスコアの Gumbel-top-3 サンプリングで推定する。
    ここではコスト削減のため近似として softmax top-k mass を使う。
    """
    from ai.calibrate import plackett_luce_place_prob

    invested = 0.0
    gross = 0.0
    rng = np.random.default_rng(42)

    for scores, positions, payout_map in zip(
        scores_per_race, finish_positions_per_race, payout_place_per_race
    ):
        if payout_map is None or len(payout_map) == 0:
            continue

        # Temperature-scaled scores for PL sampling
        scaled_scores = scores / T
        place_probs = plackett_luce_place_prob(scaled_scores, k=3, n_samples=2_000, rng=rng)

        min_payout = min(payout_map.values())
        min_odds = min_payout / 100.0

        for i in range(len(scores)):
            ev = place_probs[i] * min_odds
            if ev > ev_threshold:
                invested += 1.0
                pos_val = float(positions[i]) if i < len(positions) else float("nan")
                if not np.isnan(pos_val) and int(pos_val) in payout_map:
                    gross += payout_map[int(pos_val)]
    if invested <= 0:
        return -1.0
    return gross / invested
