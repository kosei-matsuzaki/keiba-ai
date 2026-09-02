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

    **``fit`` (payback グリッド探索) と ``fit_calibration`` (NLL) は目的が違う。**
    払戻を最大化する T を選ぶと、グリッド端 (0.1 / 10.0) に張り付いて確率が壊れる。
    実際 2026-08-23 時点の本番モデルは T_win=0.133 / T_place=10.0 で、win_prob が
    1 位に 0.999999 乗り (画面に「単勝確率 100.0%」と出る)、place_prob はほぼ一様、
    という**互いに矛盾した確率**を返していた。表示する確率は ``fit_calibration``
    (勝ち馬の NLL 最小化 = proper scoring rule) で決めること。賭ける/賭けないの
    判定は温度ではなく買い方のルール側で表現する (docs/ai-model.md「推奨ベットルール」)。
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

    def fit_calibration(
        self,
        scores_per_race: list[np.ndarray],
        finish_positions_per_race: list[np.ndarray],
        T_candidates: np.ndarray | None = None,
    ) -> float:
        """勝ち馬の負の対数尤度を最小化する T を選び、T_win / T_place に同じ値を入れる。

        payback 最大化 (:meth:`fit`) と違い、これは **確率を正直にする**ための較正。
        単勝の softmax と複勝の PL に同じ T を使うので、「単勝は 100% と言うのに
        複勝はほぼ一様」という矛盾が構造的に起きない。

        Args:
            scores_per_race: 1 レース 1 配列のスコア。
            finish_positions_per_race: 同形状の着順 (1-based, NaN 可)。
            T_candidates: 探索する温度。default は np.geomspace(0.1, 20.0, 80)。

        Returns:
            選ばれた T (= T_win = T_place)。勝ち馬のいるレースが 0 件なら 1.0。
        """
        if T_candidates is None:
            T_candidates = np.geomspace(0.1, 20.0, 80)

        pairs: list[tuple[np.ndarray, int]] = []
        for s, pos in zip(scores_per_race, finish_positions_per_race, strict=True):
            if s is None or pos is None or len(s) < 2:
                continue
            idx = np.where(np.asarray(pos, dtype=float) == 1.0)[0]
            if idx.size == 0:
                continue
            pairs.append((np.asarray(s, dtype=float), int(idx[0])))

        if not pairs:
            return 1.0

        best_T, best_nll = 1.0, float("inf")
        for T in T_candidates:
            total = 0.0
            for s, w in pairs:
                p = _softmax_with_temperature(s, float(T))
                total -= float(np.log(max(p[w], 1e-12)))
            nll = total / len(pairs)
            if nll < best_nll:
                best_nll, best_T = nll, float(T)

        self.T_win = best_T
        self.T_place = best_T
        return best_T
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
    for scores, positions, odds in zip(
        scores_per_race, finish_positions_per_race, odds_win_per_race, strict=True,
    ):
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
    from ai.core.probabilities import plackett_luce_place_prob

    invested = 0.0
    gross = 0.0
    rng = np.random.default_rng(42)

    for scores, positions, payout_map in zip(
        scores_per_race, finish_positions_per_race, payout_place_per_race, strict=True,
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
