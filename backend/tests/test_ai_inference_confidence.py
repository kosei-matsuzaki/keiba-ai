"""ai.inference.confidence — AI の本命に対する確信度。

確率専用モデル (proper scoring rule で学習) を「馬を選ぶ」ためではなく
「その馬を信じてよいか」を答えるために使う。実測の根拠はモジュールの docstring。
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from ai.inference import confidence as conf


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({"horse_id": ["A", "B", "C"], "post_position": [1, 2, 3]})


class TestPickConfidence:
    def test_returns_the_probability_of_the_requested_horse(self, frame, monkeypatch):
        """**確率モデル自身の本命ではなく、渡された馬**の確率を返すこと。"""
        monkeypatch.setattr(conf, "predict_race", lambda *a, **kw: pd.DataFrame({
            "horse_id": ["C", "A", "B"],
            "win_prob": [0.5, 0.3, 0.2],
            "place_prob": [0.8, 0.6, 0.4],
        }))
        # 既定は複勝 = 3 着内率。券種で見る確率が変わる (確信度は券種横断で
        # 「その買い目が当たる確率」と定義した)
        assert conf.pick_confidence(SimpleNamespace(), frame, "A") == pytest.approx(0.6)
        assert conf.pick_confidence(SimpleNamespace(), frame, "C") == pytest.approx(0.8)
        assert conf.pick_confidence(
            SimpleNamespace(), frame, "A", bet_type="単勝"
        ) == pytest.approx(0.3)

    def test_unknown_horse_returns_none(self, frame, monkeypatch):
        monkeypatch.setattr(conf, "predict_race", lambda *a, **kw: pd.DataFrame({
            "horse_id": ["A"], "win_prob": [0.4], "place_prob": [0.7],
        }))
        assert conf.pick_confidence(SimpleNamespace(), frame, "Z") is None

    def test_prediction_failure_returns_none(self, frame, monkeypatch):
        """推論が落ちても例外を投げない (賭けの選定を止めないため)。"""
        def _boom(*a, **kw):
            raise RuntimeError("model broken")
        monkeypatch.setattr(conf, "predict_race", _boom)
        assert conf.pick_confidence(SimpleNamespace(), frame, "A") is None

    def test_out_of_range_probability_returns_none(self, frame, monkeypatch):
        monkeypatch.setattr(conf, "predict_race", lambda *a, **kw: pd.DataFrame({
            "horse_id": ["A"], "win_prob": [1.5],
        }))
        assert conf.pick_confidence(SimpleNamespace(), frame, "A") is None


class TestIsPlaceWorthBuying:
    def test_at_or_above_threshold_buys(self):
        assert conf.is_place_worth_buying(0.30, 0.30) is True
        assert conf.is_place_worth_buying(0.45, 0.30) is True

    def test_below_threshold_skips(self):
        assert conf.is_place_worth_buying(0.29, 0.30) is False

    def test_missing_confidence_buys(self):
        """確率が取れないときは買う側に倒す。

        確信度は絞り込みの機能なので、壊れたときに賭けが止まると
        「設定していないのに挙動が変わる」ことになる。
        """
        assert conf.is_place_worth_buying(None, 0.30) is True


def test_points_for_confidence_scales_place_continuously():
    """複勝の点数は確信度に連続で反応する。段階 (x1/x2/x3) より回収率が高く点数も少ない。"""
    from ai.inference.confidence import points_for_confidence

    base = 5
    assert points_for_confidence("複勝", 0.50, base) == 5      # 基準
    assert points_for_confidence("複勝", 0.70, base) == 10     # 5 * 1.96
    assert points_for_confidence("複勝", 0.30, base) == 2      # 5 * 0.36
    assert points_for_confidence("複勝", 0.95, base) == 15     # 上限で頭打ち


def test_points_for_confidence_is_monotonic():
    """単調でないと「確信度が上がったのに賭け金が減る」が起きる。"""
    from ai.inference.confidence import points_for_confidence

    values = [points_for_confidence("複勝", c / 100, 5) for c in range(1, 100)]
    assert values == sorted(values)


def test_points_for_confidence_leaves_win_and_combo_flat():
    """単勝と連系は確信度で動かさない。

    OOF 実測で単勝は的中率が 6% → 37% と動くのに回収率が動かない (相関 −0.005)。
    市場が正しく値付けしている以上、賭け金を動かしても取り分は増えない。
    """
    from ai.inference.confidence import points_for_confidence

    for bet_type in ("単勝", "ワイド", "三連複"):
        assert points_for_confidence(bet_type, 0.9, 5) == 5
        assert points_for_confidence(bet_type, 0.1, 5) == 5


def test_points_for_confidence_without_probability_model():
    """確率が取れないときは基準のまま (壊れたときに賭け金が動かないように)。"""
    from ai.inference.confidence import points_for_confidence

    assert points_for_confidence("複勝", None, 5) == 5
