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
            "horse_id": ["C", "A", "B"], "win_prob": [0.5, 0.3, 0.2],
        }))
        assert conf.pick_confidence(SimpleNamespace(), frame, "A") == pytest.approx(0.3)
        assert conf.pick_confidence(SimpleNamespace(), frame, "C") == pytest.approx(0.5)

    def test_unknown_horse_returns_none(self, frame, monkeypatch):
        monkeypatch.setattr(conf, "predict_race", lambda *a, **kw: pd.DataFrame({
            "horse_id": ["A"], "win_prob": [0.4],
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
