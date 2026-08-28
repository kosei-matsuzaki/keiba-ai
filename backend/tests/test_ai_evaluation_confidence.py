"""backtest が実運用と同じ複勝ルールで測ること。

Dashboard の KPI は `backtest --persist` が書いた値を読む。評価側だけ確率モデルを
知らないと、**画面の数字が「利用者が実際に得る数字」でなくなる**。
2026-08-24 に同じ型のズレ（EV 条件の有無）を一度直しており、確率モデルの導入で
再発しないよう固定する。
"""

from __future__ import annotations

import inspect

from ai.evaluation import backtest


def test_evaluate_accepts_the_probability_model():
    sig = inspect.signature(backtest.evaluate)
    assert "probability_model_path" in sig.parameters
    assert "place_min_confidence" in sig.parameters


def test_place_confidence_defaults_match_deployment():
    """既定値が実運用の既定 (0.30) と一致すること。"""
    sig = inspect.signature(backtest.evaluate)
    assert sig.parameters["place_min_confidence"].default == 0.30


def test_backtest_uses_the_shared_confidence_helpers():
    """判定ロジックを二重実装しないこと (実運用とズレる原因になる)。"""
    src = inspect.getsource(backtest)
    assert "is_place_worth_buying" in src
    assert "pick_confidence" in src


def test_metrics_record_which_confidence_setting_was_used():
    """どの条件で測った数字かを metrics に残す。"""
    src = inspect.getsource(backtest.evaluate)
    assert '"probability_model"' in src
    assert '"place_min_confidence"' in src
    assert '"n_place_skipped"' in src
