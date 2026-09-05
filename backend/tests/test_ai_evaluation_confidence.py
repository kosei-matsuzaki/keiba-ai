"""backtest が実運用と同じ複勝ルールで測ること。

Dashboard の KPI は `backtest --persist` が書いた値を読む。評価側だけ確率モデルを
知らないと、**画面の数字が「利用者が実際に得る数字」でなくなる**。
2026-08-24 に同じ型のズレ（EV 条件の有無）を一度直しており、確率モデルの導入で
再発しないよう固定する。
"""

from __future__ import annotations

import inspect

from ai.evaluation import backtest
from core import settings_store


def test_evaluate_accepts_the_probability_model():
    sig = inspect.signature(backtest.evaluate)
    assert "probability_model_path" in sig.parameters
    assert "place_min_confidence" in sig.parameters


def test_place_confidence_defaults_match_deployment():
    """既定値が実運用の既定と一致すること。

    リテラルではなく settings の既定そのものと突き合わせる。旧実装は 0.30 を
    書き写していて、鍵が place_min_hit_prob (3 着内率・0.60) に変わったあとも
    テストごと旧目盛りに固定されていた。3 着内率の 0.30 はほぼ全レースが通る
    ので、既定のまま evaluate() を直接呼ぶと黙ってフィルタが消える。
    """
    sig = inspect.signature(backtest.evaluate)
    deployed = settings_store._DEFAULTS["place_min_hit_prob"]
    assert sig.parameters["place_min_confidence"].default == deployed


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


def test_win_min_odds_default_matches_deployment():
    """単勝のオッズ下限の既定も実運用と一致すること。

    place_min_confidence と同じ形で腐りうる。ここがずれても着順精度は変わらず、
    回収率だけが静かに動くのでテストからも結果からも見えない。
    """
    sig = inspect.signature(backtest.evaluate)
    deployed = settings_store._DEFAULTS["win_min_odds"]
    assert sig.parameters["win_min_odds"].default == deployed


def test_evaluate_callers_pass_the_settings_value():
    """`evaluate()` を呼ぶ本番経路は win_min_odds を settings から渡すこと。

    2026-09-05 まで Dashboard の「計測」と CLI が既定の 1.1 で走っていて、
    Settings で単勝のオッズ下限を変えても KPI だけ追従しなかった。
    """
    from api.routers import models as models_router

    src = inspect.getsource(models_router.evaluate_model)
    assert "win_min_odds=bet_settings.win_min_odds" in src

    cli = inspect.getsource(backtest._cli)
    assert "else _bet_settings.win_min_odds" in cli
    assert "win_min_odds=win_min_odds," in cli
