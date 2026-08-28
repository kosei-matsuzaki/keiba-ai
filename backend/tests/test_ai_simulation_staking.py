"""賭け金の決め方（定額 / 複利）と、破産の検知。

既定を定額にしているのは、複利だと払戻 1.0 未満の券種を数百レース買った時点で
資産が尽き、賭け金が下限に張り付いて**以降を実質評価しなくなる**ため。
実際にこれで「連系は点数が少なく測定不能」と誤って結論し、docs にもそう書いていた
（定額で測り直したら 21,570 点あり CI [0.83, 0.93] と十分測れた）。
"""

from __future__ import annotations

import inspect

from ai.simulation import engine as sim_mod


def test_flat_is_the_default():
    sig = inspect.signature(sim_mod.simulate_active_model)
    assert sig.parameters["staking"].default == "flat"


def test_result_reports_races_that_could_not_be_bet():
    """破産を黙って隠さない。0 でなければ回収率は途中までしか測れていない。"""
    r = sim_mod.SimulationResult(
        window_start=None, window_end=None, model_path="x", strategy="balanced", budget=1
    )
    assert r.n_races_broke == 0
    assert "n_races_broke" in r.as_dict()


def test_staking_is_recorded_in_conditions():
    """後から「定額で測ったのか複利で測ったのか」を判別できること。"""
    src = inspect.getsource(sim_mod.simulate_active_model)
    assert '"staking": staking' in src


def test_flat_budget_does_not_depend_on_the_running_bankroll():
    """定額では 1 レースの予算が残資産に連動しないこと（これが要点）。"""
    src = inspect.getsource(sim_mod.simulate_active_model)
    # compound の枝だけが current_bankroll を使う
    flat_branch = src[src.index("if staking == \"compound\":") : src.index("if race_budget < _MIN_STAKE")]
    else_part = flat_branch[flat_branch.index("else:") :]
    assert "current_bankroll * max_stake_per_race_pct" not in else_part


def test_flat_budget_is_independent_of_the_bankroll():
    """定額の 1 レース予算が残資産を参照しないこと。

    最初の実装は `min(race_budget, current_bankroll)` で頭打ちにしており、
    初期資産 10 万円 / 1 レース 5,000 円なら 20 レースで尽きて**複利より早く
    破産していた**（実測: 1,703 レース中 1,612 で買えず）。定額は「戦略の回収率を
    測る」ための機能なので、賭け金を資金繰りから切り離す。
    """
    src = inspect.getsource(sim_mod.simulate_active_model)
    flat_branch = src[src.index("else:", src.index('if staking == "compound":')) :]
    flat_branch = flat_branch[: flat_branch.index("if race_budget < _MIN_STAKE")]
    assert "current_bankroll" not in flat_branch


def test_required_capital_is_reported():
    """「いくら用意すれば途中で止まらずに済んだか」を出す。

    定額では資産がマイナスになりうるので、その最小値から必要資金を出す。
    """
    r = sim_mod.SimulationResult(
        window_start=None, window_end=None, model_path="x", strategy="balanced", budget=1
    )
    d = r.as_dict()
    assert "required_capital" in d
    assert "trough_bankroll" in d
