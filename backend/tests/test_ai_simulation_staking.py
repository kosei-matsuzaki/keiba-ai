"""賭け金は資金繰りに依存しない、という不変条件。

シミュレーションは **RACE 画面の予想を全レースでやったらどうなるか**を測るもので、
資産運用の再現ではない。だから初期資産・複利・破産という概念を持たない
(2026-09-01 に全廃)。

以前は初期資産から複利で回していたため、払戻 1.0 未満の券種を数百レース買うと
資産が尽き、賭け金が下限に張り付いて**以降を実質評価しなくなっていた**。
回収率は Σpayout/Σstake = 賭け金の重み付き平均なので、破産すると「早い時期の
大きい賭け金」に偏った数字になる。実際にこれで「連系は点数が少なく測定不能」と
誤って結論し、docs にもそう書いていた (定額で測り直したら十分測れた)。
"""

from __future__ import annotations

import inspect

from ai.simulation import engine as sim_mod


def test_no_bankroll_concept_in_the_signature():
    """初期資産・賭け金の決め方・戦略プリセットは引数から消えている。"""
    params = inspect.signature(sim_mod.simulate_active_model).parameters
    for gone in (
        "budget",
        "staking",
        "strategy",
        "max_stake_per_race_pct",
        "max_stake_per_race_yen",
        "exclude_low_information",
        "top_n_horses",
        # 券種ごとの 1 点あたり金額と点数上限も廃止した
        "stake_unit_by_bet_type",
        "max_points_per_bet_type",
    ):
        assert gone not in params, f"{gone} が残っている"
    assert "race_budget" in params


def test_race_budget_never_depends_on_running_profit():
    """1 レースの予算は累計損益を参照しない (これが要点)。"""
    src = inspect.getsource(sim_mod.simulate_active_model)
    # recommend_for_race に渡るのは引数の race_budget そのもの
    assert "race_budget=race_budget," in src
    assert "current_profit" in src  # 損益は積むが…
    assert "race_budget = " not in src  # …予算の計算には使わない


def test_profit_starts_at_zero_in_the_result():
    r = sim_mod.SimulationResult(
        window_start=None, window_end=None, model_path="x", race_budget=5_000
    )
    assert r.final_profit == 0
    assert r.peak_profit == 0
    assert r.trough_profit == 0
    d = r.as_dict()
    assert d["final_profit"] == 0
    assert "profit_timeseries" in d


def test_required_capital_is_reported():
    """どれだけ沈む時期があったかは残す (運用上の情報)。

    破産は起きない (賭け金が残高に依存しない) が、「途中で最大いくら
    マイナスだったか」は知りたい。
    """
    r = sim_mod.SimulationResult(
        window_start=None, window_end=None, model_path="x", race_budget=5_000
    )
    assert r.required_capital == 0
    assert "required_capital" in r.as_dict()


def test_conditions_record_the_budget_and_the_combo_rule():
    """後から「どの条件で測ったか」を判別できること。"""
    src = inspect.getsource(sim_mod.simulate_active_model)
    assert '"race_budget": race_budget' in src
    assert '"combo_min_hit_prob"' in src
