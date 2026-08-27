"""払戻と無関係な部分を損失から外す 2 つの仕組み。

- plackett_luce_loss(top_k): 払戻は 3 着までしか出ないのに全順列を当てにいくと、
  平均 14 頭のこのデータでは段の 75% が payout と無関係になる。
- place_growth_loss(temp): 実運用は score 最大の 1 頭に賭けるのに、温度 1.0 の
  softmax は配分を全馬に散らす。低温で argmax に近づける。
"""

from __future__ import annotations

import torch

from ai.model.loss import place_growth_loss, plackett_luce_loss

MASK = torch.ones(1, 5, dtype=torch.bool)
SCORES = torch.tensor([[3.0, 2.0, 1.0, 0.0, -1.0]])


class TestTruncatedPlackettLuce:
    def test_ignores_the_order_below_top_k(self):
        """4 着と 5 着を入れ替えても top_k=3 なら損失が変わらないこと。"""
        pos_a = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        pos_b = torch.tensor([[1.0, 2.0, 3.0, 5.0, 4.0]])
        a = plackett_luce_loss(SCORES, pos_a, MASK, top_k=3)
        b = plackett_luce_loss(SCORES, pos_b, MASK, top_k=3)
        assert torch.isclose(a, b)

    def test_full_permutation_does_penalise_that_swap(self):
        """既定 (top_k=None) は下位の入れ替えも罰する = 従来の挙動。"""
        pos_a = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        pos_b = torch.tensor([[1.0, 2.0, 3.0, 5.0, 4.0]])
        assert not torch.isclose(
            plackett_luce_loss(SCORES, pos_a, MASK),
            plackett_luce_loss(SCORES, pos_b, MASK),
        )

    def test_still_penalises_the_top_order(self):
        """打ち切っても上位の順序は罰する (無視してよいのは下位だけ)。"""
        good = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        bad = torch.tensor([[3.0, 2.0, 1.0, 4.0, 5.0]])
        assert plackett_luce_loss(SCORES, good, MASK, top_k=3) < plackett_luce_loss(
            SCORES, bad, MASK, top_k=3
        )

    def test_top_k_larger_than_field_is_the_full_loss(self):
        pos = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        assert torch.isclose(
            plackett_luce_loss(SCORES, pos, MASK, top_k=99),
            plackett_luce_loss(SCORES, pos, MASK),
        )


class TestPlaceGrowthTemperature:
    def test_lower_temperature_concentrates_on_the_top_pick(self):
        """1 位だけが払戻を持つとき、低温のほうが損失が小さい (= 集中が報われる)。"""
        ret = torch.tensor([[1.5, 0.0, 0.0, 0.0, 0.0]])
        assert place_growth_loss(SCORES, ret, MASK, temp=0.5) < place_growth_loss(
            SCORES, ret, MASK, temp=1.0
        )

    def test_lower_temperature_punishes_a_wrong_top_pick(self):
        """逆に 1 位が外れているときは、低温のほうが損失が大きい。"""
        ret = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.5]])
        assert place_growth_loss(SCORES, ret, MASK, temp=0.5) > place_growth_loss(
            SCORES, ret, MASK, temp=1.0
        )

    def test_races_without_any_payout_are_skipped(self):
        ret = torch.zeros(1, 5)
        assert torch.isnan(place_growth_loss(SCORES, ret, MASK))
