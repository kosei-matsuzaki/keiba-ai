"""merge_combination_sources — 単複は買うモデル、連系は確率モデル。

`_combinations_from_base` は渡された base_df から連系だけでなく単勝・複勝の候補も
作る。連系確率を確率モデルから出すつもりで base_df を丸ごと差し替えると、
`recommend_for_race` が選ぶ本命まで確率モデルのものに変わり、**黙って回収率の
低い側へ切り替わる** (確率モデルの本命は単勝 0.824 / active 0.933)。
"""

from __future__ import annotations

from ai.inference.predict import merge_combination_sources


def _c(tag: str) -> list[str]:
    return [tag]


class TestMergeCombinationSources:
    def test_win_and_place_come_from_the_bet_model(self):
        merged = merge_combination_sources(
            {"単勝": _c("active"), "複勝": _c("active"), "馬連": _c("active")},
            {"単勝": _c("prob"), "複勝": _c("prob"), "馬連": _c("prob")},
        )
        assert merged["単勝"] == _c("active")
        assert merged["複勝"] == _c("active")

    def test_exotics_come_from_the_probability_model(self):
        merged = merge_combination_sources(
            {"馬連": _c("active"), "三連単": _c("active")},
            {"馬連": _c("prob"), "三連単": _c("prob")},
        )
        assert merged["馬連"] == _c("prob")
        assert merged["三連単"] == _c("prob")

    def test_missing_win_place_in_the_bet_model_is_tolerated(self):
        """買うモデル側に単複が無ければ確率モデル側を残す (落とさない)。"""
        merged = merge_combination_sources({"馬連": _c("active")}, {"単勝": _c("prob")})
        assert merged["単勝"] == _c("prob")

    def test_inputs_are_not_mutated(self):
        a = {"単勝": _c("active")}
        b = {"単勝": _c("prob"), "馬連": _c("prob")}
        merge_combination_sources(a, b)
        assert b["単勝"] == _c("prob")
