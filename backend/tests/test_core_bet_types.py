"""core.bet_types — 予測できる馬券種への正規化。"""

from __future__ import annotations

from core.bet_types import (
    COMBINATION_BET_TYPES,
    DEFAULT_ENABLED_BET_TYPES,
    supported_bet_types,
)


class TestSupportedBetTypes:
    def test_drops_waku_ren(self):
        """枠連はオッズ・払戻はあるが AI が買い目を生成しないので落とす。

        設定に残っていると「選べるのに何も起きない」死んだ選択肢になる。
        """
        assert "枠連" not in COMBINATION_BET_TYPES
        assert supported_bet_types(["単勝", "枠連", "馬連"]) == ["単勝", "馬連"]

    def test_keeps_order(self):
        assert supported_bet_types(["三連単", "単勝"]) == ["三連単", "単勝"]

    def test_falls_back_when_everything_is_dropped(self):
        """全部落ちると 1 点も買えない設定になるので既定に戻す。"""
        assert supported_bet_types(["枠連"]) == list(DEFAULT_ENABLED_BET_TYPES)

    def test_falls_back_on_non_list(self):
        assert supported_bet_types(None) == list(DEFAULT_ENABLED_BET_TYPES)
        assert supported_bet_types("単勝") == list(DEFAULT_ENABLED_BET_TYPES)

    def test_unknown_strings_are_dropped(self):
        assert supported_bet_types(["単勝", "存在しない券種"]) == ["単勝"]
