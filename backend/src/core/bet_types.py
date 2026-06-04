"""Centralized bet-type string constants.

予測 / シミュレーション / calibration / API / UI 設定が同じ馬券種文字列を
別々にハードコードしていたので、ここを単一の出典 (single source of truth) に
まとめる。schemas.py の `BetType = Literal[...]` は PEP 586 上、文字列リテラル
を直接書く必要があるため変数参照に置き換えられない (そちらだけ二重定義のまま
残る)。
"""

from __future__ import annotations

# predict_race_with_combinations が返す bet_type キー集合。枠連は払戻には
# 出現するが当 AI の組合せ予測対象外なので含めない。
COMBINATION_BET_TYPES: tuple[str, ...] = (
    "単勝",
    "複勝",
    "馬連",
    "ワイド",
    "馬単",
    "三連複",
    "三連単",
)

# 連系のみ (combo calibrator 学習・診断で使う)
RENKEI_BET_TYPES: tuple[str, ...] = (
    "馬連",
    "ワイド",
    "馬単",
    "三連複",
    "三連単",
)

# UI のデフォルト有効馬券種 (settings.json 初期値 + 未設定時のフォールバック)
DEFAULT_ENABLED_BET_TYPES: tuple[str, ...] = (
    "単勝",
    "複勝",
    "ワイド",
    "馬連",
)
