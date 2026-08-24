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

def supported_bet_types(bet_types: object) -> list[str]:
    """予測できる馬券種だけに絞る。

    **枠連は当 AI の予測対象外**（`COMBINATION_BET_TYPES` に無い）。払戻・オッズは
    取得しているので DB や設定に文字列としては現れうるが、買い目候補は 1 件も
    生成されない。設定に残っていると「選べるのに何も起きない」死んだ選択肢になる
    ので、読み込み時にここで落とす。

    Args:
        bet_types: 設定から読んだ値 (list 以外・None も想定)。

    Returns:
        COMBINATION_BET_TYPES に含まれるものだけを元の順で。空になったら
        DEFAULT_ENABLED_BET_TYPES にフォールバックする (1 つも買えない設定を
        作らないため)。
    """
    if not isinstance(bet_types, (list, tuple)):
        return list(DEFAULT_ENABLED_BET_TYPES)
    kept = [b for b in bet_types if b in COMBINATION_BET_TYPES]
    return kept or list(DEFAULT_ENABLED_BET_TYPES)


# UI のデフォルト有効馬券種 (settings.json 初期値 + 未設定時のフォールバック)
DEFAULT_ENABLED_BET_TYPES: tuple[str, ...] = (
    "単勝",
    "複勝",
    "ワイド",
    "馬連",
)
