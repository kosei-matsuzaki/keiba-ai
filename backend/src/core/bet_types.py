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


#: 連系を買う **的中確率の下限** (券種ごと)。**連系はこれだけで決まる。**
#: 線を超えた買い目を全部買うので、買う点数がレースごとに変わる。
#:
#: 券種で値が違うのは的中確率の桁が違うため (ワイドの本命は 21% 前後、三連単は
#: 1.5% 前後)。値は OOF 14,697 レースの券種別中央値の 1.25 倍。
#:
#: 実測 (前進検証 9 fold・2019-10〜2024-10・連系のみ・1 点 100 円・上限なし):
#:
#:   下限 0.75x   0.872   7.29 点/レース   fold 0.772〜0.942
#:   下限 1.0x    0.864   5.47 点/レース   fold 0.784〜0.961
#:   **下限 1.25x  0.867   4.23 点/レース   fold 0.798〜0.979**  ← これ
#:   下限 1.5x    0.841   3.35 点/レース
#:   下限 2.0x    0.853   2.24 点/レース
#:   下限 3.0x    0.862   1.16 点/レース
#:
#: **どこに置いても 0.84〜0.87 で頭打ち**なので、1.25 倍を選んだのは回収率ではなく
#: 「同じ成績を最も少ない点数で得られる」から。参考までに、券種ごとに上位 2 点で
#: 打ち切る旧実装は 0.887 (3.19 点/レース) で **これより 0.02 高い** が、
#: 「券種ごとに何点まで」というルールベースの上限は使わない方針にした
#: (ワイドが効くレース・三連単が効くレースを一律の点数で潰してしまうため)。
#:
#: 注意: この定数は評価に使ったのと同じ OOF から採っている (中央値という分布の
#: 目印であって ROI の最適化はしていないが、完全な out-of-sample ではない)。
DEFAULT_COMBO_MIN_HIT_PROB: dict[str, float] = {
    "馬連": 0.075,
    "ワイド": 0.260,
    "馬単": 0.025,
    "三連複": 0.024,
    "三連単": 0.019,
}


def normalize_combo(combo: str) -> str:
    """払戻テーブルの combo 表記を、推論・買い目側の表記に揃える。

    netkeiba の払戻 HTML から入る combo は ``10 - 14`` / ``14 → 10`` と
    **区切りの前後に空白**が入る。一方 `predict_race_with_combinations` が
    作る買い目は空白なし (``10-14`` / ``14→10``) で、`bet_records.combo` にも
    そのまま入る。**素の文字列比較では連系が 1 つも一致しない**ので、
    突き合わせる側は必ずここを通す。

    実害: 決済 (`services.bet_settlement`) が連系をすべて外れとして確定し、
    答え合わせ・収支が嘘になる。空白を全部落とすだけで両表記が一致する。
    """
    return "".join(combo.split())
