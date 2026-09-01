"""「AI の本命をどれくらい信じてよいか」を、確率専用モデルから出す。

**なぜ別のモデルが要るのか**: 本番の active は `log_growth` 系 (回収率) で学習されて
おり、順序は良いが**確率の大きさに意味が無い**。実測 (test 19ヶ月・5,390 レース) では
本命の `win_prob` と実際の勝敗の相関が **0.073** しかない (市場の実装確率は 0.354)。
決定志向の損失は順序だけを最適化し、magnitude に確率としての意味を持たせないため。

そこで proper scoring rule (`--loss plackett_luce`) で学習したモデルを別に用意し、
**active が選んだ馬に対してその確率を引く**。この確率は相関 0.267 と実際の勝敗を
まともに追う。用途は「買う馬を変えること」ではない (確率モデルに馬を選ばせると
人気馬に寄って回収率が落ちる: 複勝 0.881 < active の 0.893)。**active の選択は
そのままに、複勝を買うかどうかだけを決める。**

実測 (前進検証で作った 4.5 年分の out-of-sample 出力、15,073 点):

    しきい値   買う割合   複勝回収率   95%CI            複勝的中率
    (なし)     100%      0.866      [0.848,0.884]    0.501
    0.30        22%      0.907      [0.887,0.927]    0.744
    0.40        11%      0.931      [0.909,0.955]    0.798

9 fold すべてで基準を上回る (最低 0.861 / 最高 0.969)。**市場価格を固定しても効く**:
本命が 1.0〜3.0 倍のレース (4,386 件) だけで上下に割ると 0.933 vs 0.845 (差 +0.088) で、
「人気馬かどうか」の代理ではなく市場に無い情報を持っている。

なお単勝には効かない。単勝のレース選別は前進検証で信号が見つからなかった
(目的変数をシャッフルした対照が最良になる)。
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session

from ai.inference.predict import predict_race
from ai.model.registry import ModelBundle
from core.logging import get_logger

log = get_logger(__name__)


def pick_confidence(
    prob_bundle: ModelBundle,
    frame: pd.DataFrame,
    horse_id: str,
    session: Session | None = None,
) -> float | None:
    """確率モデルが ``horse_id`` に与える単勝確率。取れなければ None。

    Args:
        prob_bundle: proper scoring rule で学習したモデル (`loss_type="plackett_luce"`)。
        frame: 1 レース分の feature frame。
        horse_id: **active が選んだ馬**。確率モデル自身の本命ではない。
        session: 履歴 GRU 用。**必ず渡すこと** (None だと履歴が zero に degrade する)。

    Returns:
        0.0〜1.0 の確率、または算出できないとき None。
    """
    try:
        preds = predict_race(prob_bundle, frame, session=session)
    except Exception as exc:  # noqa: BLE001
        log.warning("confidence model failed: %s", exc)
        return None
    row = preds[preds["horse_id"] == horse_id]
    if row.empty:
        return None
    value = float(row.iloc[0]["win_prob"])
    return value if 0.0 <= value <= 1.0 else None


def is_place_worth_buying(confidence: float | None, threshold: float) -> bool:
    """複勝を買ってよいか。

    確率が取れない (確率モデル未設定・推論失敗) ときは **買う側に倒す**。
    確信度は絞り込みの機能であって、壊れたときに賭けが止まると
    「設定していないのに挙動が変わる」ことになるため。
    """
    if confidence is None:
        return True
    return confidence >= threshold


#: 確信度 → 複勝の 1 点額の倍率。**しきい値を超えた先も確信度で厚みを変える。**
#:
#: 前進検証 (OOF 14,619 レース・2020-05〜2024-10) の実測:
#:
#:   定額 (全レース)              0.850
#:   確信度 0.30 以上だけ定額       0.875   ← 従来
#:   段階 x1/x2/x3               0.882   ← これ
#:
#: 5 年すべてでプラス (+0.019 / +0.001 / +0.013 / +0.005 / +0.001)。効果は小さいが
#: 符号が安定している。**単勝と連系には使わない** — 単勝は確信度で絞ると回収率が
#: 下がり (0.870 → 0.854)、連系は無相関 (0.877 → 0.879)。
PLACE_STAKE_TIERS: tuple[tuple[float, int], ...] = ((0.55, 3), (0.40, 2), (0.0, 1))


def place_stake_multiplier(confidence: float | None) -> int:
    """複勝の 1 点額を何倍にするか。確率が取れないときは 1 倍 (従来どおり)。

    ここは「買うかどうか」ではなく「いくら賭けるか」。買うかどうかは
    ``is_place_worth_buying`` が ``place_min_confidence`` で決める。
    """
    if confidence is None:
        return 1
    for lo, mult in PLACE_STAKE_TIERS:
        if confidence >= lo:
            return mult
    return 1
