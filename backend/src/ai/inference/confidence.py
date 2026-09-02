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


#: 券種 → 確率モデルのどの確率を「確信度」とするか。
#:
#: **確信度は券種をまたいで同じ意味にする**: 「その買い目が当たる確率」。
#: 単勝なら 1 着になる確率、複勝なら 3 着以内に入る確率、連系なら組合せの的中確率
#: (連系は `merge_combination_sources` が既に確率モデル由来の値を入れている)。
#:
#: 以前は複勝の判定にも **1 着確率** を使っていた。前進検証で効くことは確かめて
#: あったが、「複勝を買うかを 1 着確率で決める」は意味が通らず、画面に出しても
#: 読み手が解釈できない。3 着内率に替えても成績は同等 (OOF 14,619 レース、
#: 買う割合 25% で 0.904 → 0.907、10% で 0.925 → 0.930)。
_CONFIDENCE_COLUMN: dict[str, str] = {"単勝": "win_prob", "複勝": "place_prob"}


def pick_confidence(
    prob_bundle: ModelBundle,
    frame: pd.DataFrame,
    horse_id: str,
    session: Session | None = None,
    bet_type: str = "複勝",
) -> float | None:
    """確率モデルから見た「その買い目が当たる確率」。取れなければ None。

    Args:
        prob_bundle: proper scoring rule で学習したモデル (`loss_type="plackett_luce"`)。
        frame: 1 レース分の feature frame。
        horse_id: **active が選んだ馬**。確率モデル自身の本命ではない。
        session: 履歴 GRU 用。**必ず渡すこと** (None だと履歴が zero に degrade する)。
        bet_type: 単勝なら 1 着確率、複勝 (既定) なら 3 着内率を返す。

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
    column = _CONFIDENCE_COLUMN.get(bet_type, "place_prob")
    if column not in row.columns:
        return None
    value = float(row.iloc[0][column])
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


#: 確信度から点数を出すときの基準 (券種ごと)。**確信度の桁が券種で違う**ため
#: 1 つの基準では表せない: 単勝の確信度は 1 着確率 (中央値 0.18)、複勝は 3 着内率
#: (0.5 前後)。基準はその券種の典型値に置き、そこで base 点になるようにする。
#:
#: 連系はここに載せない。連系の「点数」は 1 組合せ = 1 点で、**何点買うかは
#: 的中確率の下限を超えた買い目の数**が決める (`DEFAULT_COMBO_MIN_HIT_PROB`)。
CONFIDENCE_REFERENCE: dict[str, float] = {"単勝": 0.25, "複勝": 0.50}

#: 基準の確信度のときに買う点数 (1 点 = 100 円)。
BASE_POINTS = 5

#: 点数の上限。確信度が高くても 1 レースに突っ込みすぎないための歯止め。
MAX_POINTS = 15

#: 確信度に対する反応の強さ。2 = 二乗。
CONFIDENCE_EXPONENT = 2


def points_for_confidence(
    bet_type: str,
    confidence: float | None,
    base_points: int = BASE_POINTS,
) -> int:
    """確信度に応じた点数 (1 点 = 100 円)。

    ``base × (確信度 / 基準)^2`` を 1〜15 点に丸める。実測 (前進検証 9 fold・
    OOF 14,700 レース):

    **複勝** (基準 0.50 = 3 着以内が半々の位置) — 効く。

        定額 5 点          0.875   3.80 点/レース
        段階 x1/x2/x3      0.882   8.87 点/レース   ← 旧実装
        連続 (p/0.5)^2     0.891   5.42 点/レース   ← これ

    5 年すべてでプラス (+0.037 / +0.007 / +0.019 / +0.008 / +0.016)。

    **単勝** (基準 0.25) — ほぼ効かないが害もない。

        定額 5 点              0.8438   5.00 点/レース   fold 0.767〜0.955
        連続 (p/0.25)^2        0.8483   4.66 点/レース   fold 0.776〜0.913

    回収率の差 +0.005 は誤差の範囲で、確信度と回収率の相関も −0.005 しかない
    (的中率は 6% → 42% と動くのに回収率が動かない = 市場が正しく値付けしている)。
    それでも同じ式にしているのは、**券種ごとに賭け金の決め方が違う理由が無い**
    のと、fold ごとのばらつきが小さくなるため。

    確信度が取れないとき (確率モデル未設定・推論失敗) は基準の点数。壊れたときに
    賭け金が動くと挙動が読めない。
    """
    reference = CONFIDENCE_REFERENCE.get(bet_type)
    if reference is None or confidence is None:
        return base_points
    raw = round(base_points * (confidence / reference) ** CONFIDENCE_EXPONENT)
    return max(1, min(MAX_POINTS, int(raw)))
