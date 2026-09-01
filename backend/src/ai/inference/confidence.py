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


#: 確信度に対する賭け金の反応。**券種で形が違うのは実測に基づく。**
#:
#: OOF 14,619 レース (2020-05〜2024-10) を確信度で 5 分位に割ったときの回収率:
#:
#:   単勝 (1着確率)  0.875 / 0.923 / 0.851 / 0.843 / 0.859   相関 −0.005
#:   複勝 (3着内率)  0.749 / 0.843 / 0.870 / 0.875 / 0.911   相関 +0.041
#:
#: **単勝は的中率が 6% → 37% と動くのに回収率が動かない** = 市場が正しく値付けして
#: いる。確信度で賭け金を動かしても取り分は増えない。複勝だけ単調に上がるので、
#: 複勝の点数だけを確信度に比例させる (連系も無相関: 0.877 → 0.879)。
PLACE_CONFIDENCE_REFERENCE = 0.50
PLACE_CONFIDENCE_EXPONENT = 2
PLACE_MAX_POINTS = 15


def points_for_confidence(
    bet_type: str,
    confidence: float | None,
    base_points: int,
) -> int:
    """確信度に応じた点数 (1 点 = stake_unit)。

    複勝は ``base × (確信度 / 0.50)^2`` を 1〜15 点に丸める。基準 0.50 は
    「3 着以内に入る確率が半々」の位置。実測 (OOF・基準 5 点):

        定額 5 点          0.875   3.80 点/レース
        段階 x1/x2/x3      0.882   8.87 点/レース   ← 旧実装
        連続 (p/0.5)^2     0.891   5.42 点/レース   ← これ

    **連続のほうが回収率が高く、使う点数も少ない。** 5 年すべてでプラス
    (+0.037 / +0.007 / +0.019 / +0.008 / +0.016)。

    単勝・連系は確信度で動かさない (上のコメントの実測どおり、回収率が反応しない)。
    確率が取れないときも基準のまま — 壊れたときに賭け金が動くと挙動が読めない。
    """
    if bet_type != "複勝" or confidence is None:
        return base_points
    raw = round(base_points * (confidence / PLACE_CONFIDENCE_REFERENCE) ** PLACE_CONFIDENCE_EXPONENT)
    return max(1, min(PLACE_MAX_POINTS, int(raw)))
