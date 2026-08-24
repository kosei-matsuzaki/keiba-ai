"""レース単位の「手元にどれだけ情報があるか」を測る。

新馬戦のように出走馬全員が初出走のレースでは、モデルが使える履歴特徴 (直近着順・
上がり・脚質・per-race 履歴 GRU) がすべて欠損し、実質「枠順・馬体重・騎手・血統・
オッズ」だけで予想することになる。同じモデルでも入力の質が別物なので、

  * シミュレーションでは除外できるようにする (`exclude_low_information`)
  * 実際の予想画面では「このレースは情報が少ない」と明示する

ために使う。**クラス名 (race_class == "新馬") では判定しない。** 未勝利戦にも初出走馬が
混ざるうえ、地方・海外からの転入など分類に現れないケースがあるため、実際に手元にある
過去走の本数で測る。実測 (test 19ヶ月・5,404 レース):

    クラス      過去走ゼロ率  平均出走数  レース数
    新馬            0.997      0.003       422
    未勝利          0.052      3.85      1,958
    1勝クラス       0.004      9.56      1,440
    OP              0.010     18.41        167

新馬とそれ以外の間には断絶があり、閾値の置き方に神経質にならなくてよい。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

#: 過去走ゼロの馬がこの割合以上なら「情報が少ないレース」とみなす。
#: 新馬 (0.997) と未勝利 (0.052) の間は大きく空いているので、この値の細かい調整で
#: 判定が揺れることはない。
LOW_INFORMATION_DEBUT_RATIO = 0.5


@dataclass(frozen=True)
class RaceInfoCoverage:
    """1 レース分の情報量。"""

    n_runners: int
    #: 過去走が 1 走も無い馬の数
    n_debut: int
    #: n_debut / n_runners (出走馬が 0 なら 0.0)
    debut_ratio: float
    #: 出走馬 1 頭あたりの過去走本数の平均
    mean_starts: float
    #: 履歴がまったく無いレースか (新馬戦など)
    is_low_information: bool

    def as_dict(self) -> dict:
        return asdict(self)


def race_info_coverage(
    frame: pd.DataFrame,
    debut_ratio_threshold: float = LOW_INFORMATION_DEBUT_RATIO,
) -> RaceInfoCoverage:
    """1 レース分の feature frame から情報量を測る。

    Args:
        frame: 1 レース分の行 (build_training_frame / build_inference_frame の出力)。
            ``recent_n_starts`` 列を使う。列が無い場合は「情報なし」とはみなさず
            0 頭 debut として扱う (誤って除外しないための安全側)。
        debut_ratio_threshold: この割合以上が初出走なら is_low_information=True。

    Returns:
        RaceInfoCoverage
    """
    n_runners = int(len(frame))
    if n_runners == 0:
        return RaceInfoCoverage(0, 0, 0.0, 0.0, False)

    if "recent_n_starts" not in frame.columns:
        # 列が無い = 判定材料が無い。除外してしまうより通す方が安全。
        return RaceInfoCoverage(n_runners, 0, 0.0, 0.0, False)

    starts = pd.to_numeric(frame["recent_n_starts"], errors="coerce").fillna(0.0)
    n_debut = int((starts <= 0).sum())
    ratio = n_debut / n_runners
    return RaceInfoCoverage(
        n_runners=n_runners,
        n_debut=n_debut,
        debut_ratio=round(float(ratio), 4),
        mean_starts=round(float(starts.mean()), 3),
        is_low_information=bool(ratio >= debut_ratio_threshold),
    )
