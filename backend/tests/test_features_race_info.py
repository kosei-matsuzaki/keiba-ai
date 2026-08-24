"""features.race_info — レース単位の情報量判定。

クラス名 (新馬) ではなく実際の過去走本数で測る。実測 (test 19ヶ月・5,404 レース) では
新馬の過去走ゼロ率が 0.997、未勝利が 0.052 と大きく離れているので、閾値 0.5 の前後で
判定が揺れることはない。
"""

from __future__ import annotations

import pandas as pd

from features.race_info import LOW_INFORMATION_DEBUT_RATIO, race_info_coverage


def _frame(starts: list[float | None]) -> pd.DataFrame:
    return pd.DataFrame({"recent_n_starts": starts})


class TestRaceInfoCoverage:
    def test_all_debut_is_low_information(self):
        """新馬戦: 全頭に過去走が無い。"""
        cov = race_info_coverage(_frame([0, 0, 0, 0]))
        assert cov.n_runners == 4
        assert cov.n_debut == 4
        assert cov.debut_ratio == 1.0
        assert cov.mean_starts == 0.0
        assert cov.is_low_information is True

    def test_experienced_field_is_not_low_information(self):
        cov = race_info_coverage(_frame([12, 8, 20, 5]))
        assert cov.n_debut == 0
        assert cov.debut_ratio == 0.0
        assert cov.mean_starts == 11.25
        assert cov.is_low_information is False

    def test_a_few_debutants_do_not_flag_the_race(self):
        """未勝利戦のように数頭だけ初出走でも、レース全体としては情報がある。"""
        cov = race_info_coverage(_frame([0, 6, 9, 4, 11]))
        assert cov.n_debut == 1
        assert cov.debut_ratio == 0.2
        assert cov.is_low_information is False

    def test_threshold_is_inclusive(self):
        cov = race_info_coverage(_frame([0, 0, 5, 5]))
        assert cov.debut_ratio == LOW_INFORMATION_DEBUT_RATIO
        assert cov.is_low_information is True

    def test_missing_starts_count_as_debut(self):
        """NaN は「過去走が取れていない」= 判断材料が無い側に倒す。"""
        cov = race_info_coverage(_frame([None, None, 7]))
        assert cov.n_debut == 2
        assert cov.is_low_information is True

    def test_empty_frame_is_not_flagged(self):
        cov = race_info_coverage(pd.DataFrame({"recent_n_starts": []}))
        assert cov.n_runners == 0
        assert cov.is_low_information is False

    def test_missing_column_does_not_flag(self):
        """判定材料が無いときに誤って除外しない (安全側に倒す)。"""
        cov = race_info_coverage(pd.DataFrame({"horse_id": ["a", "b"]}))
        assert cov.n_runners == 2
        assert cov.n_debut == 0
        assert cov.is_low_information is False

    def test_as_dict_round_trips(self):
        d = race_info_coverage(_frame([0, 0])).as_dict()
        assert set(d) == {
            "n_runners", "n_debut", "debut_ratio", "mean_starts", "is_low_information",
        }
