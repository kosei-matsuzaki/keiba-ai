"""Tests for ai.model.preprocess.NNPreprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai.model.preprocess import NNPreprocessor


def _make_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "course": ["東京", "中山", "東京", "京都", "中山"],
            "surface": ["芝", "ダ", "芝", "芝", "ダ"],
            "distance": [1600.0, 2000.0, 1200.0, 2400.0, 1800.0],
            "horse_weight": [480.0, 500.0, 470.0, 510.0, 490.0],
        }
    )


def test_fit_builds_categorical_maps_only_from_train():
    train = _make_train_df()
    pp = NNPreprocessor.fit(train, ["distance", "horse_weight"], ["course", "surface"])

    assert set(pp.categorical_maps["course"]) == {"東京", "中山", "京都"}
    assert set(pp.categorical_maps["surface"]) == {"芝", "ダ"}


def test_transform_categorical_consistent_across_splits():
    """Same categorical value must get the same int in train and at inference."""
    train = _make_train_df()
    pp = NNPreprocessor.fit(train, ["distance"], ["course"])

    train_out = pp.transform(train)
    other = pd.DataFrame({"course": ["東京", "中山"], "distance": [1600.0, 1800.0]})
    other_out = pp.transform(other)

    # value of '東京' must be identical in both transforms
    train_tokyo = train_out.loc[train["course"] == "東京", "course"].iloc[0]
    other_tokyo = other_out.loc[other["course"] == "東京", "course"].iloc[0]
    assert train_tokyo == other_tokyo


def test_transform_unknown_category_maps_to_minus_one():
    train = _make_train_df()
    pp = NNPreprocessor.fit(train, ["distance"], ["course"])

    unseen = pd.DataFrame({"course": ["札幌"], "distance": [1600.0]})
    out = pp.transform(unseen)

    assert out["course"].iloc[0] == -1.0


def test_transform_nan_category_maps_to_minus_one():
    train = _make_train_df()
    pp = NNPreprocessor.fit(train, ["distance"], ["course"])

    with_nan = pd.DataFrame({"course": [np.nan], "distance": [1600.0]})
    out = pp.transform(with_nan)

    assert out["course"].iloc[0] == -1.0


def test_transform_numeric_train_is_standardized():
    train = _make_train_df()
    pp = NNPreprocessor.fit(train, ["distance", "horse_weight"], ["course"])

    out = pp.transform(train)

    # train distance should be ~ standardized: mean ≈ 0, std ≈ 1
    assert abs(float(out["distance"].mean())) < 1e-6
    assert abs(float(out["distance"].std(ddof=0)) - 1.0) < 1e-6


def test_transform_numeric_uses_train_mean_std_not_recomputed():
    """At inference, transform must use train mean/std, not the frame's own stats."""
    train = _make_train_df()
    pp = NNPreprocessor.fit(train, ["distance"], ["course"])

    # If transform recomputed mean/std on this single-row frame, the output
    # would be 0; the correct behavior is (1600 - train_mean) / train_std.
    single = pd.DataFrame({"course": ["東京"], "distance": [1600.0]})
    out = pp.transform(single)

    expected = (1600.0 - pp.numeric_means["distance"]) / pp.numeric_stds["distance"]
    assert abs(float(out["distance"].iloc[0]) - expected) < 1e-6


def test_transform_numeric_nan_becomes_zero():
    train = _make_train_df()
    pp = NNPreprocessor.fit(train, ["distance"], ["course"])

    with_nan = pd.DataFrame({"course": ["東京"], "distance": [np.nan]})
    out = pp.transform(with_nan)

    assert out["distance"].iloc[0] == 0.0


def test_transform_missing_column_is_filled():
    train = _make_train_df()
    pp = NNPreprocessor.fit(train, ["distance"], ["course", "surface"])

    no_surface = pd.DataFrame({"course": ["東京"], "distance": [1600.0]})
    out = pp.transform(no_surface)

    assert "surface" in out.columns
    assert out["surface"].iloc[0] == -1.0  # categorical missing → -1


def test_save_load_roundtrip(tmp_path):
    train = _make_train_df()
    pp = NNPreprocessor.fit(train, ["distance"], ["course"])
    p = tmp_path / "preprocessor.pkl"
    pp.save(p)

    loaded = NNPreprocessor.load(p)

    assert loaded.categorical_maps == pp.categorical_maps
    assert loaded.numeric_means == pp.numeric_means
    assert loaded.numeric_stds == pp.numeric_stds
    # apply identical transform
    out_orig = pp.transform(train)
    out_loaded = loaded.transform(train)
    pd.testing.assert_frame_equal(out_orig, out_loaded)


def test_categorical_cardinalities_property():
    train = _make_train_df()
    pp = NNPreprocessor.fit(train, ["distance"], ["course", "surface"])

    cards = pp.categorical_cardinalities
    assert cards["course"] == 3  # 東京, 中山, 京都
    assert cards["surface"] == 2  # 芝, ダ
    assert "distance" not in cards  # numeric


def test_horse_cat_metadata_returns_positions_and_cardinalities():
    train = _make_train_df()
    horse_cols = ["distance", "course", "horse_weight", "surface"]
    race_cols: list[str] = []
    pp = NNPreprocessor.fit(train, horse_cols, race_cols)

    positions, cards = pp.horse_cat_metadata()
    # course at position 1, surface at position 3
    assert positions == [1, 3]
    assert cards == [3, 2]


def test_race_cat_metadata_returns_positions_and_cardinalities():
    train = _make_train_df()
    horse_cols = ["distance"]
    race_cols = ["course", "horse_weight", "surface"]
    pp = NNPreprocessor.fit(train, horse_cols, race_cols)

    positions, cards = pp.race_cat_metadata()
    assert positions == [0, 2]
    assert cards == [3, 2]


def test_constant_numeric_column_has_std_one():
    """If a numeric column is constant in train, std is clamped so /std doesn't blow up."""
    train = pd.DataFrame(
        {
            "course": ["東京"] * 5,
            "constant_feat": [3.0] * 5,
        }
    )
    pp = NNPreprocessor.fit(train, ["constant_feat"], ["course"])

    out = pp.transform(train)
    # (3 - 3) / 1 = 0  — finite, not NaN
    assert (out["constant_feat"] == 0.0).all()


# ---------------------------------------------------------------------------
# odds_feature_cols group (odds-at-scoring head)
# ---------------------------------------------------------------------------


def _make_odds_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "distance": [1600.0, 2000.0, 1200.0, 2400.0],
            "odds_win": [2.0, 10.0, 5.0, 30.0],
            "popularity": [1.0, 4.0, 2.0, 8.0],
        }
    )


def test_odds_feature_cols_standardized():
    """odds_feature_cols は train mean/std で標準化される (encoder ではなく head 用)。"""
    train = _make_odds_train_df()
    pp = NNPreprocessor.fit(
        train, ["distance"], [], odds_feature_cols=["odds_win", "popularity"]
    )
    assert pp.odds_feature_cols == ["odds_win", "popularity"]
    out = pp.transform(train)
    # 標準化後は平均~0
    assert abs(float(out["odds_win"].mean())) < 1e-6
    assert abs(float(out["popularity"].mean())) < 1e-6
    # odds は categorical ではなく numeric
    assert "odds_win" in pp.numeric_means


def test_empty_odds_cols_equals_current_behavior():
    train = _make_odds_train_df()
    pp_a = NNPreprocessor.fit(train, ["distance", "odds_win"], [])
    pp_b = NNPreprocessor.fit(train, ["distance", "odds_win"], [], odds_feature_cols=[])
    assert pp_a.odds_feature_cols == [] == pp_b.odds_feature_cols
    pd.testing.assert_frame_equal(pp_a.transform(train), pp_b.transform(train))


def test_log_features_flag_off_by_default(monkeypatch):
    """KEIBA_LOG_FEATURES 未設定なら log 変換は無効 (log_transform_cols 空)。"""
    monkeypatch.delenv("KEIBA_LOG_FEATURES", raising=False)
    train = _make_odds_train_df()
    pp = NNPreprocessor.fit(train, ["distance", "odds_win"], [])
    assert pp.log_transform_cols == []


def test_log_features_flag_applies_log1p(monkeypatch):
    """KEIBA_LOG_FEATURES=1 で odds_win に log1p→標準化が適用される。"""
    monkeypatch.setenv("KEIBA_LOG_FEATURES", "1")
    train = _make_odds_train_df()
    pp = NNPreprocessor.fit(train, ["distance", "odds_win"], [])
    assert "odds_win" in pp.log_transform_cols
    # mean/std は log1p 後の分布から取られている
    expected_mean = float(np.log1p(train["odds_win"]).mean())
    assert abs(pp.numeric_means["odds_win"] - expected_mean) < 1e-6
    # transform も log1p→標準化: 値は (log1p(x)-mean)/std と一致
    out = pp.transform(train)
    std = pp.numeric_stds["odds_win"]
    expected = (np.log1p(train["odds_win"]) - expected_mean) / std
    np.testing.assert_allclose(out["odds_win"].to_numpy(), expected.to_numpy(), atol=1e-6)


def test_log_features_compresses_outlier_scale(monkeypatch):
    """log 変換で人気薄外れ値のスケール支配が緩み、人気馬間の分解能が上がる。"""
    train = pd.DataFrame({"odds_win": [1.5, 3.0, 10.0, 200.0] * 10})
    monkeypatch.setenv("KEIBA_LOG_FEATURES", "1")
    pp_log = NNPreprocessor.fit(train, ["odds_win"], [])
    monkeypatch.setenv("KEIBA_LOG_FEATURES", "0")
    pp_raw = NNPreprocessor.fit(train, ["odds_win"], [])
    head = pd.DataFrame({"odds_win": [1.5, 3.0, 10.0]})
    log_out = pp_log.transform(head)["odds_win"].to_numpy()
    raw_out = pp_raw.transform(head)["odds_win"].to_numpy()
    # 人気馬3頭の標準化値の広がりは log の方が大きい (分解能が高い)
    assert np.ptp(log_out) > np.ptp(raw_out)


def test_log_transform_backward_compat_without_attribute():
    """log_transform_cols 属性を持たない旧 pickle 由来インスタンスでも transform 可。"""
    pp = NNPreprocessor.__new__(NNPreprocessor)
    pp.horse_feature_cols = ["odds_win"]
    pp.race_feature_cols = []
    pp.categorical_maps = {}
    pp.numeric_means = {"odds_win": 3.0}
    pp.numeric_stds = {"odds_win": 2.0}
    pp.odds_feature_cols = []
    # log_transform_cols 属性は意図的に未設定
    out = pp.transform(pd.DataFrame({"odds_win": [5.0]}))
    assert out["odds_win"].iloc[0] == (5.0 - 3.0) / 2.0
