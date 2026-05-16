"""Tests for ai.nn.preprocess.NNPreprocessor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ai.nn.preprocess import NNPreprocessor


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
