"""NN preprocessing: categorical encoding + numeric standardization.

NN は train で fit してから valid / test / inference の特徴量を同じ統計量で
変換する必要がある (NN は scale 不変ではないため)。

このクラスは:
    - カテゴリ列を train で見た値のみで int にマップする
      (split をまたいで同じカテゴリは必ず同じ整数 — これをやらないと
       train で `Tokyo=5`, valid で `Tokyo=2` のように別物として学習される)
    - 数値列を train mean/std で標準化する
    - 未知カテゴリ / NaN は -1 へ、NaN な数値は標準化後 0 (= 平均) へ

"""

from __future__ import annotations

import math
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ai.model._pickle_compat import legacy_pickle_load
from features.builder import CATEGORICAL_FEATURES

_STD_EPS = 1e-6

# log1p を噛ませる歪んだ正の量 (A2)。odds_win は対数正規で、生のまま z-score すると
# 100 倍の人気薄 1 頭がスケールを支配し NN の勾配学習を歪める。days_since_last_race /
# recent_n_starts も右に長い裾を持つ。tree の gain では log_odds_win が冗長と判定された
# が、NN は相関が高くても *表現 (幾何)* が変わるため別問題なので再検証する。
# KEIBA_LOG_FEATURES=1 で有効化。標準化の前段で適用し、適用列を preprocessor に保存して
# 推論時も同じ変換を再現する。
_LOG_FEATURE_CANDIDATES: tuple[str, ...] = (
    "odds_win",
    "days_since_last_race",
    "recent_n_starts",
)


def _log_features_flag_set() -> bool:
    """KEIBA_LOG_FEATURES が truthy か (大小文字無視)。"""
    raw = os.environ.get("KEIBA_LOG_FEATURES", "").strip().lower()
    return raw in {"1", "true", "yes"}


def _log_feature_candidates() -> tuple[str, ...]:
    """log1p 候補列。KEIBA_LOG_FEATURE_COLS (カンマ区切り) で上書き可。

    ablation 用: odds_win を外して days_since/recent_n_starts のみ log にする等を
    コード変更なしで試せる。未設定なら既定の _LOG_FEATURE_CANDIDATES。
    """
    raw = os.environ.get("KEIBA_LOG_FEATURE_COLS", "").strip()
    if not raw:
        return _LOG_FEATURE_CANDIDATES
    return tuple(c.strip() for c in raw.split(",") if c.strip())


@dataclass
class NNPreprocessor:
    """Categorical label mapping + numeric standardization fitted on train.

    Attributes:
        horse_feature_cols: per-horse feature columns the model expects.
        race_feature_cols:  race-level feature columns the model expects.
        categorical_maps:   {col: {str(value): int}}.  Values not in the map
            (including NaN) become -1 at transform time.
        numeric_means:      {col: float}  train mean per numeric column.
        numeric_stds:       {col: float}  train std per numeric column (clamped >= _STD_EPS).
    """

    horse_feature_cols: list[str]
    race_feature_cols: list[str]
    categorical_maps: dict[str, dict[str, int]] = field(default_factory=dict)
    numeric_means: dict[str, float] = field(default_factory=dict)
    numeric_stds: dict[str, float] = field(default_factory=dict)
    # odds-at-scoring head 用の odds 特徴列 (odds_win/popularity)。encoder ではなく
    # head で使うが、標準化は同じ仕組みで行う。空 = 現行 (odds は horse 側)。
    odds_feature_cols: list[str] = field(default_factory=list)
    # log1p を噛ませてから標準化する数値列 (A2)。fit 時に決まり、transform / 推論で
    # 同じ列に同じ変換を適用する。空 = 変換なし (KEIBA_LOG_FEATURES 未設定)。
    log_transform_cols: list[str] = field(default_factory=list)

    @classmethod
    def fit(
        cls,
        train_df: pd.DataFrame,
        horse_feature_cols: list[str],
        race_feature_cols: list[str],
        odds_feature_cols: list[str] | None = None,
    ) -> NNPreprocessor:
        """Fit categorical maps and numeric standardization stats on train only."""
        cat_set = set(CATEGORICAL_FEATURES)
        odds_feature_cols = list(odds_feature_cols or [])
        all_cols = list(horse_feature_cols) + list(race_feature_cols) + odds_feature_cols

        # log1p を噛ませる数値列 (A2)。候補のうち all_cols に存在し、かつ非カテゴリのもの。
        log_set = set(_log_feature_candidates()) if _log_features_flag_set() else set()

        categorical_maps: dict[str, dict[str, int]] = {}
        numeric_means: dict[str, float] = {}
        numeric_stds: dict[str, float] = {}
        log_transform_cols: list[str] = []

        for col in all_cols:
            series = (
                train_df[col]
                if col in train_df.columns
                else pd.Series([], dtype="float64")
            )

            is_categorical = col in cat_set or series.dtype == object
            if is_categorical:
                unique_vals = sorted(
                    (v for v in series.dropna().unique()), key=str
                )
                categorical_maps[col] = {str(v): i for i, v in enumerate(unique_vals)}
            else:
                numeric = pd.to_numeric(series, errors="coerce").dropna()
                if col in log_set:
                    # 負値ガード後 log1p。mean/std は変換後の分布から取る。
                    numeric = np.log1p(numeric.clip(lower=0.0))
                    log_transform_cols.append(col)
                if numeric.empty:
                    mean, std = 0.0, 1.0
                else:
                    mean = float(numeric.mean())
                    std = float(numeric.std(ddof=0))
                if not math.isfinite(mean):
                    mean = 0.0
                if not math.isfinite(std) or std < _STD_EPS:
                    std = 1.0
                numeric_means[col] = mean
                numeric_stds[col] = std

        return cls(
            horse_feature_cols=list(horse_feature_cols),
            race_feature_cols=list(race_feature_cols),
            categorical_maps=categorical_maps,
            numeric_means=numeric_means,
            numeric_stds=numeric_stds,
            odds_feature_cols=odds_feature_cols,
            log_transform_cols=log_transform_cols,
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply categorical mapping + numeric standardization.

        Returns a copy of ``frame`` with the feature columns converted to
        float32-compatible values:
            - categorical: unknown / NaN -> -1.0, known -> mapped int
            - numeric:     (x - mean) / std, NaN -> 0.0
        Non-feature columns are passed through unchanged.
        """
        result = frame.copy()
        odds_cols = list(self.odds_feature_cols or [])
        all_cols = list(self.horse_feature_cols) + list(self.race_feature_cols) + odds_cols
        # getattr: 旧 pickle には log_transform_cols 属性が無いので空リストへフォールバック。
        log_cols = set(getattr(self, "log_transform_cols", None) or [])

        for col in all_cols:
            if col in self.categorical_maps:
                mapping = self.categorical_maps[col]
                if col not in result.columns:
                    result[col] = -1.0
                    continue
                ser = result[col]
                mapped = ser.astype(str).map(mapping)
                mapped = mapped.where(ser.notna(), other=np.nan)
                result[col] = mapped.fillna(-1.0).astype(float)
            else:
                mean = self.numeric_means.get(col, 0.0)
                std = self.numeric_stds.get(col, 1.0)
                if col not in result.columns:
                    result[col] = 0.0
                    continue
                numeric = pd.to_numeric(result[col], errors="coerce")
                if col in log_cols:
                    # fit と同じ負値ガード + log1p を適用してから標準化。
                    numeric = np.log1p(numeric.clip(lower=0.0))
                result[col] = ((numeric - mean) / std).fillna(0.0).astype(float)

        return result

    @property
    def categorical_cardinalities(self) -> dict[str, int]:
        """{col: number of distinct values seen in train}.

        The embedding table needs ``cardinality + 1`` slots — the extra slot is
        reserved for the "unknown / NaN" bucket (preprocessor encodes those as
        -1, which the model shifts to index 0 on lookup).
        """
        return {col: len(m) for col, m in self.categorical_maps.items()}

    def horse_cat_metadata(self) -> tuple[list[int], list[int]]:
        """Return (positions, cardinalities) of categorical horse-level columns.

        Positions are indices into ``horse_feature_cols`` (matching the input
        feature tensor's last axis order).
        """
        positions: list[int] = []
        cardinalities: list[int] = []
        for i, col in enumerate(self.horse_feature_cols):
            if col in self.categorical_maps:
                positions.append(i)
                cardinalities.append(len(self.categorical_maps[col]))
        return positions, cardinalities

    def race_cat_metadata(self) -> tuple[list[int], list[int]]:
        """Same as :meth:`horse_cat_metadata` but for race-level columns."""
        positions: list[int] = []
        cardinalities: list[int] = []
        for i, col in enumerate(self.race_feature_cols):
            if col in self.categorical_maps:
                positions.append(i)
                cardinalities.append(len(self.categorical_maps[col]))
        return positions, cardinalities

    def save(self, path: Path) -> None:
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> NNPreprocessor:
        with path.open("rb") as f:
            obj = legacy_pickle_load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected NNPreprocessor, got {type(obj).__name__}")
        return obj
