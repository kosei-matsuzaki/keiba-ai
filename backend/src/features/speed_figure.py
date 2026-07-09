"""Speed figure (タイム指数) — par-time + track-variant adjusted speed quality.

集約スカラー (`recent_avg_finish_time_norm` = global 標準化された time/distance) も
履歴トークンの `race_avg_finish_time_norm` (レース内平均) も、**コース・距離・馬場・
開催日のばらつきを par で正規化していない**。そのため「遅いレースで勝った遅い馬」と
「速いレースで勝った速い馬」が区別できない。本モジュールは競馬予想で最も情報量の高い
派生量である絶対速度品質 (Beyer / JRDB タイム指数の核) を生の finish_time から作る:

    speed_fig = (par_time + track_variant − finish_time) / par_time   (高 = 速い = 好走)

leak-safety:
- **par_time(course, distance, surface)** は **train split のレースのみ** から fit する
  (NNPreprocessor / fit_history_normalizer と同じ train-only 規約)。学習済み artifact
  として保存し推論で再ロードする。
- **track_variant(course, date)** はその日そのコースの **当日レースのみ** から計算する。
  履歴トークンは常に target race より厳密に過去の走りなので、その走りの当日 (≤ 走破日
  < target 日) のデータだけを使う variant は未来情報を含まない。train/test 境界も
  またがない (同日集約のみ)。

par が無い (新規 course/dist/surf 組) ときは (surface, distance) → global の順に
フォールバックし、それも無ければ speed_fig は NaN (履歴正規化が nan→0 で無効化)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models.entry import Entry
from db.models.race import Race


@dataclass
class SpeedFigureModel:
    """train-fit par テーブル + 日次 track-variant。

    par[(course, distance, surface)] = train の勝ち馬 finish_time 中央値 (秒)。
    par_by_surf_dist[(surface, distance)] / par_global は欠損組のフォールバック。
    variants[(course, date)] = その日そのコースの mean(winner_time − par) (秒)。
    """

    par: dict[tuple[str, int, str], float] = field(default_factory=dict)
    par_by_surf_dist: dict[tuple[str, int], float] = field(default_factory=dict)
    par_global: float = float("nan")
    variants: dict[tuple[str, str], float] = field(default_factory=dict)

    def lookup_par(self, course: str, distance: int, surface: str) -> float:
        """par をフォールバック付きで引く。無ければ NaN。"""
        v = self.par.get((course, int(distance), surface))
        if v is not None:
            return v
        v = self.par_by_surf_dist.get((surface, int(distance)))
        if v is not None:
            return v
        return self.par_global


def _winner_times_frame(session: Session) -> pd.DataFrame:
    """全レースの勝ち馬 finish_time + course/distance/surface/date を 1 クエリで。"""
    stmt = (
        select(
            Race.race_id, Race.course, Race.distance, Race.surface, Race.date,
            Entry.finish_time,
        )
        .join(Entry, Entry.race_id == Race.race_id)
        .where(Entry.finish_position == 1)
        .where(Entry.finish_time.is_not(None))
        .where(Entry.finish_time > 0)
        .where(Race.date.is_not(None))
    )
    df = pd.DataFrame(
        session.execute(stmt).all(),
        columns=["race_id", "course", "distance", "surface", "date", "finish_time"],
    )
    if not df.empty:
        df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
        df["finish_time"] = pd.to_numeric(df["finish_time"], errors="coerce")
        df = df.dropna(subset=["distance", "finish_time"])
        df["distance"] = df["distance"].astype(int)
    return df


def build_speed_figure_model(
    session: Session,
    train_race_ids: set[str],
) -> SpeedFigureModel:
    """par を train split から、variant を全日付から (同日集約のみ) 構築する。

    par: train レースの勝ち馬 finish_time 中央値 per (course, distance, surface)、
        フォールバック用に (surface, distance) と global も持つ。
    variant: 全レースを使い (course, date) ごとに mean(winner_time − par(combo))。
        当日集約なので過去走トークンに使っても未来リークしない。
    """
    df = _winner_times_frame(session)
    if df.empty:
        return SpeedFigureModel()

    train_df = df[df["race_id"].isin(train_race_ids)]
    if train_df.empty:
        # train が空 (full-frame fallback 等) のときは全体で fit して退行を避ける。
        train_df = df

    par: dict[tuple[str, int, str], float] = {
        (c, int(d), s): float(t)
        for (c, d, s), t in train_df.groupby(
            ["course", "distance", "surface"]
        )["finish_time"].median().items()
    }
    par_by_surf_dist: dict[tuple[str, int], float] = {
        (s, int(d)): float(t)
        for (s, d), t in train_df.groupby(["surface", "distance"])[
            "finish_time"
        ].median().items()
    }
    par_global = float(train_df["finish_time"].median())

    model = SpeedFigureModel(
        par=par,
        par_by_surf_dist=par_by_surf_dist,
        par_global=par_global,
    )

    # track_variant: 各 (course, date) で winner の (実時計 − par) の平均。
    pars = np.array([
        model.lookup_par(c, d, s)
        for c, d, s in zip(df["course"], df["distance"], df["surface"], strict=False)
    ])
    resid = df["finish_time"].to_numpy() - pars
    vdf = pd.DataFrame({
        "course": df["course"].to_numpy(),
        "date": df["date"].to_numpy(),
        "resid": resid,
    }).dropna(subset=["resid"])
    variants: dict[tuple[str, str], float] = {
        (c, dt): float(v)
        for (c, dt), v in vdf.groupby(["course", "date"])["resid"].mean().items()
    }
    model.variants = variants
    return model


def add_speed_figure_column(
    df: pd.DataFrame,
    model: SpeedFigureModel,
) -> np.ndarray:
    """tokenize 用 df に speed_fig 値 [N] を返す (高 = 速い = 好走、欠損 NaN)。

    df は course / distance / surface / date / finish_time 列を持つこと。
    speed_fig = (par + variant − finish_time) / par。par 欠損や finish_time 欠損は NaN。
    """
    if df.empty:
        return np.zeros(0, dtype="float64")
    course = df["course"].to_numpy()
    dist = pd.to_numeric(df["distance"], errors="coerce").to_numpy()
    surf = df["surface"].to_numpy()
    date = df["date"].to_numpy()
    ftime = pd.to_numeric(df["finish_time"], errors="coerce").to_numpy(dtype="float64")

    par = np.array([
        model.lookup_par(c, int(d), s) if np.isfinite(d) else np.nan
        for c, d, s in zip(course, dist, surf, strict=False)
    ], dtype="float64")
    variant = np.array([
        model.variants.get((c, dt), 0.0) for c, dt in zip(course, date, strict=False)
    ], dtype="float64")

    with np.errstate(invalid="ignore", divide="ignore"):
        speed = (par + variant - ftime) / par
    speed[~np.isfinite(par) | (par <= 0)] = np.nan
    speed[~np.isfinite(ftime)] = np.nan
    return speed
