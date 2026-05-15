"""Horse historical performance features.

All aggregations are strictly before before_date to prevent target leakage.

Two implementations:

  compute_horse_history          — per-horse SQL query (1 query per call). Used
                                   for one-off inference (build_inference_frame).

  HorseHistoryCache + compute_horse_history_from_cache
                                 — preload-all-once then per-call pandas slice.
                                   Used by build_training_frame to eliminate the
                                   N+1 SQL pattern (the bulk pre-load is one
                                   query for the entire training run).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import date

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models.entry import Entry
from db.models.race import Race

# ---------------------------------------------------------------------------
# Race class weight — レース格を 1-8 の整数 weight にマップする
# ---------------------------------------------------------------------------
# 強いレースほど重い weight。重み付き average に使う。
# G1 が最も重く、新馬 / 未勝利が最も軽い。

_RACE_CLASS_WEIGHTS: dict[str, int] = {
    "G1": 8,
    "GI": 8,
    "G2": 7,
    "GII": 7,
    "G3": 6,
    "GIII": 6,
    "Listed": 5,
    "L": 5,
    "OP": 5,
    "オープン": 5,
    "3勝クラス": 4,
    "2勝クラス": 3,
    "1勝クラス": 2,
    "未勝利": 1,
    "新馬": 1,
}

# G1 / G2 / G3 (グレード戦) を判定する set。high_class_* 特徴量で使う。
_HIGH_CLASS_SET: frozenset[str] = frozenset(
    ["G1", "GI", "G2", "GII", "G3", "GIII"]
)


def race_class_weight(race_class: str | None) -> int:
    """race_class 文字列 → 1-8 の整数 weight。Unknown は 1 に丸める。"""
    if race_class is None:
        return 1
    return _RACE_CLASS_WEIGHTS.get(race_class, 1)


def is_high_class(race_class: str | None) -> bool:
    """G1 / G2 / G3 のいずれかなら True。Listed や OP は False。"""
    return race_class in _HIGH_CLASS_SET if race_class is not None else False


# ---------------------------------------------------------------------------
# Margin / passing parsers — netkeiba 着差・通過順位文字列を float / int に変換
# ---------------------------------------------------------------------------
# `entries.margin` は前着馬との差を 馬身単位 で表す文字列:
#   ハナ / アタマ / クビ        — 1 馬身未満 (head / nose / neck)
#   1/2, 3/4                    — 純粋分数
#   1.1/4, 2.3/4 など           — 整数 + 分数
#   1, 2, 3, ..., 10           — 整数馬身
#   大 / 大差                    — 10 馬身超 (10 として保守的に保存)
#   同着                         — 0 馬身 (dead heat)
#   3+ハナ など                  — 稀少 (~0.001%)。None で無視

_MARGIN_LITERAL: dict[str, float] = {
    "ハナ": 0.05,
    "アタマ": 0.10,
    "クビ": 0.15,
    "同着": 0.0,
    "大": 12.0,
    "大差": 12.0,
}


def parse_margin(s: str | None) -> float | None:
    """着差文字列を 馬身単位 float に変換する。

    Returns None for unparseable / empty / None / NaN inputs.
    勝ち馬の着差は通常 None (前者が居ないため) なので呼び出し側で 0.0 に補完する。
    """
    if s is None:
        return None
    # pandas は string 列の None を float NaN に変換するので isinstance check も入れる
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    if s in _MARGIN_LITERAL:
        return _MARGIN_LITERAL[s]
    if "/" in s:
        try:
            if "." in s:
                # "1.1/4" → 1 + 1/4
                int_str, frac_str = s.split(".", 1)
                num_str, den_str = frac_str.split("/", 1)
                return float(int_str) + float(num_str) / float(den_str)
            num_str, den_str = s.split("/", 1)
            return float(num_str) / float(den_str)
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_passing(s: str | None) -> list[int] | None:
    """通過順位文字列 (例 "11-12-11-10") を int リストに変換する。

    Returns None for invalid / empty / None / NaN inputs.
    """
    if s is None:
        return None
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return [int(p) for p in s.split("-") if p]
    except ValueError:
        return None


def _passing_std(positions: list[int] | None) -> float:
    """通過順位 list の標準偏差 (population std)。len < 2 のとき NaN。"""
    if not positions or len(positions) < 2:
        return math.nan
    return float(statistics.pstdev(positions))


def _empty_horse_history() -> dict[str, float | int | None]:
    """履歴 0 件のときの返却値テンプレート。SQL/cache 両方が同一を返すよう一元化。"""
    nan = math.nan
    return {
        "recent_avg_finish": nan,
        "recent_n_starts": 0,
        "starts_same_distance": 0,
        "starts_same_course": 0,
        "recent_avg_agari_3f": nan,
        "days_since_last_race": nan,
        "wins_same_course": 0,
        "horse_course_place_rate": nan,
        "recent_finish_1": nan,
        "recent_finish_2": nan,
        "recent_finish_3": nan,
        "recent_avg_class_weight": nan,
        "high_class_starts": 0,
        "high_class_places": 0,
        "recent_avg_margin": nan,
        "recent_avg_finish_time_norm": nan,
        "recent_best_margin_in_top3": nan,
        "recent_avg_position_change": nan,
        "recent_passing_volatility": nan,
    }


def compute_horse_history(
    session: Session,
    horse_id: str,
    before_date: date,
    distance: int | None = None,
    course: str | None = None,
    n_recent: int = 5,
) -> dict[str, float | int | None]:
    """Compute horse performance aggregates using only races strictly before before_date.

    Returns a dict with NaN values for horses with no prior history, letting
    LightGBM handle missing data natively.

    Returned keys:
        recent_avg_finish, recent_n_starts, starts_same_distance,
        starts_same_course, wins_same_course, horse_course_place_rate,
        recent_avg_agari_3f, days_since_last_race,
        recent_finish_1, recent_finish_2, recent_finish_3,
        recent_avg_class_weight, high_class_starts, high_class_places,
        recent_avg_margin, recent_avg_finish_time_norm,
        recent_best_margin_in_top3, recent_avg_position_change,
        recent_passing_volatility.

    horse_course_place_rate is the share of finishes <= 3 at the given course
    (NaN when course is None or starts_same_course == 0). It is consumed by
    builder.py to compute `course_place_rate_vs_field` and is not itself part
    of FEATURE_COLUMNS.

    Race-level (Q4) keys:
      recent_avg_class_weight  — 直近 n_recent 戦の race_class を 1-8 weight
                                 に変換した平均値。出走経験のレベル指標。
                                 履歴 0 件のとき NaN。
      high_class_starts        — G1/G2/G3 のいずれかへの出走回数 (生涯, 過去 only)。
      high_class_places        — G1/G2/G3 で finish_position ≤ 3 だった回数。

    Phase B keys (margin / finish_time / passing 由来):
      recent_avg_margin             — 直近 n_recent 戦の 着差 (馬身) 平均。
                                      勝ち馬は 0 として扱う。NaN if 全 None。
      recent_avg_finish_time_norm   — 直近 n_recent 戦の finish_time/distance
                                      (秒/m) 平均。距離正規化スピード。
      recent_best_margin_in_top3    — 直近 n_recent 戦のうち 3 着以内に入った
                                      ときの最小着差 (馬身)。 NaN if no top-3.
      recent_avg_position_change    — 直近 n_recent 戦の (last passing - finish)
                                      平均。正なら追い込み, 負なら垂れた。
      recent_passing_volatility     — 直近 n_recent 戦の通過順位 std の平均。
                                      レース内ポジション変動の指標。
    """
    before_str = before_date.isoformat()

    stmt = (
        select(Entry, Race)
        .join(Race, Entry.race_id == Race.race_id)
        .where(Entry.horse_id == horse_id)
        .where(Race.date < before_str)
        .order_by(Race.date.desc())
    )
    rows = session.execute(stmt).all()

    nan = math.nan

    if not rows:
        return _empty_horse_history()

    recent_rows = rows[:n_recent]
    finish_positions = [
        r.Entry.finish_position
        for r in recent_rows
        if r.Entry.finish_position is not None
    ]

    recent_avg_finish = (
        sum(finish_positions) / len(finish_positions) if finish_positions else nan
    )

    starts_same_distance = (
        sum(1 for r in rows if distance is not None and r.Race.distance == distance)
        if distance is not None
        else 0
    )
    starts_same_course = (
        sum(1 for r in rows if course is not None and r.Race.course == course)
        if course is not None
        else 0
    )

    # PR-C: average agari_3f over last n_recent starts (skip None values)
    agari_values = [
        r.Entry.agari_3f
        for r in recent_rows
        if r.Entry.agari_3f is not None
    ]
    recent_avg_agari_3f = sum(agari_values) / len(agari_values) if agari_values else nan

    # PR-C: days between the most recent race and before_date
    last_race_date_str = rows[0].Race.date
    try:
        last_race_date = date.fromisoformat(last_race_date_str)
        days_since_last_race = float((before_date - last_race_date).days)
    except (ValueError, TypeError):
        days_since_last_race = nan

    # Win / place counts at the given course (before before_date).
    # place rate uses starts_same_course as denominator; NaN when 0 starts.
    wins_same_course = 0
    places_same_course = 0
    if course is not None:
        for r in rows:
            if r.Race.course != course:
                continue
            pos = r.Entry.finish_position
            if pos == 1:
                wins_same_course += 1
            if pos is not None and pos <= 3:
                places_same_course += 1
    horse_course_place_rate = (
        places_same_course / starts_same_course
        if starts_same_course > 0
        else nan
    )

    # PR-C: individual finish positions for the last 3 races
    def _nth_finish(n: int) -> float:
        """Return finish_position of the n-th most recent race (1-indexed), or NaN."""
        if len(rows) >= n:
            pos = rows[n - 1].Entry.finish_position
            return float(pos) if pos is not None else nan
        return nan

    # Q4: race-level features
    # recent_avg_class_weight = mean over last n_recent of class weight (1-8)
    recent_class_weights = [race_class_weight(r.Race.race_class) for r in recent_rows]
    recent_avg_class_weight = (
        sum(recent_class_weights) / len(recent_class_weights)
        if recent_class_weights
        else nan
    )

    high_class_starts = sum(1 for r in rows if is_high_class(r.Race.race_class))
    high_class_places = sum(
        1
        for r in rows
        if is_high_class(r.Race.race_class)
        and r.Entry.finish_position is not None
        and r.Entry.finish_position <= 3
    )

    # Phase B: margin / finish_time / passing 由来の特徴量
    margin_values: list[float] = []
    top3_margins: list[float] = []
    finish_time_norms: list[float] = []
    position_changes: list[float] = []
    passing_stds: list[float] = []
    for r in recent_rows:
        m = parse_margin(r.Entry.margin)
        if m is None and r.Entry.finish_position == 1:
            # 勝ち馬は 着差 文字列が無い → 0 馬身として扱う
            m = 0.0
        if m is not None:
            margin_values.append(m)
            if r.Entry.finish_position is not None and r.Entry.finish_position <= 3:
                top3_margins.append(m)
        if (
            r.Entry.finish_time is not None
            and r.Race.distance is not None
            and r.Race.distance > 0
        ):
            finish_time_norms.append(r.Entry.finish_time / r.Race.distance)
        passing_positions = parse_passing(r.Entry.passing)
        if passing_positions:
            if r.Entry.finish_position is not None:
                position_changes.append(
                    float(passing_positions[-1] - r.Entry.finish_position)
                )
            std_val = _passing_std(passing_positions)
            if not math.isnan(std_val):
                passing_stds.append(std_val)

    recent_avg_margin = (
        sum(margin_values) / len(margin_values) if margin_values else nan
    )
    recent_avg_finish_time_norm = (
        sum(finish_time_norms) / len(finish_time_norms) if finish_time_norms else nan
    )
    recent_best_margin_in_top3 = min(top3_margins) if top3_margins else nan
    recent_avg_position_change = (
        sum(position_changes) / len(position_changes) if position_changes else nan
    )
    recent_passing_volatility = (
        sum(passing_stds) / len(passing_stds) if passing_stds else nan
    )

    return {
        "recent_avg_finish": recent_avg_finish,
        "recent_n_starts": len(rows),
        "starts_same_distance": starts_same_distance,
        "starts_same_course": starts_same_course,
        "recent_avg_agari_3f": recent_avg_agari_3f,
        "days_since_last_race": days_since_last_race,
        "wins_same_course": wins_same_course,
        "horse_course_place_rate": horse_course_place_rate,
        "recent_finish_1": _nth_finish(1),
        "recent_finish_2": _nth_finish(2),
        "recent_finish_3": _nth_finish(3),
        "recent_avg_class_weight": recent_avg_class_weight,
        "high_class_starts": high_class_starts,
        "high_class_places": high_class_places,
        "recent_avg_margin": recent_avg_margin,
        "recent_avg_finish_time_norm": recent_avg_finish_time_norm,
        "recent_best_margin_in_top3": recent_best_margin_in_top3,
        "recent_avg_position_change": recent_avg_position_change,
        "recent_passing_volatility": recent_passing_volatility,
    }


# ---------------------------------------------------------------------------
# Bulk-preload variant — eliminates N+1 SQL by loading all horse history once.
# ---------------------------------------------------------------------------


@dataclass
class HorseHistoryCache:
    """Pre-loaded horse race history for fast feature lookup.

    Build once at the start of build_training_frame; pass to
    compute_horse_history_from_cache for each (horse, target_date) pair.

    Memory: ~24 bytes per past entry × ~100k entries (2 years) ≈ 2 MB.
    """

    # DataFrame with columns: horse_id, date (str), distance (int), course (str),
    # finish_position (int|None), agari_3f (float|None).
    # Rows already sorted by (horse_id, date desc) for fast head() of recent.
    df: pd.DataFrame
    # Per-horse subset cache so each lookup avoids a full-frame filter.
    # Sorted by date desc within each group.
    by_horse: dict[str, pd.DataFrame]


def build_horse_history_cache(session: Session) -> HorseHistoryCache:
    """Load all horse race history in one SQL pass.

    Returns a HorseHistoryCache pre-grouped by horse_id with each group
    sorted descending by date (for fast recent-N head() calls).

    Q4: race_class も同時取得し、recent_avg_class_weight などの計算で使う。
    Phase B: margin / finish_time / passing も取得し、関連 5 特徴量の計算に使う。
    """
    query = (
        select(
            Entry.horse_id,
            Race.date,
            Race.distance,
            Race.course,
            Race.race_class,
            Entry.finish_position,
            Entry.agari_3f,
            Entry.margin,
            Entry.finish_time,
            Entry.passing,
        )
        .join(Race, Entry.race_id == Race.race_id)
    )
    rows = session.execute(query).all()
    df = pd.DataFrame(
        rows,
        columns=[
            "horse_id", "date", "distance", "course", "race_class",
            "finish_position", "agari_3f",
            "margin", "finish_time", "passing",
        ],
    )
    # Pre-compute class_weight column once (vectorised) so per-pair lookup
    # is just a column read.
    df["class_weight"] = df["race_class"].map(_RACE_CLASS_WEIGHTS).fillna(1).astype(int)

    # Phase B: margin / finish_time / passing 由来の派生列を 1 度だけ計算しておく。
    # 行ごとに parse する SQL 版と同じ semantics を維持しつつ、per-pair lookup を
    # 単純な column read で済ませるためのキャッシュ。
    margin_values = df["margin"].map(parse_margin)
    # 勝ち馬 (finish_position == 1) かつ margin 文字列が parse できないときは
    # 0 馬身として扱う (前者がいないため netkeiba は空にする)。
    winner_no_margin = (df["finish_position"] == 1) & margin_values.isna()
    margin_values = margin_values.where(~winner_no_margin, 0.0)
    df["margin_value"] = margin_values

    # finish_time / distance: distance が 0 や None は inf / NaN を避けて NaN 化。
    distance_safe = df["distance"].where(df["distance"] > 0)
    df["finish_time_norm"] = df["finish_time"] / distance_safe

    passing_lists = df["passing"].map(parse_passing)
    df["passing_std"] = passing_lists.map(_passing_std)
    # last_pos - finish_position; 正なら追い込み (last 位より上昇), 負なら垂れた。
    df["passing_last_pos"] = passing_lists.map(
        lambda lst: float(lst[-1]) if lst else math.nan
    )
    df["position_change"] = df["passing_last_pos"] - df["finish_position"]

    # Sort by date desc so head(n_recent) gives the n most recent races.
    # Stable sort keeps insertion order across ties (rare but harmless).
    df = df.sort_values(["horse_id", "date"], ascending=[True, False], kind="stable")

    by_horse = {hid: g for hid, g in df.groupby("horse_id", sort=False)}
    return HorseHistoryCache(df=df, by_horse=by_horse)


def compute_horse_history_from_cache(
    cache: HorseHistoryCache,
    horse_id: str,
    before_date: date,
    distance: int | None = None,
    course: str | None = None,
    n_recent: int = 5,
) -> dict[str, float | int | None]:
    """Cached version of compute_horse_history.

    Output is bit-for-bit identical to compute_horse_history; only the data
    source changes (pandas slice instead of SQL query). The function exists to
    enable batch feature building without per-pair SQL round-trips.
    """
    nan = math.nan
    horse_df = cache.by_horse.get(horse_id)
    if horse_df is None:
        return _empty_horse_history()

    before_str = before_date.isoformat()
    h = horse_df[horse_df["date"] < before_str]

    if h.empty:
        return _empty_horse_history()

    # h is already sorted by date desc within the group — head(n_recent) gives
    # the n most recent rows.
    recent = h.head(n_recent)

    finish_positions_recent = recent["finish_position"].dropna()
    recent_avg_finish = (
        float(finish_positions_recent.mean()) if not finish_positions_recent.empty else nan
    )

    starts_same_distance = (
        int((h["distance"] == distance).sum()) if distance is not None else 0
    )

    if course is not None:
        same_course_mask = h["course"] == course
        starts_same_course = int(same_course_mask.sum())
    else:
        same_course_mask = None
        starts_same_course = 0

    agari_recent = recent["agari_3f"].dropna()
    recent_avg_agari_3f = float(agari_recent.mean()) if not agari_recent.empty else nan

    # Most recent date is the first row (sorted desc).
    last_race_date_str = str(h["date"].iloc[0])
    try:
        last_race_date = date.fromisoformat(last_race_date_str)
        days_since_last_race = float((before_date - last_race_date).days)
    except (ValueError, TypeError):
        days_since_last_race = nan

    wins_same_course = 0
    places_same_course = 0
    if same_course_mask is not None and starts_same_course > 0:
        same_course = h[same_course_mask]
        # finish_position can be None; use sum() with bool conditions excluding NaN.
        wins_same_course = int((same_course["finish_position"] == 1).sum())
        # Place: finish in [1, 3]; pandas treats NaN as False in comparisons.
        places_same_course = int(
            ((same_course["finish_position"] >= 1) & (same_course["finish_position"] <= 3)).sum()
        )

    horse_course_place_rate = (
        places_same_course / starts_same_course
        if starts_same_course > 0
        else nan
    )

    def _nth_finish(n: int) -> float:
        if len(h) >= n:
            pos = h["finish_position"].iloc[n - 1]
            return float(pos) if pd.notna(pos) else nan
        return nan

    # Q4: race-level features
    if "class_weight" in recent.columns and not recent.empty:
        recent_avg_class_weight = float(recent["class_weight"].mean())
    else:
        recent_avg_class_weight = nan

    if "race_class" in h.columns:
        high_class_mask = h["race_class"].isin(_HIGH_CLASS_SET)
        high_class_starts = int(high_class_mask.sum())
        high_class_places = int(
            (
                high_class_mask
                & (h["finish_position"] >= 1)
                & (h["finish_position"] <= 3)
            ).sum()
        )
    else:
        high_class_starts = 0
        high_class_places = 0

    # Phase B: margin / finish_time / passing 由来 (recent n_recent races のみ)
    margin_recent = recent["margin_value"].dropna() if "margin_value" in recent.columns else pd.Series(dtype=float)
    recent_avg_margin = float(margin_recent.mean()) if not margin_recent.empty else nan

    ftn_recent = recent["finish_time_norm"].dropna() if "finish_time_norm" in recent.columns else pd.Series(dtype=float)
    recent_avg_finish_time_norm = float(ftn_recent.mean()) if not ftn_recent.empty else nan

    if "margin_value" in recent.columns:
        top3_mask = (recent["finish_position"] >= 1) & (recent["finish_position"] <= 3)
        top3_margins = recent.loc[top3_mask, "margin_value"].dropna()
        recent_best_margin_in_top3 = (
            float(top3_margins.min()) if not top3_margins.empty else nan
        )
    else:
        recent_best_margin_in_top3 = nan

    pc_recent = recent["position_change"].dropna() if "position_change" in recent.columns else pd.Series(dtype=float)
    recent_avg_position_change = float(pc_recent.mean()) if not pc_recent.empty else nan

    pv_recent = recent["passing_std"].dropna() if "passing_std" in recent.columns else pd.Series(dtype=float)
    recent_passing_volatility = float(pv_recent.mean()) if not pv_recent.empty else nan

    return {
        "recent_avg_finish": recent_avg_finish,
        "recent_n_starts": int(len(h)),
        "starts_same_distance": starts_same_distance,
        "starts_same_course": starts_same_course,
        "recent_avg_agari_3f": recent_avg_agari_3f,
        "days_since_last_race": days_since_last_race,
        "wins_same_course": wins_same_course,
        "horse_course_place_rate": horse_course_place_rate,
        "recent_finish_1": _nth_finish(1),
        "recent_finish_2": _nth_finish(2),
        "recent_finish_3": _nth_finish(3),
        "recent_avg_class_weight": recent_avg_class_weight,
        "high_class_starts": high_class_starts,
        "high_class_places": high_class_places,
        "recent_avg_margin": recent_avg_margin,
        "recent_avg_finish_time_norm": recent_avg_finish_time_norm,
        "recent_best_margin_in_top3": recent_best_margin_in_top3,
        "recent_avg_position_change": recent_avg_position_change,
        "recent_passing_volatility": recent_passing_volatility,
    }
