"""Per-(race, horse) past-race *sequence* tokens for the NN history encoder.

集約スカラー (recent_avg_finish 等) が捨てている「過去走ごとの文脈」を残すため、
各 (race_id, horse_id) について **現レースより厳密に過去** の走りを最大 MAX_HIST 走、
per-past-race トークン行列 [L, H] にして返す。各トークンは

    [その馬のその走りの成績] ++ [そのレース全体の文脈(ペース/速度)] ++ [surface one-hot]

= ユーザーの「レース全体の特徴 × その馬の走り」を 1 走分に凝縮したもの。

leak-safe: horse_history.py と同じく `Race.date < target` のみ参照
(chronological 並びの index < i で担保、同日後続レースは除外)。
scripts/seq_experiment.py のトークン化 (_margin_num / class_rank / finish_norm) を移植し、
レース全体の集約 (race_avg_agari_3f / race_avg_finish_time_norm) を追加した。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models.entry import Entry
from db.models.race import Race

MAX_HIST = 15  # 直近 N 走 (キャリア中央値 6, p90 ~21)
TOKEN_SPEC_VERSION = 1  # トークン仕様を変えたら +1 (キャッシュ無効化用)

_SURFACES = ("芝", "ダ", "障")
# coarse class ordering (higher = stronger company); unknown -> 0
_CLASS_RANK: dict[str, int] = {
    "新馬": 1, "未勝利": 1, "1勝クラス": 2, "2勝クラス": 3, "3勝クラス": 4,
    "OP": 5, "Listed": 5, "重賞": 6, "G3": 6, "G2": 7, "G1": 8,
}

# トークン特徴の順序 (H 次元)。学習/正規化/モデルで一貫させる単一の真実。
HORSE_TOKEN_FEATURES = [
    "finish_norm",      # finish_position / field_size  (∈(0,1], 低=好走)
    "field_size",
    "agari_3f",
    "margin",
    "passing_first",
    "weight_carried",
    "horse_weight",
    "distance",
    "class_rank",
    "days_since_prev",
    "won",
]
RACE_CONTEXT_FEATURES = [
    "race_avg_agari_3f",        # そのレースの平均上がり (ペース/瞬発の場の速さ)
    "race_avg_finish_time_norm",  # そのレースの平均 finish_time/distance (レース全体の速さ)
]
SURFACE_FEATURES = [f"surface_{s}" for s in _SURFACES]
TOKEN_FEATURE_NAMES = HORSE_TOKEN_FEATURES + RACE_CONTEXT_FEATURES + SURFACE_FEATURES
H = len(TOKEN_FEATURE_NAMES)  # = 16


def _margin_num(margin: object) -> float:
    """着差を馬身 (float) へ。勝ち/'クビ'/'ハナ' 等は小値、不明/非文字列(NaN)は NaN。"""
    if not isinstance(margin, str) or margin == "":
        return np.nan
    m = margin.strip()
    small = {"同着": 0.0, "ハナ": 0.05, "アタマ": 0.1, "クビ": 0.2, "大差": 10.0}
    if m in small:
        return small[m]
    try:
        return float(m.split()[0].split("/")[0].replace("+", ""))
    except (ValueError, IndexError):
        return np.nan


def _passing_first(passing: object) -> float:
    """'3-3-2-1' 形式の第1コーナー位置。無し/非文字列(NaN)は NaN。"""
    if not isinstance(passing, str) or not passing:
        return np.nan
    head = passing.split("-")[0].strip()
    try:
        return float(head)
    except ValueError:
        return np.nan


def _surf_onehot(surface: str | None) -> list[float]:
    return [1.0 if surface == s else 0.0 for s in _SURFACES]


@dataclass
class HistorySequenceCache:
    """(race_id, horse_id) -> 過去走トークン行列 [L, H] (raw, 未正規化)。

    L=0 (過去走なし) の馬はキーを持たない (呼び出し側で zero-length 扱い)。
    正規化は fit_history_normalizer で train split から fit し、dataset で適用する。
    """

    seqs: dict[tuple[str, str], np.ndarray]
    feature_names: list[str]
    max_len: int

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


def build_history_sequences(session: Session, max_len: int = MAX_HIST) -> HistorySequenceCache:
    """全 (race_id, horse_id) の leak-safe 過去走トークン列を 1 パスで構築。

    horse_history.build_horse_history_cache と同じ bulk ロード (entries+races
    1 クエリ) を使い N+1 を回避。各馬の chronological 列を 1 度作り、レースごとに
    [max(0,i-max_len):i] をスライスするだけなので per-entry の pandas filter は無い。
    """
    stmt = (
        select(
            Entry.horse_id,
            Entry.race_id,
            Race.date,
            Entry.finish_position,
            Entry.finish_time,
            Entry.margin,
            Entry.agari_3f,
            Entry.passing,
            Entry.weight_carried,
            Entry.horse_weight,
            Race.distance,
            Race.surface,
            Race.race_class,
        )
        .join(Race, Entry.race_id == Race.race_id)
        .where(Entry.finish_position.is_not(None))
        .where(Race.date.is_not(None))
        .order_by(Entry.horse_id, Race.date, Entry.race_id)
    )
    df = pd.DataFrame(
        session.execute(stmt).all(),
        columns=[
            "horse_id", "race_id", "date", "finish_position", "finish_time",
            "margin", "agari_3f", "passing", "weight_carried", "horse_weight",
            "distance", "surface", "race_class",
        ],
    )
    if df.empty:
        return HistorySequenceCache({}, list(TOKEN_FEATURE_NAMES), max_len)

    # --- per-race aggregates (レース全体の文脈) を 1 度だけ groupby で precompute ---
    df["_ftn"] = df["finish_time"] / df["distance"].where(df["distance"] > 0)
    race_grp = df.groupby("race_id", sort=False)
    field_size = race_grp["finish_position"].transform("size").to_numpy()
    race_avg_agari = race_grp["agari_3f"].transform("mean").to_numpy()
    race_avg_ftn = race_grp["_ftn"].transform("mean").to_numpy()

    # --- 行ごとのトークン (horse-own + race-context + surface) を vectorized に近い形で ---
    fin = df["finish_position"].to_numpy(dtype="float64")
    margin = df["margin"].map(_margin_num).to_numpy(dtype="float64")
    passing_first = df["passing"].map(_passing_first).to_numpy(dtype="float64")
    agari = pd.to_numeric(df["agari_3f"], errors="coerce").to_numpy(dtype="float64")
    wcar = pd.to_numeric(df["weight_carried"], errors="coerce").to_numpy(dtype="float64")
    hw = pd.to_numeric(df["horse_weight"], errors="coerce").to_numpy(dtype="float64")
    dist = pd.to_numeric(df["distance"], errors="coerce").to_numpy(dtype="float64")
    class_rank = df["race_class"].map(lambda c: float(_CLASS_RANK.get(c, 0))).to_numpy(dtype="float64")
    won = (fin == 1.0).astype("float64")
    finish_norm = np.divide(fin, field_size, out=np.full_like(fin, np.nan), where=field_size > 0)
    surf_oh = np.array([_surf_onehot(s) for s in df["surface"]], dtype="float64")  # [N, 3]

    horse_ids = df["horse_id"].to_numpy()
    race_ids = df["race_id"].to_numpy()
    dates = df["date"].to_numpy()

    # days_since_prev: 同一馬の前走からの日数 (chronological 並び前提)
    date_dt = pd.to_datetime(df["date"], errors="coerce")
    prev_date = date_dt.groupby(df["horse_id"], sort=False).shift(1)
    days_prev = (date_dt - prev_date).dt.days.to_numpy(dtype="float64")

    # [N, H] のトークン行列を一括構築
    tokens = np.column_stack([
        finish_norm, field_size.astype("float64"), agari, margin, passing_first,
        wcar, hw, dist, class_rank, days_prev, won,
        race_avg_agari, race_avg_ftn,
        surf_oh[:, 0], surf_oh[:, 1], surf_oh[:, 2],
    ]).astype("float32")
    assert tokens.shape[1] == H, (tokens.shape, H)

    # --- 馬ごとに連続区間を取り、各レースに「厳密に過去」のスライスを割り当てる ---
    seqs: dict[tuple[str, str], np.ndarray] = {}
    n = len(df)
    start = 0
    while start < n:
        end = start
        hid = horse_ids[start]
        while end < n and horse_ids[end] == hid:
            end += 1
        # df は (horse_id, date, race_id) ソート済 → [start,end) が 1 頭の時系列
        for i in range(start + 1, end):  # i==start は過去走なし → skip
            lo = max(start, i - max_len)
            seqs[(race_ids[i], hid)] = tokens[lo:i]
        start = end

    return HistorySequenceCache(seqs, list(TOKEN_FEATURE_NAMES), max_len)


def fit_history_normalizer(
    cache: HistorySequenceCache,
    train_race_ids: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    """train split のターゲットレースに属する系列のトークンから mean/std を fit。

    NaN-robust (nanmean/nanstd)。std には 1e-6 を足して 0 除算回避。
    surface one-hot 等の定数列も標準化されるが害はない。
    """
    parts = [
        seq for (rid, _hid), seq in cache.seqs.items()
        if rid in train_race_ids and len(seq) > 0
    ]
    if not parts:
        mean = np.zeros(cache.n_features, dtype="float32")
        std = np.ones(cache.n_features, dtype="float32")
        return mean, std
    stacked = np.concatenate(parts, axis=0)  # [sumL, H]
    # all-NaN な列 (例: passing が無いデータ) は nanmean/nanstd が NaN になるため、
    # mean=0 / std=1 にフォールバック (= apply 時に (x-0)/1, nan→0 で実質無効化)。
    col_valid = ~np.all(np.isnan(stacked), axis=0)
    mean = np.where(col_valid, np.nan_to_num(np.nanmean(np.where(np.isnan(stacked), 0.0, stacked), axis=0)), 0.0)
    raw_std = np.zeros(stacked.shape[1])
    if col_valid.any():
        raw_std[col_valid] = np.nanstd(stacked[:, col_valid], axis=0)
    std = np.where(col_valid & (raw_std > 0), raw_std + 1e-6, 1.0)
    return mean.astype("float32"), std.astype("float32")
