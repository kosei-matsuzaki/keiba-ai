"""Feature frame construction for training and inference.

build_training_frame and build_inference_frame both delegate to _build_entry_row,
which strictly uses only data before each race's date to prevent leakage.

Within-race relative features (compute_within_race_features) require all
entries for a race to be available simultaneously. To populate
`jockey_recent_win_rate_vs_field` and `course_place_rate_vs_field` without
duplicating DB queries, we build raw entry rows first (which compute jockey
recent win rate and horse same-course place rate), then derive the per-race
relative dict from those values and merge it back.

Caching: build_training_frame は entry × ~6 SQL の N+1 構造で 3,300
race のフルスキャンに 15-20 分かかる。DB の mtime + (start, end) を
key に pickle で feature DataFrame を data/cache/training_frames/ に
キャッシュし、同じ条件での 2 回目以降の呼び出しを秒で済ませる。
DB が更新されたら mtime が変わり cache は自動的に miss するので、
古い結果を返してしまう心配はない。
KEIBA_DISABLE_FRAME_CACHE=1 で無効化可能。
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import pickle
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from keiba_ai.core.paths import data_dir
from keiba_ai.db.models.entry import Entry
from keiba_ai.db.models.horse import Horse
from keiba_ai.db.models.race import Race
from keiba_ai.features.course import extract_race_features
from keiba_ai.features.horse_history import (
    HorseHistoryCache,
    build_horse_history_cache,
    compute_horse_history,
    compute_horse_history_from_cache,
)
from keiba_ai.features.jockey import (
    JockeyHistoryCache,
    build_jockey_history_cache,
    compute_jockey_stats,
    compute_jockey_stats_from_cache,
)
from keiba_ai.features.odds import extract_odds_features
from keiba_ai.features.pedigree import (
    PedigreeCache,
    build_pedigree_cache,
    compute_pedigree_features,
    compute_pedigree_features_from_cache,
)
from keiba_ai.features.relative_features import compute_within_race_features
from keiba_ai.features.trainer import (
    TrainerHistoryCache,
    build_trainer_history_cache,
    compute_trainer_stats,
    compute_trainer_stats_from_cache,
)

log = logging.getLogger(__name__)

# Fixed column order — must stay stable across training and inference.
FEATURE_COLUMNS: list[str] = [
    # Race / course
    "distance",
    "n_runners",
    "post_position",
    "post_position_ratio",
    # Entry basics
    "age",
    "horse_weight",
    "horse_weight_diff",
    # Odds / market
    "odds_win",
    "popularity",
    "log_odds_win",
    # Horse history (original)
    "recent_avg_finish",
    "recent_n_starts",
    "starts_same_distance",
    "starts_same_course",
    # Jockey
    "jockey_recent_win_rate",
    "jockey_recent_place_rate",
    "jockey_course_place_rate",
    # Trainer
    "trainer_course_place_rate",
    # Categorical (listed last; referenced by name in CATEGORICAL_FEATURES)
    "surface",
    "course",
    "weather",
    "track_condition",
    "race_class",
    "sex",
    # Horse history extensions (PR-C)
    "recent_avg_agari_3f",
    "days_since_last_race",
    "wins_same_course",
    "recent_finish_1",
    "recent_finish_2",
    "recent_finish_3",
    # Race-level (Q4): クラス重み付きの履歴指標
    "recent_avg_class_weight",
    "high_class_starts",
    "high_class_places",
    # Phase B: margin / finish_time / passing 由来の履歴指標
    "recent_avg_margin",
    "recent_avg_finish_time_norm",
    "recent_best_margin_in_top3",
    "recent_avg_position_change",
    "recent_passing_volatility",
    # Within-race relative features (PR-C)
    "horse_weight_pct",
    "odds_win_rank",
    "weight_carried_pct",
    "jockey_recent_win_rate_vs_field",
    "course_place_rate_vs_field",
    "odds_win_diff_from_favorite",
    # Pedigree (PR-C)
    "sire_progeny_win_rate",
    "dam_progeny_win_rate",
]

CATEGORICAL_FEATURES: list[str] = [
    "surface",
    "course",
    "weather",
    "track_condition",
    "race_class",
    "sex",
]

# 単勝オッズ由来の特徴量。市場予想を直接 model に流し込みたくない A/B 評価で
# 除外可能にしておく。KEIBA_EXCLUDE_ODDS_FEATURES=1 のとき get_active_features
# はこれらを取り除いた FEATURE_COLUMNS を返す。
ODDS_FEATURE_COLUMNS: list[str] = [
    "odds_win",
    "popularity",
    "log_odds_win",
    "odds_win_rank",
    "odds_win_diff_from_favorite",
]


def _exclude_odds_flag_set() -> bool:
    """KEIBA_EXCLUDE_ODDS_FEATURES が truthy か。

    "1" / "true" / "yes" を真として扱う（大小文字無視）。
    """
    raw = os.environ.get("KEIBA_EXCLUDE_ODDS_FEATURES", "").strip().lower()
    return raw in {"1", "true", "yes"}


def get_active_features() -> list[str]:
    """学習・推論で実際に使う特徴量列を返す。

    KEIBA_EXCLUDE_ODDS_FEATURES=1 のとき ODDS_FEATURE_COLUMNS を除外した
    FEATURE_COLUMNS を返す。それ以外は FEATURE_COLUMNS のコピーをそのまま返す。

    呼び出しごとに環境変数を読むため、テストや CLI 一回限りの上書きが効く。
    """
    if _exclude_odds_flag_set():
        excluded = set(ODDS_FEATURE_COLUMNS)
        return [c for c in FEATURE_COLUMNS if c not in excluded]
    return list(FEATURE_COLUMNS)


def _build_entry_row(
    session: Session,
    race: Race,
    entry: Entry,
    n_runners: int,
    race_date: date,
    horse_cache: HorseHistoryCache | None = None,
    jockey_cache: JockeyHistoryCache | None = None,
    trainer_cache: TrainerHistoryCache | None = None,
    pedigree_cache: PedigreeCache | None = None,
) -> dict[str, object]:
    """Build a single feature row for one entry in one race.

    All historical lookups use race_date (strictly before) to prevent leakage.
    Relative features are absent here — _build_race_rows merges them in a
    second pass once the full field is known.

    *_cache: 渡された場合は per-call SQL を avoid し、preload 済みの pandas
    DataFrame から該当の特徴量を計算する（build_training_frame の N+1 解消用）。
    None なら従来通り SQL を発行する。
    """
    if horse_cache is not None:
        horse_feats = compute_horse_history_from_cache(
            horse_cache,
            entry.horse_id,
            before_date=race_date,
            distance=race.distance,
            course=race.course,
        )
    else:
        horse_feats = compute_horse_history(
            session,
            entry.horse_id,
            before_date=race_date,
            distance=race.distance,
            course=race.course,
        )
    if entry.jockey_id:
        if jockey_cache is not None:
            jockey_feats = compute_jockey_stats_from_cache(
                jockey_cache,
                entry.jockey_id,
                before_date=race_date,
                course=race.course,
                days=30,
            )
        else:
            jockey_feats = compute_jockey_stats(
                session,
                entry.jockey_id,
                before_date=race_date,
                course=race.course,
                days=30,
            )
    else:
        jockey_feats = {
            "jockey_recent_win_rate": math.nan,
            "jockey_recent_place_rate": math.nan,
            "jockey_course_place_rate": math.nan,
        }
    if entry.trainer_id:
        if trainer_cache is not None:
            trainer_feats = compute_trainer_stats_from_cache(
                trainer_cache,
                entry.trainer_id,
                before_date=race_date,
                course=race.course,
            )
        else:
            trainer_feats = compute_trainer_stats(
                session,
                entry.trainer_id,
                before_date=race_date,
                course=race.course,
            )
    else:
        trainer_feats = {"trainer_course_place_rate": math.nan}

    race_feats = extract_race_features(race, entry, n_runners)
    odds_feats = extract_odds_features(entry)

    # Pedigree features (PR-C); None → NaN so LightGBM gets float columns
    if pedigree_cache is not None:
        sire, dam = pedigree_cache.horse_to_sire_dam.get(
            entry.horse_id, (None, None)
        )
        _ped = compute_pedigree_features_from_cache(
            pedigree_cache, sire, dam, race_date
        )
    else:
        horse = session.get(Horse, entry.horse_id)
        sire = horse.sire if horse else None
        dam = horse.dam if horse else None
        _ped = compute_pedigree_features(session, sire, dam, race_date)
    pedigree_feats = {
        k: (v if v is not None else math.nan) for k, v in _ped.items()
    }

    row: dict[str, object] = {
        "race_id": race.race_id,
        "horse_id": entry.horse_id,
        "date": race.date,
        "finish_position": entry.finish_position,
        "payout_place": race.payout_place,
    }
    row.update(race_feats)
    row.update(odds_feats)
    row.update(horse_feats)
    row.update(jockey_feats)
    row.update(trainer_feats)
    row.update(pedigree_feats)
    return row


def _build_race_rows(
    session: Session,
    race: Race,
    entries: list[Entry],
    horse_cache: HorseHistoryCache | None = None,
    jockey_cache: JockeyHistoryCache | None = None,
    trainer_cache: TrainerHistoryCache | None = None,
    pedigree_cache: PedigreeCache | None = None,
) -> list[dict[str, object]]:
    """Build all entry rows for a single race, including within-race relative features.

    Two-phase: build raw rows first so jockey_recent_win_rate and
    horse_course_place_rate become available, then derive the relative dict
    and merge it back. This avoids re-querying the DB for those stats.
    """
    n_runners = race.n_runners or len(entries)
    race_date = date.fromisoformat(race.date)

    rows = [
        _build_entry_row(
            session, race, entry, n_runners, race_date,
            horse_cache=horse_cache,
            jockey_cache=jockey_cache,
            trainer_cache=trainer_cache,
            pedigree_cache=pedigree_cache,
        )
        for entry in entries
    ]

    nan = math.nan
    jockey_recent_win_rates = {
        row["horse_id"]: row.get("jockey_recent_win_rate", nan) for row in rows
    }
    horse_course_place_rates = {
        row["horse_id"]: row.get("horse_course_place_rate", nan) for row in rows
    }

    relative_dict = compute_within_race_features(
        entries,
        jockey_recent_win_rates=jockey_recent_win_rates,
        horse_course_place_rates=horse_course_place_rates,
    )
    for row in rows:
        row.update(relative_dict.get(row["horse_id"], {}))
    return rows


def _frame_cache_dir() -> Path:
    d = data_dir() / "cache" / "training_frames"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _frame_cache_key(
    db_path_str: str | None,
    train_start: str | None,
    train_end: str | None,
) -> str:
    """Build a cache key from DB mtime + (start, end). DB content change
    invalidates everything (mtime moves), and per-range outputs stay separate.
    """
    if db_path_str and Path(db_path_str).exists():
        mtime = int(os.path.getmtime(db_path_str))
        size = os.path.getsize(db_path_str)
    else:
        mtime, size = 0, 0
    raw = f"{db_path_str}|{mtime}|{size}|{train_start}|{train_end}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _frame_cache_load(key: str) -> pd.DataFrame | None:
    path = _frame_cache_dir() / f"{key}.pkl"
    if not path.exists():
        return None
    try:
        df = pd.read_pickle(path)
        log.info("Loaded cached training frame from %s (rows=%d)", path, len(df))
        return df
    except Exception as exc:  # noqa: BLE001 — cache corruption shouldn't fail builds
        log.warning("Failed to read frame cache %s: %s", path, exc)
        return None


def _frame_cache_save(key: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        return  # skip empty frames; trivial to recompute and avoids polluting cache
    path = _frame_cache_dir() / f"{key}.pkl"
    try:
        frame.to_pickle(path)
        log.info("Cached training frame to %s (rows=%d)", path, len(frame))
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to write frame cache %s: %s", path, exc)


def _session_db_path(session: Session) -> str | None:
    """Best-effort extraction of the underlying SQLite file path from a Session.

    Returns None for in-memory DBs (no stable mtime → cache key would collide
    across independent in-memory engines, leading to cross-contamination).
    """
    bind = session.get_bind()
    try:
        path = bind.url.database  # type: ignore[union-attr]
    except AttributeError:
        return None
    if not path or path == ":memory:":
        return None
    return path


def _load_races_in_range(
    session: Session,
    start_date: str | None,
    end_date: str | None,
) -> list[Race]:
    stmt = select(Race).order_by(Race.date)
    if start_date:
        stmt = stmt.where(Race.date >= start_date)
    if end_date:
        stmt = stmt.where(Race.date <= end_date)
    return list(session.scalars(stmt).all())


def build_training_frame(
    session: Session,
    train_start: str | None = None,
    train_end: str | None = None,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Build a feature DataFrame for all races in [train_start, train_end].

    Includes finish_position for label assignment.
    Leakage prevention: each entry's features are computed using only records
    strictly before that race's date.

    `use_cache=True` (default) なら DB mtime + (start, end) ベースで pickle
    キャッシュを読み書きする。env `KEIBA_DISABLE_FRAME_CACHE=1` でも無効化。
    """
    cache_disabled = os.getenv("KEIBA_DISABLE_FRAME_CACHE") == "1"
    cache_active = use_cache and not cache_disabled
    cache_key: str | None = None

    if cache_active:
        db_path_str = _session_db_path(session)
        if db_path_str is None:
            # In-memory DB or unknown bind — no stable mtime, so refuse to cache.
            cache_active = False
        else:
            cache_key = _frame_cache_key(db_path_str, train_start, train_end)
            cached = _frame_cache_load(cache_key)
            if cached is not None:
                return cached

    races = _load_races_in_range(session, train_start, train_end)
    if not races:
        return pd.DataFrame(
            columns=["race_id", "horse_id", "date", "finish_position"] + FEATURE_COLUMNS
        )

    n_total = len(races)
    import time as _time

    # 全 horse / jockey / trainer / pedigree の race history を一括ロードし、
    # per-call SQL を完全に排除する (N+1 SQL の最大要因)。
    log.info("Pre-loading horse / jockey / trainer / pedigree caches...")
    t_preload = _time.perf_counter()
    horse_cache = build_horse_history_cache(session)
    jockey_cache = build_jockey_history_cache(session)
    trainer_cache = build_trainer_history_cache(session)
    pedigree_cache = build_pedigree_cache(session)
    log.info(
        "Pre-loaded: horse=%d, jockey=%d, trainer=%d rows, "
        "pedigree=%d horses (%d sires, %d dams) in %.1fs",
        len(horse_cache.df),
        len(jockey_cache.df),
        len(trainer_cache.df),
        len(pedigree_cache.horse_to_sire_dam),
        len(pedigree_cache.by_sire),
        len(pedigree_cache.by_dam),
        _time.perf_counter() - t_preload,
    )

    log.info("Building features for %d races (all DB lookups now bulk-cached)", n_total)

    rows: list[dict[str, object]] = []
    # 100 race ごとに進捗ログを出すのでユーザが進行状況を把握できる
    progress_step = max(50, n_total // 50)

    t0 = _time.perf_counter()
    for i, race in enumerate(races):
        entry_stmt = select(Entry).where(Entry.race_id == race.race_id)
        entries = list(session.scalars(entry_stmt).all())
        if not entries:
            continue
        rows.extend(_build_race_rows(
            session, race, entries,
            horse_cache=horse_cache,
            jockey_cache=jockey_cache,
            trainer_cache=trainer_cache,
            pedigree_cache=pedigree_cache,
        ))
        if (i + 1) % progress_step == 0 or (i + 1) == n_total:
            elapsed = _time.perf_counter() - t0
            eta_sec = elapsed / (i + 1) * (n_total - i - 1)
            log.info(
                "  feature progress: %d/%d races (%.0fs elapsed, ETA %.0fs)",
                i + 1,
                n_total,
                elapsed,
                eta_sec,
            )

    df = pd.DataFrame(rows)
    # Ensure all feature columns exist (fill with NaN if missing)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")

    if cache_active and cache_key is not None:
        _frame_cache_save(cache_key, df)

    return df


def build_inference_frame(session: Session, race_id: str) -> pd.DataFrame:
    """Build a feature DataFrame for a single race (no finish_position).

    Usable at entry-form stage — finish_position is excluded.
    Uses the race's own date as the cutoff for historical lookups.
    """
    race = session.get(Race, race_id)
    if race is None:
        raise ValueError(f"Race {race_id!r} not found")

    entry_stmt = select(Entry).where(Entry.race_id == race_id)
    entries = list(session.scalars(entry_stmt).all())

    rows = _build_race_rows(session, race, entries)
    for row in rows:
        row.pop("finish_position", None)

    df = pd.DataFrame(rows)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = float("nan")
    return df


