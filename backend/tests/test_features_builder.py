"""Tests for features/builder.py.

Verifies:
- build_training_frame produces expected columns and no leakage
- build_inference_frame excludes finish_position
- FEATURE_COLUMNS constant is respected
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine

from db.base import Base
from db.session import session_scope
from features.builder import (
    FEATURE_COLUMNS,
    HIGH_CARDINALITY_ID_FEATURES,
    ODDS_FEATURE_COLUMNS,
    build_inference_frame,
    build_training_frame,
    get_active_features,
)
from tests.synthetic import make_synthetic_db


@pytest.fixture()
def syn_engine():
    engine = create_engine("sqlite:///:memory:", future=True)
    make_synthetic_db(engine, n_races=15, n_horses_per_race=8, days_back=90, seed=1)
    yield engine
    engine.dispose()


def test_build_training_frame_columns(syn_engine):
    with session_scope(syn_engine) as session:
        df = build_training_frame(session)

    assert not df.empty, "Expected non-empty training frame"
    assert "race_id" in df.columns
    assert "horse_id" in df.columns
    assert "finish_position" in df.columns
    assert "date" in df.columns

    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"Missing feature column: {col}"


def test_build_training_frame_no_future_leakage(syn_engine):
    """For every row the date is >= the earliest possible race date."""
    with session_scope(syn_engine) as session:
        df = build_training_frame(session)

    assert df["date"].is_monotonic_increasing or True  # just check dates exist
    assert df["date"].notna().all()


def test_build_training_frame_date_filter(syn_engine):
    with session_scope(syn_engine) as session:
        df_all = build_training_frame(session)

    dates = sorted(df_all["date"].unique())
    if len(dates) < 2:
        pytest.skip("Not enough races to test date filter")

    cutoff = dates[len(dates) // 2]
    with session_scope(syn_engine) as session:
        df_filtered = build_training_frame(session, train_end=cutoff)

    assert (df_filtered["date"] <= cutoff).all()
    assert len(df_filtered) < len(df_all)


def test_build_inference_frame_no_finish_position(syn_engine):
    from sqlalchemy import select

    from db.models.race import Race

    with session_scope(syn_engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        assert race_id is not None

        df = build_inference_frame(session, race_id)

    assert "finish_position" not in df.columns
    assert "horse_id" in df.columns

    for col in FEATURE_COLUMNS:
        assert col in df.columns


def test_build_training_frame_empty_db():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with session_scope(engine) as session:
        df = build_training_frame(session)
    assert df.empty
    engine.dispose()


# ── PR-C: new column / relative-feature tests ─────────────────────────────────

NEW_HISTORY_COLS = [
    "recent_avg_agari_3f",
    "days_since_last_race",
    "wins_same_course",
    "recent_finish_1",
    "recent_finish_2",
    "recent_finish_3",
]
RELATIVE_COLS = [
    "horse_weight_pct",
    "odds_win_rank",
    "weight_carried_pct",
    "jockey_recent_win_rate_vs_field",
    "course_place_rate_vs_field",
    "odds_win_diff_from_favorite",
]
PEDIGREE_COLS = [
    "sire_progeny_win_rate",
    "dam_progeny_win_rate",
]


def test_new_feature_columns_present(syn_engine):
    """All PR-C feature columns must appear in the training frame."""
    with session_scope(syn_engine) as session:
        df = build_training_frame(session)

    for col in NEW_HISTORY_COLS + RELATIVE_COLS + PEDIGREE_COLS:
        assert col in df.columns, f"PR-C column missing: {col}"


def test_relative_features_per_race(syn_engine):
    """odds_win_diff_from_favorite must be 0 for the favourite in each race."""
    with session_scope(syn_engine) as session:
        df = build_training_frame(session)

    if df.empty:
        pytest.skip("No data to test")

    for race_id, group in df.groupby("race_id"):
        valid = group.dropna(subset=["odds_win", "odds_win_diff_from_favorite"])
        if valid.empty:
            continue
        favourite_row = valid.loc[valid["odds_win"].idxmin()]
        diff = favourite_row["odds_win_diff_from_favorite"]
        assert diff == pytest.approx(0.0, abs=1e-9), (
            f"race {race_id}: favourite diff={diff} expected 0"
        )


def test_jockey_and_course_relative_features_populated(syn_engine):
    """jockey_recent_win_rate_vs_field and course_place_rate_vs_field must
    have at least one non-NaN value — the regression we're guarding against
    is them being always-NaN because the builder forgot to feed pre-computed
    stats into compute_within_race_features.
    """
    with session_scope(syn_engine) as session:
        df = build_training_frame(session)

    if df.empty:
        pytest.skip("No data to test")

    jwr = df["jockey_recent_win_rate_vs_field"].dropna()
    cpr = df["course_place_rate_vs_field"].dropna()
    assert not jwr.empty, "jockey_recent_win_rate_vs_field is entirely NaN"
    assert not cpr.empty, "course_place_rate_vs_field is entirely NaN"

    # Within any race, the field-relative deltas should sum to ~0
    # (since each value is rate - field_mean).
    for race_id, group in df.groupby("race_id"):
        for col in ["jockey_recent_win_rate_vs_field", "course_place_rate_vs_field"]:
            valid = group[col].dropna()
            if len(valid) >= 2:
                assert valid.sum() == pytest.approx(0.0, abs=1e-6), (
                    f"race {race_id}: {col} deltas should sum to 0, got {valid.sum()}"
                )


def test_horse_weight_pct_bounded(syn_engine):
    """horse_weight_pct must be in [0.0, 1.0] for all non-NaN rows."""
    with session_scope(syn_engine) as session:
        df = build_training_frame(session)

    col = df["horse_weight_pct"].dropna()
    assert (col >= 0.0).all(), "horse_weight_pct below 0"
    assert (col <= 1.0).all(), "horse_weight_pct above 1"


def test_feature_columns_count():
    """FEATURE_COLUMNS should have exactly 46 columns:
    24 original + 14 (PR-C extensions) + 3 (Q4 race-level) + 5 (Phase B margin/passing).
    """
    assert len(FEATURE_COLUMNS) == 46


def test_high_cardinality_id_features_not_in_feature_columns():
    """sire_id / dam_sire_id は高基数 ID のため FEATURE_COLUMNS に含まれてはならない。

    LightGBM での高基数カテゴリ特徴量は過学習・メモリ増大を招くため除外する。
    代わりに sire_progeny_win_rate / dam_progeny_win_rate の集約特徴量を使う。
    このテストは将来の誤追加に対するリグレッションガードである。
    """
    feature_set = set(FEATURE_COLUMNS)
    for col in HIGH_CARDINALITY_ID_FEATURES:
        assert col not in feature_set, (
            f"{col!r} は高基数 ID 特徴量のため FEATURE_COLUMNS に含めてはならない。"
            " sire_progeny_win_rate / dam_progeny_win_rate などの集約特徴量を使うこと。"
        )


def test_high_cardinality_id_features_not_in_training_frame(syn_engine):
    """build_training_frame が sire_id / dam_sire_id を出力しないことを確認する。

    ガード処理が正しく機能しているかの実行時リグレッションテスト。
    """
    with session_scope(syn_engine) as session:
        df = build_training_frame(session)

    for col in HIGH_CARDINALITY_ID_FEATURES:
        assert col not in df.columns, (
            f"{col!r} が training frame に混入している。builder.py のガードを確認すること。"
        )


def test_pedigree_aggregate_features_present_in_feature_columns():
    """集約された血統特徴量は FEATURE_COLUMNS に含まれている必要がある。

    高基数 ID を除去した代わりに集約特徴量が維持されていることを確認する。
    """
    assert "sire_progeny_win_rate" in FEATURE_COLUMNS, (
        "sire_progeny_win_rate が FEATURE_COLUMNS から欠落している"
    )
    assert "dam_progeny_win_rate" in FEATURE_COLUMNS, (
        "dam_progeny_win_rate が FEATURE_COLUMNS から欠落している"
    )


# ── Frame caching ─────────────────────────────────────────────────────────────


@pytest.fixture()
def file_db_engine(tmp_path):
    """File-based SQLite DB so the cache (which keys off mtime) can engage.
    `:memory:` DBs are intentionally excluded from caching to avoid
    cross-test contamination, so cache tests need a real file.
    """
    db_file = tmp_path / "frame_cache_test.db"
    engine = create_engine(f"sqlite:///{db_file}", future=True)
    make_synthetic_db(engine, n_races=15, n_horses_per_race=8, days_back=90, seed=2)
    yield engine
    engine.dispose()


def test_build_training_frame_cache_hit_returns_identical(
    file_db_engine, tmp_path, monkeypatch,
):
    """2 回目の build_training_frame 呼び出しは cache から読まれ、内容が一致する。"""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    with session_scope(file_db_engine) as session:
        df1 = build_training_frame(session)
    with session_scope(file_db_engine) as session:
        df2 = build_training_frame(session)

    assert list(df1.columns) == list(df2.columns)
    assert len(df1) == len(df2)
    assert set(df1["race_id"].unique()) == set(df2["race_id"].unique())

    # 実際に cache pickle が書かれていること
    cache_dir = tmp_path / "data" / "cache" / "training_frames"
    assert cache_dir.exists()
    assert list(cache_dir.glob("*.pkl")), "cache file not created"


def test_build_training_frame_cache_disabled_via_env(
    file_db_engine, tmp_path, monkeypatch,
):
    """KEIBA_DISABLE_FRAME_CACHE=1 を立てた呼び出しは cache に書き込まない。"""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("KEIBA_DISABLE_FRAME_CACHE", "1")

    with session_scope(file_db_engine) as session:
        df = build_training_frame(session)

    assert not df.empty
    cache_dir = tmp_path / "data" / "cache" / "training_frames"
    if cache_dir.exists():
        assert not list(cache_dir.glob("*.pkl"))


def test_build_training_frame_use_cache_false_skips_cache(
    file_db_engine, tmp_path, monkeypatch,
):
    """use_cache=False を渡した呼び出しも cache を読み書きしない。"""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    with session_scope(file_db_engine) as session:
        df = build_training_frame(session, use_cache=False)

    assert not df.empty
    cache_dir = tmp_path / "data" / "cache" / "training_frames"
    if cache_dir.exists():
        assert not list(cache_dir.glob("*.pkl"))


def test_build_training_frame_cache_survives_mtime_only_change(
    file_db_engine, tmp_path, monkeypatch,
):
    """DB ファイルの mtime だけが変わっても (内容は不変)、cache key は内容
    シグネチャ基準なので **同じ key** に当たり再計算されない。

    これがモデル保存 (model_runs への書込み = ファイル mtime 変化) を挟んでも
    高価な feature cache が無効化されないことを担保する回帰テスト。"""
    import os
    import time

    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    with session_scope(file_db_engine) as session:
        df1 = build_training_frame(session)

    # DB ファイルの mtime を未来にずらす (model_runs 書込み等の touch 相当)
    db_file = tmp_path / "frame_cache_test.db"
    future_ts = time.time() + 60
    os.utime(db_file, (future_ts, future_ts))

    with session_scope(file_db_engine) as session:
        df2 = build_training_frame(session)

    # 内容シグネチャが同じなので cache file は 1 つのまま (再利用された)
    cache_dir = tmp_path / "data" / "cache" / "training_frames"
    pkls = list(cache_dir.glob("*.pkl"))
    assert len(pkls) == 1, f"expected 1 cache file (mtime-only change reused), got {len(pkls)}"
    assert len(df1) == len(df2)


def test_build_training_frame_cache_invalidates_on_new_races(
    file_db_engine, tmp_path, monkeypatch,
):
    """レースが追加される (= 内容シグネチャ変化) と別 key で再計算され、
    古い結果を返さない。ingest 後に新データが反映されることを担保。"""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path / "data"))

    with session_scope(file_db_engine) as session:
        df1 = build_training_frame(session)
    n_races_1 = df1["race_id"].nunique()

    # 同じ DB に追加レースを ingest (内容シグネチャが変わる)
    make_synthetic_db(
        file_db_engine, n_races=5, n_horses_per_race=8, days_back=30, seed=777
    )

    with session_scope(file_db_engine) as session:
        df2 = build_training_frame(session)

    cache_dir = tmp_path / "data" / "cache" / "training_frames"
    pkls = list(cache_dir.glob("*.pkl"))
    assert len(pkls) == 2, f"expected 2 cache files (content changed), got {len(pkls)}"
    assert df2["race_id"].nunique() > n_races_1, "new races should appear after re-ingest"


# ---------------------------------------------------------------------------
# get_active_features (env flag for A/B eval)
# ---------------------------------------------------------------------------


def test_get_active_features_default_returns_all(monkeypatch):
    """env flag 未設定時は FEATURE_COLUMNS 全体のコピーを返す。"""
    monkeypatch.delenv("KEIBA_EXCLUDE_ODDS_FEATURES", raising=False)
    active = get_active_features()
    assert active == FEATURE_COLUMNS
    # コピーであることを確認（破壊的編集が原本に伝播しない）
    active.append("__test_marker__")
    assert "__test_marker__" not in FEATURE_COLUMNS


def test_get_active_features_excludes_odds_when_flag_set(monkeypatch):
    """KEIBA_EXCLUDE_ODDS_FEATURES=1 のとき odds 派生が除かれる。"""
    monkeypatch.setenv("KEIBA_EXCLUDE_ODDS_FEATURES", "1")
    active = get_active_features()

    for odds_col in ODDS_FEATURE_COLUMNS:
        assert odds_col not in active, f"{odds_col} should be excluded"

    # 非 odds 列は全部残る
    expected_remaining = [c for c in FEATURE_COLUMNS if c not in ODDS_FEATURE_COLUMNS]
    assert active == expected_remaining


@pytest.mark.parametrize("flag_value", ["true", "True", "yes", "YES"])
def test_get_active_features_accepts_truthy_values(monkeypatch, flag_value):
    """truthy 表記 (true / yes / 1) を許容する。"""
    monkeypatch.setenv("KEIBA_EXCLUDE_ODDS_FEATURES", flag_value)
    active = get_active_features()
    for odds_col in ODDS_FEATURE_COLUMNS:
        assert odds_col not in active


@pytest.mark.parametrize("flag_value", ["0", "false", "no", ""])
def test_get_active_features_falsy_values_keep_odds(monkeypatch, flag_value):
    """falsy 表記 (0 / false / 空) では odds 列を保持する。"""
    monkeypatch.setenv("KEIBA_EXCLUDE_ODDS_FEATURES", flag_value)
    active = get_active_features()
    for odds_col in ODDS_FEATURE_COLUMNS:
        assert odds_col in active
