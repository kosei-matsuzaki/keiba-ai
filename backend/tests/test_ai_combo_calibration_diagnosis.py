"""Smoke tests for ai/combo_calibration_diagnosis.py.

Focus on the bundle-first refactor: ensure diagnose_combo_calibration runs
end-to-end on a tiny synthetic GBDT model and returns the expected schema.
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine

import db.models  # noqa: F401
from ai.combo_calibration_diagnosis import _is_hit, diagnose_combo_calibration
from tests.synthetic import make_synthetic_db, train_synthetic_nn

# ---------------------------------------------------------------------------
# _is_hit unit cases — sanity checks on combo string parsing
# ---------------------------------------------------------------------------


def test_is_hit_umaren_match():
    # Top-3 (finish=1: pp=3, finish=2: pp=7, finish=3: pp=5)
    top3 = [(1, 3), (2, 7), (3, 5)]
    assert _is_hit("馬連", "3-7", top3) is True
    assert _is_hit("馬連", "7-3", top3) is True  # order-insensitive
    assert _is_hit("馬連", "3-5", top3) is False  # 3 着馬は対象外


def test_is_hit_wide_includes_third():
    top3 = [(1, 3), (2, 7), (3, 5)]
    assert _is_hit("ワイド", "3-7", top3) is True
    assert _is_hit("ワイド", "3-5", top3) is True
    assert _is_hit("ワイド", "7-5", top3) is True
    assert _is_hit("ワイド", "3-9", top3) is False  # 9 は top3 外


def test_is_hit_umatan_order_sensitive():
    top3 = [(1, 3), (2, 7), (3, 5)]
    assert _is_hit("馬単", "3→7", top3) is True
    assert _is_hit("馬単", "7→3", top3) is False  # 順序が逆


def test_is_hit_sanrenpuku_unordered():
    top3 = [(1, 3), (2, 7), (3, 5)]
    assert _is_hit("三連複", "3-5-7", top3) is True
    assert _is_hit("三連複", "5-3-7", top3) is True
    assert _is_hit("三連複", "1-3-5", top3) is False


def test_is_hit_sanrentan_ordered():
    top3 = [(1, 3), (2, 7), (3, 5)]
    assert _is_hit("三連単", "3→7→5", top3) is True
    assert _is_hit("三連単", "3→5→7", top3) is False


def test_is_hit_invalid_combo_returns_false():
    """壊れた combo 文字列で例外を投げず False を返す。"""
    top3 = [(1, 3), (2, 7), (3, 5)]
    assert _is_hit("馬連", "abc-def", top3) is False
    assert _is_hit("三連単", "", top3) is False


# ---------------------------------------------------------------------------
# End-to-end smoke test (bundle-first, NN path)
# ---------------------------------------------------------------------------


def test_diagnose_combo_calibration_runs_via_bundle_on_nn(tmp_path):
    """End-to-end: 合成 DB に対して tiny NN を学習し、
    diagnose_combo_calibration を bundle 経由で実行する。

    bundle-first refactor のリグレッション gate。NN bundle で
    predict_race_with_combinations が動くことを担保する。
    """
    db_file = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_file}", future=True)
    make_synthetic_db(engine, n_races=30, n_horses_per_race=10, days_back=180, seed=42)

    os.environ["KEIBA_DATA_DIR"] = str(tmp_path / "data")
    model_dir = train_synthetic_nn(db_file)

    diag = diagnose_combo_calibration(model_path=model_dir, db=db_file)

    assert "model_path" in diag
    assert "n_races" in diag
    assert "results" in diag
    assert diag["n_races"] > 0
    # 5 馬券種すべてのキーが存在 (空でもよい)
    for bt in ("馬連", "ワイド", "馬単", "三連複", "三連単"):
        assert bt in diag["results"], f"missing bet type: {bt}"
        info = diag["results"][bt]
        assert "n_combos" in info
        assert "buckets" in info
        assert "brier" in info
