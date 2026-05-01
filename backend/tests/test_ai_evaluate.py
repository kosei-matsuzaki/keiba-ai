"""Tests for ai/evaluate.py — backtest metrics."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from sqlalchemy import create_engine

import keiba_ai.db.models  # noqa: F401
from keiba_ai.ai.evaluate import evaluate
from keiba_ai.ai.train import train
from tests.synthetic import make_synthetic_db


@pytest.fixture()
def trained_scenario(tmp_path):
    db_file = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_file}", future=True)
    make_synthetic_db(engine, n_races=30, n_horses_per_race=10, days_back=180, seed=99)

    os.environ["KEIBA_DATA_DIR"] = str(tmp_path / "data")
    result = train(db=db_file, train_end=None, valid_months=2, test_months=1)
    model_dir = Path(result["model_dir"])
    return db_file, model_dir


def test_evaluate_returns_all_metric_keys(trained_scenario):
    db_file, model_dir = trained_scenario
    metrics = evaluate(model_path=model_dir, db=db_file)

    required_keys = {
        "ndcg1", "ndcg3", "top1_hit", "place_hit",
        "win_bets", "win_invested", "win_gross_payout", "payback_win", "n_races",
    }
    assert required_keys.issubset(metrics.keys()), (
        f"Missing keys: {required_keys - metrics.keys()}"
    )


def test_evaluate_payback_semantics(trained_scenario):
    """payback_win = gross_payout / invested。1.0 が損益分岐点。"""
    db_file, model_dir = trained_scenario
    metrics = evaluate(model_path=model_dir, db=db_file)

    if metrics["win_bets"] == 0:
        return  # ベットが発生しない synthetic データの場合はスキップ

    expected = metrics["win_gross_payout"] / metrics["win_invested"]
    assert abs(metrics["payback_win"] - expected) < 1e-9
    assert metrics["payback_win"] >= 0.0  # 払戻金は非負


def test_evaluate_n_races_positive(trained_scenario):
    db_file, model_dir = trained_scenario
    metrics = evaluate(model_path=model_dir, db=db_file)
    assert metrics["n_races"] > 0


def test_evaluate_hit_rates_in_range(trained_scenario):
    db_file, model_dir = trained_scenario
    metrics = evaluate(model_path=model_dir, db=db_file)

    import math
    if not math.isnan(metrics["top1_hit"]):
        assert 0.0 <= metrics["top1_hit"] <= 1.0
    if not math.isnan(metrics["place_hit"]):
        assert 0.0 <= metrics["place_hit"] <= 1.0


def test_evaluate_with_baseline_returns_nested(trained_scenario):
    """baseline=='favorite' でネスト dict {model, baseline_favorite, delta} を返す。"""
    db_file, model_dir = trained_scenario
    out = evaluate(model_path=model_dir, db=db_file, baseline="favorite")

    assert set(out.keys()) == {"model", "baseline_favorite", "delta"}

    flat_keys = {
        "ndcg1", "ndcg3", "top1_hit", "place_hit",
        "win_bets", "win_invested", "win_gross_payout", "payback_win",
        "place_bets", "place_invested", "place_gross_payout", "payback_place",
        "n_races",
    }
    assert flat_keys.issubset(out["model"].keys())
    assert flat_keys.issubset(out["baseline_favorite"].keys())

    # baseline は毎レース 1 番人気に win/place を 1 ベットずつ。
    # 評価対象に有効レースが存在する場合 win_bets > 0 になる
    if out["baseline_favorite"]["n_races"] > 0:
        assert out["baseline_favorite"]["win_bets"] == out["baseline_favorite"]["n_races"]


def test_evaluate_baseline_unchanged_default(trained_scenario):
    """baseline を渡さなければ既存どおり flat dict を返す（後方互換）。"""
    db_file, model_dir = trained_scenario
    out = evaluate(model_path=model_dir, db=db_file)
    # ネスト dict の頂上キーが入っていないことを確認
    assert "model" not in out
    assert "baseline_favorite" not in out
    assert "delta" not in out
    assert "ndcg1" in out  # flat 構造


def test_evaluate_baseline_delta_consistency(trained_scenario):
    """delta = model − baseline の関係が成立する（NaN 以外）。"""
    db_file, model_dir = trained_scenario
    out = evaluate(model_path=model_dir, db=db_file, baseline="favorite")

    import math
    for key in ["ndcg1", "ndcg3", "top1_hit", "place_hit", "payback_win", "payback_place"]:
        d = out["delta"][key]
        m = out["model"][key]
        b = out["baseline_favorite"][key]
        if math.isnan(m) or math.isnan(b):
            assert math.isnan(d), f"{key}: expected NaN delta when input is NaN"
        else:
            assert abs(d - (m - b)) < 1e-9, f"{key}: delta should be model − baseline"
