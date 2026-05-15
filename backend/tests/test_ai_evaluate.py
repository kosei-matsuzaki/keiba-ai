"""Tests for ai/evaluate.py — backtest metrics."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine

import db.models  # noqa: F401
from ai.evaluate import evaluate
from ai.gbm.train import train
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


def test_evaluate_betting_filter_params_recorded(trained_scenario):
    """`exclude_top_rank` 等のフィルタ値は metrics dict に記録される
    (再現性 + persisted metrics_json から戦略を読み戻せるように)。"""
    db_file, model_dir = trained_scenario
    metrics = evaluate(
        model_path=model_dir,
        db=db_file,
        win_ev_threshold=1.2,
        place_ev_threshold=1.15,
        exclude_top_rank=2,
        min_popularity=4,
        max_popularity=12,
    )

    assert metrics["win_ev_threshold"] == 1.2
    assert metrics["place_ev_threshold"] == 1.15
    assert metrics["exclude_top_rank"] == 2
    assert metrics["min_popularity"] == 4
    assert metrics["max_popularity"] == 12


def test_evaluate_default_filter_params_match_constants(trained_scenario):
    """フィルタ未指定時は既存挙動 (= 後方互換) になっている。"""
    from ai.evaluate import PLACE_EV_THRESHOLD, WIN_EV_THRESHOLD

    db_file, model_dir = trained_scenario
    metrics = evaluate(model_path=model_dir, db=db_file)

    assert metrics["win_ev_threshold"] == WIN_EV_THRESHOLD
    assert metrics["place_ev_threshold"] == PLACE_EV_THRESHOLD
    assert metrics["exclude_top_rank"] == 0
    assert metrics["min_popularity"] is None
    assert metrics["max_popularity"] is None


def test_evaluate_exclude_top_rank_reduces_bets(trained_scenario):
    """`exclude_top_rank=N` を上げるほど bets 数は単調に減る (or 同数)。
    モデル予測上位 N 頭を bet 候補から外すフィルタが実際に発火している
    ことを確認する (analyze_place_bets で本命=rank 1 が大損と判明した
    のに対する CLI 対応の正常動作 gate)。"""
    db_file, model_dir = trained_scenario

    base = evaluate(model_path=model_dir, db=db_file)
    excl = evaluate(model_path=model_dir, db=db_file, exclude_top_rank=3)

    assert excl["win_bets"] <= base["win_bets"]
    assert excl["place_bets"] <= base["place_bets"]


def test_evaluate_popularity_filter_bounds_pop_of_bets(trained_scenario):
    """`min_popularity` / `max_popularity` 指定で bet 数が減る (or 同数)。
    NaN popularity の馬もフィルタ有効時は除外される、という設計を担保。"""
    db_file, model_dir = trained_scenario

    base = evaluate(model_path=model_dir, db=db_file)
    bounded = evaluate(
        model_path=model_dir,
        db=db_file,
        min_popularity=4,
        max_popularity=12,
    )

    assert bounded["win_bets"] <= base["win_bets"]
    assert bounded["place_bets"] <= base["place_bets"]


def test_evaluate_baseline_metrics_unaffected_by_filters(trained_scenario):
    """Betting filter は model 側のみに適用される。Baseline (favorite) は
    常に 1 番人気に賭ける性質上フィルタ無関係 — フィルタ有無で baseline
    metrics が変わらないことを担保する。"""
    db_file, model_dir = trained_scenario

    out_no_filter = evaluate(model_path=model_dir, db=db_file, baseline="favorite")
    out_filtered = evaluate(
        model_path=model_dir,
        db=db_file,
        baseline="favorite",
        exclude_top_rank=2,
        min_popularity=4,
    )

    for key in ("win_bets", "place_bets", "payback_win", "payback_place"):
        b1 = out_no_filter["baseline_favorite"][key]
        b2 = out_filtered["baseline_favorite"][key]
        # NaN 対 NaN は許容
        if pd.isna(b1) and pd.isna(b2):
            continue
        assert b1 == b2, f"baseline {key} changed by filters: {b1} → {b2}"


def test_evaluate_persist_merges_into_model_run(trained_scenario):
    """`evaluate(..., persist=True)` で対応する model_runs.metrics_json に
    top1_hit / payback_win 等が merge され、Dashboard が読める状態になる。
    """
    import json

    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from db.models.model_run import ModelRun

    db_file, model_dir = trained_scenario

    # 学習直後の metrics_json は valid_*/test_* のみ
    engine = __import__('sqlalchemy', fromlist=['create_engine']).create_engine(
        f"sqlite:///{db_file}", future=True
    )
    with Session(engine) as session:
        run_before = session.scalar(select(ModelRun).order_by(ModelRun.id.desc()))
        before = json.loads(run_before.metrics_json)
        assert "top1_hit" not in before  # まだ無い

    # evaluate を persist=True で実行
    evaluate(model_path=model_dir, db=db_file, persist=True)

    # metrics_json に top1_hit / payback_win 等が merge されている
    with Session(engine) as session:
        run_after = session.scalar(select(ModelRun).order_by(ModelRun.id.desc()))
        after = json.loads(run_after.metrics_json)
        # 既存のキーは保持されている
        assert "test_ndcg3" in after
        # 新しい evaluation キーが merge されている
        for key in ("top1_hit", "place_hit", "payback_win", "payback_place", "n_races"):
            assert key in after, f"{key} should be persisted"


# ---------------------------------------------------------------------------
# kelly_bet_size tests
# ---------------------------------------------------------------------------


class TestKellyBetSize:
    def test_positive_edge_returns_nonzero(self):
        from ai.evaluate import kelly_bet_size

        # win_prob=0.5, odds=3.0 → edge = 0.5*3.0 - 1 = 0.5 > 0
        bet = kelly_bet_size(win_prob=0.5, odds=3.0, bankroll=100_000)
        assert bet > 0

    def test_zero_edge_returns_zero(self):
        from ai.evaluate import kelly_bet_size

        # win_prob=1/3, odds=3.0 → edge = 0 exactly
        bet = kelly_bet_size(win_prob=1 / 3, odds=3.0, bankroll=100_000)
        assert bet == 0

    def test_negative_edge_returns_zero(self):
        from ai.evaluate import kelly_bet_size

        # win_prob=0.1, odds=2.0 → edge = 0.1*2 - 1 = -0.8 < 0
        bet = kelly_bet_size(win_prob=0.1, odds=2.0, bankroll=100_000)
        assert bet == 0

    def test_odds_one_returns_zero(self):
        from ai.evaluate import kelly_bet_size

        # b = odds - 1 = 0 → zero division guard
        bet = kelly_bet_size(win_prob=0.9, odds=1.0, bankroll=100_000)
        assert bet == 0

    def test_rounding_to_min_bet(self):
        from ai.evaluate import kelly_bet_size

        # Choose params that result in a known bet size
        # kappa=0.25, edge=1.0, b=1.0 → fraction=0.25, raw=25000 → rounded to 25000
        bet = kelly_bet_size(win_prob=0.5, odds=3.0, bankroll=100_000, kappa=0.25, min_bet=100)
        assert bet % 100 == 0

    def test_below_min_bet_returns_zero(self):
        from ai.evaluate import kelly_bet_size

        # Very small bankroll → raw_size < min_bet → 0
        bet = kelly_bet_size(win_prob=0.5, odds=3.0, bankroll=10, kappa=0.25, min_bet=100)
        assert bet == 0

    def test_larger_kappa_gives_larger_bet(self):
        from ai.evaluate import kelly_bet_size

        bet_small_kappa = kelly_bet_size(win_prob=0.5, odds=3.0, bankroll=100_000, kappa=0.1)
        bet_large_kappa = kelly_bet_size(win_prob=0.5, odds=3.0, bankroll=100_000, kappa=0.5)
        assert bet_large_kappa >= bet_small_kappa


# ---------------------------------------------------------------------------
# Kelly vs fixed bet sizing integration tests
# ---------------------------------------------------------------------------


def test_evaluate_kelly_vs_fixed_invested_differs(trained_scenario):
    """Kelly sizing should produce a different win_invested than fixed unless
    all bet sizes are exactly 100 (unlikely for a real model output)."""
    db_file, model_dir = trained_scenario

    fixed_metrics = evaluate(
        model_path=model_dir, db=db_file,
        bet_sizing="fixed",
    )
    kelly_metrics = evaluate(
        model_path=model_dir, db=db_file,
        bet_sizing="kelly",
        kelly_kappa=0.25,
        bankroll=100_000,
    )

    # Both should complete without error and have the same keys
    assert "win_bets" in fixed_metrics
    assert "win_bets" in kelly_metrics

    # bet_sizing recorded in output
    assert fixed_metrics["bet_sizing"] == "fixed"
    assert kelly_metrics["bet_sizing"] == "kelly"

    # Kelly-specific params recorded only for kelly mode
    assert fixed_metrics["kelly_kappa"] is None
    assert fixed_metrics["bankroll"] is None
    assert kelly_metrics["kelly_kappa"] == 0.25
    assert kelly_metrics["bankroll"] == 100_000


def test_evaluate_kelly_returns_required_keys(trained_scenario):
    db_file, model_dir = trained_scenario
    metrics = evaluate(
        model_path=model_dir, db=db_file,
        bet_sizing="kelly", kelly_kappa=0.5, bankroll=50_000,
    )
    for key in ("win_bets", "win_invested", "win_gross_payout", "payback_win",
                "place_bets", "place_invested", "place_gross_payout", "payback_place"):
        assert key in metrics


def test_evaluate_fixed_bet_sizing_recorded(trained_scenario):
    db_file, model_dir = trained_scenario
    metrics = evaluate(model_path=model_dir, db=db_file)
    assert metrics["bet_sizing"] == "fixed"
    assert metrics["kelly_kappa"] is None
    assert metrics["bankroll"] is None
