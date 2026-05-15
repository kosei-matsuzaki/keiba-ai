"""Tests for predict_race_with_combinations_gbdt and derive_wide_prob_from_triple."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

import db.models  # noqa: F401
from ai.gbm.train import train
from ai.predict import (
    CombinationPrediction,
    derive_wide_prob_from_triple,
    predict_race_with_combinations_gbdt,
)
from ai.registry import load_model
from db.base import Base
from db.models.race import Race
from features.builder import build_inference_frame
from tests.synthetic import make_synthetic_db

_RNG = np.random.default_rng(0)
_N_SAMPLES = 2_000  # reduced for test speed


@pytest.fixture(scope="module")
def trained_combo_model(tmp_path_factory):
    """Train a small model once and reuse across tests in this module."""
    tmp_path = tmp_path_factory.mktemp("combo_model")
    db_file = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_file}", future=True)
    make_synthetic_db(engine, n_races=30, n_horses_per_race=8, days_back=180, seed=11)

    import os
    os.environ["KEIBA_DATA_DIR"] = str(tmp_path / "data")

    result = train(db=db_file, train_end=None, valid_months=2, test_months=1)
    model_dir = Path(result["model_dir"])
    return engine, db_file, model_dir


# ---------------------------------------------------------------------------
# derive_wide_prob_from_triple
# ---------------------------------------------------------------------------

def test_derive_wide_prob_symmetry():
    """wide_matrix must be symmetric."""
    rng = np.random.default_rng(99)
    scores = rng.standard_normal(8)
    from ai.calibrate import compute_all_combination_probs
    probs = compute_all_combination_probs(scores, k=3, n_samples=5_000, rng=rng)
    wide = derive_wide_prob_from_triple(probs["triple"], len(scores))
    np.testing.assert_allclose(wide, wide.T, atol=1e-12, err_msg="wide matrix is not symmetric")


def test_derive_wide_prob_diagonal_zero():
    """Diagonal of wide matrix must be 0 (horse cannot pair with itself)."""
    rng = np.random.default_rng(77)
    scores = rng.standard_normal(6)
    from ai.calibrate import compute_all_combination_probs
    probs = compute_all_combination_probs(scores, k=3, n_samples=3_000, rng=rng)
    wide = derive_wide_prob_from_triple(probs["triple"], len(scores))
    np.testing.assert_allclose(np.diag(wide), 0.0, atol=1e-12)


def test_derive_wide_prob_values_in_range():
    """All wide probabilities must be in [0, 1]."""
    rng = np.random.default_rng(55)
    scores = rng.standard_normal(6)
    from ai.calibrate import compute_all_combination_probs
    probs = compute_all_combination_probs(scores, k=3, n_samples=3_000, rng=rng)
    wide = derive_wide_prob_from_triple(probs["triple"], len(scores))
    assert (wide >= 0).all()
    assert (wide <= 1.0 + 1e-9).all()


def test_derive_wide_prob_sum_geq_triple_sum():
    """Sum of wide probs >= sum of triple_prob (each triple contributes to 3 pairs)."""
    rng = np.random.default_rng(33)
    scores = rng.standard_normal(7)
    from ai.calibrate import compute_all_combination_probs
    probs = compute_all_combination_probs(scores, k=3, n_samples=5_000, rng=rng)
    wide = derive_wide_prob_from_triple(probs["triple"], len(scores))
    triple_total = sum(probs["triple"].values())
    # Each triple adds to 3 pairs (both directions) → sum of off-diagonal / 2 = 3 * triple_total
    off_diag_sum = (wide.sum() - np.diag(wide).sum()) / 2
    assert off_diag_sum == pytest.approx(3 * triple_total, rel=0.01)


# ---------------------------------------------------------------------------
# predict_race_with_combinations_gbdt output schema
# ---------------------------------------------------------------------------

def test_predict_combinations_output_keys(trained_combo_model):
    """All 7 bet type keys must be present in the result."""
    engine, db_file, model_dir = trained_combo_model
    model = load_model(model_dir)

    with Session(engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        frame = build_inference_frame(session, race_id)
        result = predict_race_with_combinations_gbdt(
            model, frame, session=session, n_samples=_N_SAMPLES, rng=np.random.default_rng(1)
        )

    expected_keys = {"単勝", "複勝", "馬連", "ワイド", "馬単", "三連複", "三連単"}
    assert set(result.keys()) == expected_keys


def test_predict_combinations_items_are_dataclass(trained_combo_model):
    """Each element in each list must be a CombinationPrediction.

    When no race_odds is provided, est_odds and ev are None (no baseline fallback).
    """
    engine, db_file, model_dir = trained_combo_model
    model = load_model(model_dir)

    with Session(engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        frame = build_inference_frame(session, race_id)
        result = predict_race_with_combinations_gbdt(
            model, frame, session=session, n_samples=_N_SAMPLES, rng=np.random.default_rng(2)
        )

    for bet_type, preds in result.items():
        for p in preds:
            assert isinstance(p, CombinationPrediction), f"{bet_type}: item is not CombinationPrediction"
            assert isinstance(p.combo, str)
            assert isinstance(p.prob, float)
            # est_odds and ev are None when no race_odds provided (no baseline fallback)
            assert p.est_odds is None
            assert p.ev is None
            assert isinstance(p.post_positions, tuple)


def test_predict_combinations_ev_equals_prob_times_odds(trained_combo_model):
    """When race_odds is provided, EV must equal prob * est_odds for confirmed combos."""
    engine, db_file, model_dir = trained_combo_model
    model = load_model(model_dir)

    with Session(engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        frame = build_inference_frame(session, race_id)
        # Provide synthetic race_odds so some combos have confirmed odds
        race_odds = {"単勝": {"1": 5.0, "2": 8.0, "3": 12.0}}
        result = predict_race_with_combinations_gbdt(
            model, frame, session=session, n_samples=_N_SAMPLES, rng=np.random.default_rng(3),
            race_odds=race_odds,
        )

    # For 単勝 combos that have confirmed odds, ev == prob * est_odds
    for p in result.get("単勝", []):
        if p.est_odds is not None and p.ev is not None:
            assert p.ev == pytest.approx(p.prob * p.est_odds, rel=1e-5), (
                f"単勝 {p.combo}: ev mismatch"
            )
        else:
            # combos without confirmed odds must have both None
            assert p.est_odds is None
            assert p.ev is None


def test_predict_combinations_probs_in_range(trained_combo_model):
    """All probabilities must be in [0, 1]."""
    engine, db_file, model_dir = trained_combo_model
    model = load_model(model_dir)

    with Session(engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        frame = build_inference_frame(session, race_id)
        result = predict_race_with_combinations_gbdt(
            model, frame, session=session, n_samples=_N_SAMPLES, rng=np.random.default_rng(4)
        )

    for bet_type, preds in result.items():
        for p in preds:
            assert 0.0 <= p.prob <= 1.0 + 1e-9, f"{bet_type} {p.combo}: prob={p.prob} out of range"


def test_predict_combinations_tansho_probs_sum_to_one(trained_combo_model):
    """Win probabilities across all horses should sum to ~1."""
    engine, db_file, model_dir = trained_combo_model
    model = load_model(model_dir)

    with Session(engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        frame = build_inference_frame(session, race_id)
        result = predict_race_with_combinations_gbdt(
            model, frame, session=session, n_samples=_N_SAMPLES, rng=np.random.default_rng(5)
        )

    total = sum(p.prob for p in result["単勝"])
    assert total == pytest.approx(1.0, abs=1e-5)


def test_predict_combinations_umaren_prob_sum_approx(trained_combo_model):
    """Sum of 馬連 probs should be close to 1 (exactly one pair wins top-2)."""
    engine, db_file, model_dir = trained_combo_model
    model = load_model(model_dir)

    with Session(engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        frame = build_inference_frame(session, race_id)
        result = predict_race_with_combinations_gbdt(
            model, frame, session=session, n_samples=_N_SAMPLES, rng=np.random.default_rng(6)
        )

    total = sum(p.prob for p in result["馬連"])
    assert total == pytest.approx(1.0, abs=0.05)


def test_predict_combinations_sanrenpuku_prob_sum_approx(trained_combo_model):
    """Sum of 三連複 probs should be close to 1 (exactly one triple wins top-3)."""
    engine, db_file, model_dir = trained_combo_model
    model = load_model(model_dir)

    with Session(engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        frame = build_inference_frame(session, race_id)
        result = predict_race_with_combinations_gbdt(
            model, frame, session=session, n_samples=_N_SAMPLES, rng=np.random.default_rng(7)
        )

    total = sum(p.prob for p in result["三連複"])
    assert total == pytest.approx(1.0, abs=0.05)


def test_predict_combinations_sorted_ev_none_last(trained_combo_model):
    """When race_odds=None, all ev are None and sorted at the end (all rows equal).

    When race_odds is provided, non-None ev rows appear before None ev rows,
    and among non-None rows they are sorted descending.
    """
    engine, db_file, model_dir = trained_combo_model
    model = load_model(model_dir)

    with Session(engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        frame = build_inference_frame(session, race_id)
        # All combos get None ev when no race_odds provided
        result_no_odds = predict_race_with_combinations_gbdt(
            model, frame, session=session, n_samples=_N_SAMPLES, rng=np.random.default_rng(8)
        )
        # With race_odds: 単勝 gets confirmed odds, others are None
        race_odds = {"単勝": {str(i + 1): float(5 + i) for i in range(len(frame))}}
        result_with_odds = predict_race_with_combinations_gbdt(
            model, frame, session=session, n_samples=_N_SAMPLES, rng=np.random.default_rng(8),
            race_odds=race_odds,
        )

    # Without race_odds: all ev are None
    for bet_type, preds in result_no_odds.items():
        assert all(p.ev is None for p in preds), f"{bet_type}: expected all None ev"

    # With race_odds: 単勝 has all non-None ev, sorted descending
    tansho = result_with_odds["単勝"]
    non_none_evs = [p.ev for p in tansho if p.ev is not None]
    assert non_none_evs == sorted(non_none_evs, reverse=True), "単勝: non-None ev not sorted descending"
    # None ev rows must appear after all non-None ev rows
    saw_none = False
    for p in tansho:
        if p.ev is None:
            saw_none = True
        elif saw_none:
            pytest.fail(f"単勝: non-None ev row after None ev row: {p}")


def test_predict_combinations_top_k(trained_combo_model):
    """top_k_combinations limits the output to at most K entries per bet type."""
    engine, db_file, model_dir = trained_combo_model
    model = load_model(model_dir)

    with Session(engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        frame = build_inference_frame(session, race_id)
        result = predict_race_with_combinations_gbdt(
            model, frame, session=session, n_samples=_N_SAMPLES,
            rng=np.random.default_rng(9), top_k_combinations=5,
        )

    for bet_type, preds in result.items():
        assert len(preds) <= 5, f"{bet_type}: expected <= 5 items, got {len(preds)}"


def test_predict_combinations_combo_format_umaren(trained_combo_model):
    """馬連 combo strings must be 'low-high' format with ascending post_positions."""
    engine, db_file, model_dir = trained_combo_model
    model = load_model(model_dir)

    with Session(engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        frame = build_inference_frame(session, race_id)
        result = predict_race_with_combinations_gbdt(
            model, frame, session=session, n_samples=_N_SAMPLES, rng=np.random.default_rng(10)
        )

    for p in result["馬連"]:
        parts = p.combo.split("-")
        assert len(parts) == 2, f"馬連 combo should have 2 parts: {p.combo}"
        a, b = int(parts[0]), int(parts[1])
        assert a < b, f"馬連 combo must be ascending: {p.combo}"
        assert p.post_positions == (a, b)


def test_predict_combinations_combo_format_umatan(trained_combo_model):
    """馬単 combo strings must be 'a→b' format."""
    engine, db_file, model_dir = trained_combo_model
    model = load_model(model_dir)

    with Session(engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        frame = build_inference_frame(session, race_id)
        result = predict_race_with_combinations_gbdt(
            model, frame, session=session, n_samples=_N_SAMPLES, rng=np.random.default_rng(11)
        )

    for p in result["馬単"]:
        assert "→" in p.combo, f"馬単 combo must use '→': {p.combo}"
        parts = p.combo.split("→")
        assert len(parts) == 2
        assert p.post_positions == (int(parts[0]), int(parts[1]))


def test_predict_combinations_combo_format_sanrentan(trained_combo_model):
    """三連単 combo strings must be 'a→b→c' format."""
    engine, db_file, model_dir = trained_combo_model
    model = load_model(model_dir)

    with Session(engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        frame = build_inference_frame(session, race_id)
        result = predict_race_with_combinations_gbdt(
            model, frame, session=session, n_samples=_N_SAMPLES,
            rng=np.random.default_rng(12), top_k_combinations=20,
        )

    for p in result["三連単"]:
        parts = p.combo.split("→")
        assert len(parts) == 3, f"三連単 combo must have 3 parts: {p.combo}"
        assert p.post_positions == tuple(int(x) for x in parts)


def test_predict_combinations_performance(trained_combo_model, monkeypatch):
    """Full predict_race_with_combinations_gbdt (10k samples, 8 horses) must complete within 200ms."""
    engine, db_file, model_dir = trained_combo_model
    model = load_model(model_dir)

    monkeypatch.setenv("KEIBA_PLACE_PROB_METHOD", "plackett_luce")

    with Session(engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        frame = build_inference_frame(session, race_id)

    # Warm-up to exclude first-call overhead
    predict_race_with_combinations_gbdt(
        model, frame, n_samples=100, rng=np.random.default_rng(0)
    )

    start = time.perf_counter()
    predict_race_with_combinations_gbdt(
        model, frame, n_samples=10_000, rng=np.random.default_rng(0)
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < 200, f"predict_race_with_combinations_gbdt took {elapsed_ms:.1f} ms, expected < 200 ms"


def test_predict_combinations_wide_greater_or_equal_pair(trained_combo_model):
    """P(wide) >= P(馬連) for each pair because wide requires top-3, pair requires top-2."""
    engine, db_file, model_dir = trained_combo_model
    model = load_model(model_dir)

    with Session(engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        frame = build_inference_frame(session, race_id)
        result = predict_race_with_combinations_gbdt(
            model, frame, session=session, n_samples=5_000, rng=np.random.default_rng(13)
        )

    # Build index: post_position pair → prob
    umaren_by_key = {p.post_positions: p.prob for p in result["馬連"]}
    wide_by_key = {p.post_positions: p.prob for p in result["ワイド"]}

    for key, wide_prob in wide_by_key.items():
        umaren_prob = umaren_by_key.get(key, 0.0)
        # ワイド includes all triples where both are in top-3,
        # which strictly includes the top-2 case → wide >= pair
        assert wide_prob >= umaren_prob - 1e-6, (
            f"Wide prob {wide_prob} < umaren prob {umaren_prob} for pair {key}"
        )


def test_predict_combinations_tansho_combo_is_post_position(trained_combo_model):
    """単勝 combo must be the post_position string, not horse_id.

    Regression test for the bug where combo=str(horse_id) was used instead of
    str(post_position), which broke matching against bet_records.combo and
    payouts.combo (both store post_position strings).
    """
    engine, db_file, model_dir = trained_combo_model
    model = load_model(model_dir)

    with Session(engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        frame = build_inference_frame(session, race_id)
        result = predict_race_with_combinations_gbdt(
            model, frame, session=session, n_samples=_N_SAMPLES, rng=np.random.default_rng(14)
        )

    race_post_positions = {str(pp) for pp in frame["post_position"].values}
    for p in result["単勝"]:
        assert p.combo in race_post_positions, (
            f"単勝 combo {p.combo!r} is not a post_position in the race "
            f"(expected one of {sorted(race_post_positions)})"
        )
        assert p.combo == str(p.post_positions[0]), (
            f"単勝 combo {p.combo!r} does not match post_positions[0]={p.post_positions[0]}"
        )


def test_predict_combinations_fukusho_combo_is_post_position(trained_combo_model):
    """複勝 combo must be the post_position string, not horse_id.

    Regression test for the bug where combo=str(horse_id) was used instead of
    str(post_position), which broke matching against bet_records.combo and
    payouts.combo (both store post_position strings).
    """
    engine, db_file, model_dir = trained_combo_model
    model = load_model(model_dir)

    with Session(engine) as session:
        race_id = session.scalars(select(Race.race_id).limit(1)).first()
        frame = build_inference_frame(session, race_id)
        result = predict_race_with_combinations_gbdt(
            model, frame, session=session, n_samples=_N_SAMPLES, rng=np.random.default_rng(15)
        )

    race_post_positions = {str(pp) for pp in frame["post_position"].values}
    for p in result["複勝"]:
        assert p.combo in race_post_positions, (
            f"複勝 combo {p.combo!r} is not a post_position in the race "
            f"(expected one of {sorted(race_post_positions)})"
        )
        assert p.combo == str(p.post_positions[0]), (
            f"複勝 combo {p.combo!r} does not match post_positions[0]={p.post_positions[0]}"
        )
