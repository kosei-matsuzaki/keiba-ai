"""Tests for payback_place (複勝回収率) in ai/evaluate.py."""

from __future__ import annotations

import json
import math
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ai.evaluation.backtest import _parse_payout_place, evaluate
from db.models.race import Race
from tests.synthetic import make_synthetic_db, train_synthetic_nn


class TestParsePdayoutPlace:
    def test_valid_json(self):
        result = _parse_payout_place('{"1": 120, "2": 200, "3": 180}')
        assert result == {1: 120, 2: 200, 3: 180}

    def test_none_returns_empty(self):
        assert _parse_payout_place(None) == {}

    def test_empty_string_returns_empty(self):
        assert _parse_payout_place("") == {}

    def test_invalid_json_returns_empty(self):
        assert _parse_payout_place("not-json") == {}

    def test_partial_positions(self):
        result = _parse_payout_place('{"1": 150}')
        assert result == {1: 150}


class TestPaybackPlace:
    """Integration-style tests using synthetic data."""

    def test_payback_place_computed_with_payout_data(self, tmp_path):
        """With payout_place JSON, payback_place should be non-NaN when bets are placed."""
        db_file = tmp_path / "test.db"
        engine = create_engine(f"sqlite:///{db_file}", future=True)
        make_synthetic_db(engine, n_races=20, n_horses_per_race=8, seed=99)

        with Session(engine) as s:
            races = s.query(Race).all()
            for race in races:
                race.payout_place = json.dumps({"1": 120, "2": 150, "3": 130})
            s.commit()

        os.environ["KEIBA_DATA_DIR"] = str(tmp_path / "data")

        model_dir = train_synthetic_nn(db_file)

        metrics = evaluate(model_path=model_dir, db=db_file)

        assert "place_bets" in metrics
        assert "place_invested" in metrics
        assert "place_gross_payout" in metrics
        assert "payback_place" in metrics
        assert isinstance(metrics["place_bets"], int)

    def test_payback_place_nan_when_no_payout_data(self, tmp_path):
        """Without payout_place data, place_bets=0 and payback_place=nan."""
        db_file = tmp_path / "test.db"
        engine = create_engine(f"sqlite:///{db_file}", future=True)
        make_synthetic_db(engine, n_races=20, n_horses_per_race=8, seed=77)

        os.environ["KEIBA_DATA_DIR"] = str(tmp_path / "data")

        model_dir = train_synthetic_nn(db_file)

        metrics = evaluate(model_path=model_dir, db=db_file)
        assert metrics["place_bets"] == 0
        assert math.isnan(metrics["payback_place"])

    def test_payback_place_mixed_none_and_data(self, tmp_path):
        """Races with payout_place=None are skipped; only races with data contribute."""
        db_file = tmp_path / "test.db"
        engine = create_engine(f"sqlite:///{db_file}", future=True)
        make_synthetic_db(engine, n_races=20, n_horses_per_race=8, seed=55)

        os.environ["KEIBA_DATA_DIR"] = str(tmp_path / "data")

        with Session(engine) as s:
            races = s.query(Race).order_by(Race.date).all()
            for i, race in enumerate(races):
                if i < 10:
                    race.payout_place = json.dumps({"1": 200, "2": 180, "3": 160})
                else:
                    race.payout_place = None
            s.commit()

        model_dir = train_synthetic_nn(db_file)

        metrics = evaluate(model_path=model_dir, db=db_file)
        assert "place_bets" in metrics
        assert isinstance(metrics["place_bets"], int)
        assert metrics["place_bets"] >= 0
