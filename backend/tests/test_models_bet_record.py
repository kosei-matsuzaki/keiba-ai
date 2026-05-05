"""ORM CRUD tests for BetRecord model."""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from keiba_ai.db.models.bet_record import BetRecord
from keiba_ai.db.models.race import Race


def _add_race(session: Session, race_id: str = "202406010101") -> Race:
    race = Race(
        race_id=race_id,
        date="2024-06-01",
        course="東京",
        surface="芝",
        distance=2400,
    )
    session.add(race)
    session.flush()
    return race


def _make_bet(race_id: str = "202406010101", **kwargs) -> BetRecord:
    defaults = {
        "created_at": "2024-06-01T10:00:00+00:00",
        "race_id": race_id,
        "bet_type": "単勝",
        "combo": "5",
        "stake": 1000,
        "source": "manual",
    }
    defaults.update(kwargs)
    return BetRecord(**defaults)


class TestBetRecordModel:
    def test_insert_select(self, db_session):
        _add_race(db_session)
        bet = _make_bet()
        db_session.add(bet)
        db_session.commit()

        result = db_session.execute(
            select(BetRecord).where(BetRecord.race_id == "202406010101")
        ).scalar_one()
        assert result.bet_type == "単勝"
        assert result.combo == "5"
        assert result.stake == 1000
        assert result.source == "manual"
        assert result.settled_at is None
        assert result.payout is None
        assert result.profit is None

    def test_optional_fields_nullable(self, db_session):
        _add_race(db_session)
        bet = _make_bet(recommendation_id=None, notes=None)
        db_session.add(bet)
        db_session.commit()

        result = db_session.execute(select(BetRecord)).scalar_one()
        assert result.recommendation_id is None
        assert result.notes is None

    def test_settled_fields(self, db_session):
        _add_race(db_session)
        bet = _make_bet(
            settled_at="2024-06-01T16:00:00+00:00",
            payout=2800,
            profit=1800,
        )
        db_session.add(bet)
        db_session.commit()

        result = db_session.execute(select(BetRecord)).scalar_one()
        assert result.settled_at == "2024-06-01T16:00:00+00:00"
        assert result.payout == 2800
        assert result.profit == 1800

    def test_indexes_in_table_args(self):
        index_names = {idx.name for idx in BetRecord.__table__.indexes}
        assert "ix_bet_records_race_id" in index_names
        assert "ix_bet_records_created_at" in index_names
        assert "ix_bet_records_settled_at" in index_names

    def test_fk_restrict_delete_race(self, db_session):
        """bet_records がある races の削除は RESTRICT で失敗する。"""
        _add_race(db_session)
        bet = _make_bet()
        db_session.add(bet)
        db_session.commit()

        race = db_session.execute(
            select(Race).where(Race.race_id == "202406010101")
        ).scalar_one()
        db_session.delete(race)
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_multiple_bets_same_race(self, db_session):
        _add_race(db_session)
        for combo in ("3", "7", "12"):
            db_session.add(_make_bet(bet_type="複勝", combo=combo, stake=500))
        db_session.commit()

        results = db_session.scalars(
            select(BetRecord).where(BetRecord.race_id == "202406010101")
        ).all()
        assert len(results) == 3

    def test_source_recommendation_with_recommendation_id(self, db_session):
        _add_race(db_session)
        bet = _make_bet(source="recommendation", recommendation_id=42)
        db_session.add(bet)
        db_session.commit()

        result = db_session.execute(select(BetRecord)).scalar_one()
        assert result.source == "recommendation"
        assert result.recommendation_id == 42
