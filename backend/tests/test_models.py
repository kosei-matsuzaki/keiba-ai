"""ORM model tests — basic CRUD and FK/index constraints."""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from keiba_ai.db.models.entry import Entry
from keiba_ai.db.models.horse import Horse
from keiba_ai.db.models.jockey import Jockey
from keiba_ai.db.models.model_run import ModelRun
from keiba_ai.db.models.payout import Payout
from keiba_ai.db.models.race import Race
from keiba_ai.db.models.scrape_log import ScrapeLog
from keiba_ai.db.models.trainer import Trainer

# ── helpers ──────────────────────────────────────────────────────────────────

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


def _add_horse(session: Session, horse_id: str = "2019105293") -> Horse:
    horse = Horse(horse_id=horse_id, name="テストホース")
    session.add(horse)
    session.flush()
    return horse


def _add_entry(
    session: Session,
    race_id: str = "202406010101",
    horse_id: str = "2019105293",
) -> Entry:
    entry = Entry(race_id=race_id, horse_id=horse_id, post_position=1)
    session.add(entry)
    session.flush()
    return entry


# ── Race ─────────────────────────────────────────────────────────────────────

class TestRaceModel:
    def test_insert_select(self, db_session):
        _add_race(db_session)
        db_session.commit()
        result = db_session.execute(select(Race).where(Race.race_id == "202406010101")).scalar_one()
        assert result.course == "東京"
        assert result.surface == "芝"
        assert result.distance == 2400

    def test_optional_fields_nullable(self, db_session):
        race = Race(race_id="202406010102", date="2024-06-01", course="中山", surface="ダ", distance=1800)
        db_session.add(race)
        db_session.commit()
        r = db_session.execute(select(Race).where(Race.race_id == "202406010102")).scalar_one()
        assert r.weather is None
        assert r.race_class is None


# ── Horse ─────────────────────────────────────────────────────────────────────

class TestHorseModel:
    def test_insert_select(self, db_session):
        _add_horse(db_session)
        db_session.commit()
        h = db_session.execute(select(Horse).where(Horse.horse_id == "2019105293")).scalar_one()
        assert h.name == "テストホース"

    def test_optional_fields(self, db_session):
        horse = Horse(horse_id="H001", name="テスト2", sex="牡", sire="ディープインパクト")
        db_session.add(horse)
        db_session.commit()
        h = db_session.execute(select(Horse).where(Horse.horse_id == "H001")).scalar_one()
        assert h.sire == "ディープインパクト"
        assert h.dam is None


# ── Jockey / Trainer ─────────────────────────────────────────────────────────

class TestJockeyTrainerModel:
    def test_jockey_insert(self, db_session):
        j = Jockey(jockey_id="01011", name="横山武史")
        db_session.add(j)
        db_session.commit()
        result = db_session.execute(select(Jockey).where(Jockey.jockey_id == "01011")).scalar_one()
        assert result.name == "横山武史"

    def test_trainer_insert(self, db_session):
        t = Trainer(trainer_id="01096", name="田中博康")
        db_session.add(t)
        db_session.commit()
        result = db_session.execute(select(Trainer).where(Trainer.trainer_id == "01096")).scalar_one()
        assert result.name == "田中博康"


# ── Entry ─────────────────────────────────────────────────────────────────────

class TestEntryModel:
    def test_insert_select(self, db_session):
        _add_race(db_session)
        _add_horse(db_session)
        _add_entry(db_session)
        db_session.commit()
        e = db_session.execute(select(Entry).where(Entry.race_id == "202406010101")).scalar_one()
        assert e.horse_id == "2019105293"
        assert e.post_position == 1

    def test_composite_indexes_in_table_args(self):
        """Verify composite indexes are declared in __table_args__."""
        index_names = {idx.name for idx in Entry.__table__.indexes}
        assert "ix_entries_race_id_horse_id" in index_names
        assert "ix_entries_horse_id_finish_position" in index_names

    def test_unique_constraint_race_horse(self, db_session):
        """Inserting duplicate (race_id, horse_id) must raise IntegrityError."""
        _add_race(db_session)
        _add_horse(db_session)
        _add_entry(db_session)
        db_session.add(Entry(race_id="202406010101", horse_id="2019105293", post_position=2))
        with pytest.raises(IntegrityError):
            db_session.flush()

    def test_fk_cascade_delete_race(self, db_session):
        """Deleting a race must cascade-delete its entries."""
        _add_race(db_session)
        _add_horse(db_session)
        _add_entry(db_session)
        db_session.commit()

        race = db_session.execute(select(Race).where(Race.race_id == "202406010101")).scalar_one()
        db_session.delete(race)
        db_session.commit()

        entries = db_session.execute(select(Entry).where(Entry.race_id == "202406010101")).scalars().all()
        assert len(entries) == 0

    def test_fk_restrict_delete_horse(self, db_session):
        """Deleting a horse that has entries must raise IntegrityError (RESTRICT)."""
        _add_race(db_session)
        _add_horse(db_session)
        _add_entry(db_session)
        db_session.commit()

        horse = db_session.execute(select(Horse).where(Horse.horse_id == "2019105293")).scalar_one()
        db_session.delete(horse)
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_fk_set_null_jockey(self, db_session):
        """Deleting a jockey should SET NULL on entries.jockey_id."""
        _add_race(db_session)
        _add_horse(db_session)
        j = Jockey(jockey_id="J001", name="テスト騎手")
        db_session.add(j)
        db_session.flush()
        entry = Entry(race_id="202406010101", horse_id="2019105293", jockey_id="J001")
        db_session.add(entry)
        db_session.commit()

        jockey = db_session.execute(select(Jockey).where(Jockey.jockey_id == "J001")).scalar_one()
        db_session.delete(jockey)
        db_session.commit()

        e = db_session.execute(select(Entry).where(Entry.race_id == "202406010101")).scalar_one()
        assert e.jockey_id is None


# ── Payout ────────────────────────────────────────────────────────────────────

class TestPayoutModel:
    def test_insert_select(self, db_session):
        _add_race(db_session)
        p = Payout(race_id="202406010101", bet_type="単勝", combo="5", amount=280, popularity=1)
        db_session.add(p)
        db_session.commit()
        result = db_session.execute(select(Payout).where(Payout.race_id == "202406010101")).scalar_one()
        assert result.bet_type == "単勝"
        assert result.amount == 280

    def test_payout_index_in_table_args(self):
        index_names = {idx.name for idx in Payout.__table__.indexes}
        assert "ix_payouts_race_id_bet_type" in index_names

    def test_fk_cascade_delete_race(self, db_session):
        _add_race(db_session)
        p = Payout(race_id="202406010101", bet_type="単勝", combo="5", amount=280)
        db_session.add(p)
        db_session.commit()

        race = db_session.execute(select(Race).where(Race.race_id == "202406010101")).scalar_one()
        db_session.delete(race)
        db_session.commit()

        payouts = db_session.execute(select(Payout).where(Payout.race_id == "202406010101")).scalars().all()
        assert len(payouts) == 0


# ── ScrapeLog ─────────────────────────────────────────────────────────────────

class TestScrapeLogModel:
    def test_insert_select(self, db_session):
        log = ScrapeLog(
            url="https://db.netkeiba.com/race/202406010101/",
            fetched_at="2024-06-01T12:00:00+00:00",
            status="ok",
            content_hash="abc123",
        )
        db_session.add(log)
        db_session.commit()
        result = db_session.execute(select(ScrapeLog).where(ScrapeLog.status == "ok")).scalar_one()
        assert result.url == "https://db.netkeiba.com/race/202406010101/"

    def test_index_in_table_args(self):
        index_names = {idx.name for idx in ScrapeLog.__table__.indexes}
        assert "ix_scrape_log_url_status" in index_names


# ── ModelRun ──────────────────────────────────────────────────────────────────

class TestModelRunModel:
    def test_insert_select(self, db_session):
        mr = ModelRun(
            created_at="2024-06-01T00:00:00+00:00",
            model_path="data/models/20240601/model.lgb",
            is_active=0,
        )
        db_session.add(mr)
        db_session.commit()
        result = db_session.execute(select(ModelRun)).scalar_one()
        assert result.model_path == "data/models/20240601/model.lgb"
        assert result.is_active == 0

    def test_is_active_default(self, db_session):
        """is_active defaults to 0 when not explicitly set via server_default."""
        mr = ModelRun(
            created_at="2024-06-01T00:00:00+00:00",
            model_path="data/models/x/model.lgb",
        )
        db_session.add(mr)
        db_session.commit()
        result = db_session.execute(select(ModelRun)).scalar_one()
        # server_default applies at DB level; ORM default also set
        assert result.is_active == 0
