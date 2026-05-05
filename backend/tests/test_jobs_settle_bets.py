"""Tests for jobs/settle_bets.py CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from keiba_ai.db.models.bet_record import BetRecord
from keiba_ai.db.models.payout import Payout
from keiba_ai.db.models.race import Race


def _insert_race(session: Session, race_id: str = "202406010101") -> Race:
    race = Race(
        race_id=race_id,
        date="2024-06-01",
        course="東京",
        surface="芝",
        distance=2400,
    )
    session.add(race)
    session.commit()
    return race


def _insert_payout(session: Session, race_id: str, bet_type: str, combo: str, amount: int) -> None:
    session.add(Payout(race_id=race_id, bet_type=bet_type, combo=combo, amount=amount, popularity=1))
    session.commit()


def _insert_bet(session: Session, race_id: str = "202406010101", settled: bool = False) -> BetRecord:
    bet = BetRecord(
        created_at="2024-06-01T10:00:00+00:00",
        race_id=race_id,
        bet_type="単勝",
        combo="5",
        stake=1000,
        source="manual",
        settled_at="2024-06-01T16:00:00+00:00" if settled else None,
        payout=2800 if settled else None,
        profit=1800 if settled else None,
    )
    session.add(bet)
    session.commit()
    return bet


class TestSettleBetsCLI:
    def test_run_settles_pending(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """run() が未確定 bet を確定し、確定件数を返す。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        monkeypatch.setenv("KEIBA_DATA_DIR", str(data_dir))

        import keiba_ai.core.paths as _paths
        monkeypatch.setattr(_paths, "data_dir", lambda: data_dir)

        from keiba_ai.core.paths import db_path
        from keiba_ai.db.base import Base
        from keiba_ai.db.session import make_engine, session_scope

        engine = make_engine(db_path())
        Base.metadata.create_all(engine)

        with session_scope(engine) as session:
            _insert_race(session)
            _insert_payout(session, "202406010101", "単勝", "5", 280)
            _insert_bet(session)

        from keiba_ai.jobs.settle_bets import run

        result = run(dry_run=False)
        assert result == 1

    def test_dry_run_returns_count_without_settling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--dry-run は確定せず対象件数のみ返す。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        monkeypatch.setenv("KEIBA_DATA_DIR", str(data_dir))

        import keiba_ai.core.paths as _paths
        monkeypatch.setattr(_paths, "data_dir", lambda: data_dir)

        from keiba_ai.core.paths import db_path
        from keiba_ai.db.base import Base
        from keiba_ai.db.session import make_engine, session_scope

        engine = make_engine(db_path())
        Base.metadata.create_all(engine)

        with session_scope(engine) as session:
            _insert_race(session)
            _insert_payout(session, "202406010101", "単勝", "5", 280)
            _insert_bet(session)  # 未確定 bet

        from keiba_ai.jobs.settle_bets import run

        result = run(dry_run=True)
        assert result == 1  # 件数を返す

        # DB が変更されていないことを確認
        with session_scope(engine) as session:
            from sqlalchemy import select
            bets = session.scalars(select(BetRecord)).all()
            assert all(b.settled_at is None for b in bets)

    def test_run_empty_pending(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """未確定 bet がない場合は 0 を返す。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        monkeypatch.setenv("KEIBA_DATA_DIR", str(data_dir))

        import keiba_ai.core.paths as _paths
        monkeypatch.setattr(_paths, "data_dir", lambda: data_dir)

        from keiba_ai.core.paths import db_path
        from keiba_ai.db.base import Base
        from keiba_ai.db.session import make_engine, session_scope

        engine = make_engine(db_path())
        Base.metadata.create_all(engine)

        with session_scope(engine) as session:
            _insert_race(session)
            _insert_bet(session, settled=True)  # 確定済み

        from keiba_ai.jobs.settle_bets import run

        result = run(dry_run=False)
        assert result == 0

    def test_dry_run_zero_pending(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        monkeypatch.setenv("KEIBA_DATA_DIR", str(data_dir))

        import keiba_ai.core.paths as _paths
        monkeypatch.setattr(_paths, "data_dir", lambda: data_dir)

        from keiba_ai.core.paths import db_path
        from keiba_ai.db.base import Base
        from keiba_ai.db.session import make_engine

        engine = make_engine(db_path())
        Base.metadata.create_all(engine)

        from keiba_ai.jobs.settle_bets import run

        result = run(dry_run=True)
        assert result == 0
