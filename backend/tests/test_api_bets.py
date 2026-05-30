"""Tests for /api/bets endpoints."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from db.models.bet_record import BetRecord
from db.models.payout import Payout
from db.models.race import Race


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


def _insert_payout(
    session: Session,
    race_id: str,
    bet_type: str,
    combo: str,
    amount: int,
) -> None:
    session.add(Payout(race_id=race_id, bet_type=bet_type, combo=combo, amount=amount, popularity=1))
    session.commit()


def _insert_settled_bet(session: Session, race_id: str = "202406010101") -> BetRecord:
    """Helper: insert an already-settled bet directly."""
    bet = BetRecord(
        created_at="2024-06-01T10:00:00+00:00",
        race_id=race_id,
        bet_type="単勝",
        combo="5",
        stake=1000,
        source="manual",
        settled_at="2024-06-01T16:00:00+00:00",
        payout=2800,
        profit=1800,
    )
    session.add(bet)
    session.commit()
    return bet


def _get_engine_and_session(app_with_temp_db):
    from core.paths import db_path
    from db.session import make_engine
    engine = make_engine(db_path())
    return engine


# ── POST /api/bets ────────────────────────────────────────────────────────────

class TestCreateBet:
    def test_create_bet_201(self, app_with_temp_db: FastAPI) -> None:
        engine = _get_engine_and_session(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session)

        with TestClient(app_with_temp_db) as client:
            resp = client.post("/api/bets", json={
                "race_id": "202406010101",
                "bet_type": "単勝",
                "combo": "5",
                "stake": 1000,
                "source": "manual",
            })
        assert resp.status_code == 201
        data = resp.json()
        assert data["race_id"] == "202406010101"
        assert data["bet_type"] == "単勝"
        assert data["stake"] == 1000
        assert data["settled_at"] is None  # payouts なし → 未確定

    def test_create_bet_immediately_settled(self, app_with_temp_db: FastAPI) -> None:
        """payouts が既に存在する場合は POST と同時に確定する。"""
        engine = _get_engine_and_session(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session)
            _insert_payout(session, "202406010101", "単勝", "5", 280)

        with TestClient(app_with_temp_db) as client:
            resp = client.post("/api/bets", json={
                "race_id": "202406010101",
                "bet_type": "単勝",
                "combo": "5",
                "stake": 1000,
                "source": "recommendation",
                "recommendation_id": 99,
            })
        assert resp.status_code == 201
        data = resp.json()
        assert data["settled_at"] is not None
        assert data["payout"] == 2800
        assert data["profit"] == 1800

    def test_create_bet_race_not_found_404(self, app_with_temp_db: FastAPI) -> None:
        with TestClient(app_with_temp_db) as client:
            resp = client.post("/api/bets", json={
                "race_id": "NONEXISTENT",
                "bet_type": "単勝",
                "combo": "5",
                "stake": 1000,
                "source": "manual",
            })
        assert resp.status_code == 404

    def test_create_bet_invalid_bet_type_422(self, app_with_temp_db: FastAPI) -> None:
        """bet_type が Literal 外なら 422 を返す。"""
        engine = _get_engine_and_session(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session)

        with TestClient(app_with_temp_db) as client:
            resp = client.post("/api/bets", json={
                "race_id": "202406010101",
                "bet_type": "不明な馬券",
                "combo": "5",
                "stake": 1000,
                "source": "manual",
            })
        assert resp.status_code == 422

    def test_create_bet_invalid_source_422(self, app_with_temp_db: FastAPI) -> None:
        engine = _get_engine_and_session(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session)

        with TestClient(app_with_temp_db) as client:
            resp = client.post("/api/bets", json={
                "race_id": "202406010101",
                "bet_type": "単勝",
                "combo": "5",
                "stake": 1000,
                "source": "unknown_source",
            })
        assert resp.status_code == 422


# ── GET /api/bets ─────────────────────────────────────────────────────────────

class TestListBets:
    def test_list_empty(self, app_with_temp_db: FastAPI) -> None:
        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/bets")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["items"] == []

    def test_list_all(self, app_with_temp_db: FastAPI) -> None:
        engine = _get_engine_and_session(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session)
            for combo in ("3", "7"):
                session.add(BetRecord(
                    created_at="2024-06-01T10:00:00+00:00",
                    race_id="202406010101",
                    bet_type="複勝",
                    combo=combo,
                    stake=500,
                    source="manual",
                ))
            session.commit()

        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/bets")
        assert resp.status_code == 200
        assert resp.json()["total"] == 2

    def test_filter_by_race_id(self, app_with_temp_db: FastAPI) -> None:
        engine = _get_engine_and_session(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session, "202406010101")
            _insert_race(session, "202406010201")
            session.add(BetRecord(
                created_at="2024-06-01T10:00:00+00:00",
                race_id="202406010101",
                bet_type="単勝", combo="5", stake=100, source="manual",
            ))
            session.add(BetRecord(
                created_at="2024-06-01T10:00:00+00:00",
                race_id="202406010201",
                bet_type="単勝", combo="3", stake=100, source="manual",
            ))
            session.commit()

        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/bets?race_id=202406010101")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["items"][0]["race_id"] == "202406010101"

    def test_filter_settled_true(self, app_with_temp_db: FastAPI) -> None:
        engine = _get_engine_and_session(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session)
            _insert_settled_bet(session)
            session.add(BetRecord(
                created_at="2024-06-01T11:00:00+00:00",
                race_id="202406010101",
                bet_type="複勝", combo="3", stake=100, source="manual",
            ))
            session.commit()

        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/bets?settled=true")
        assert resp.json()["total"] == 1
        assert resp.json()["items"][0]["settled_at"] is not None

    def test_filter_settled_false(self, app_with_temp_db: FastAPI) -> None:
        engine = _get_engine_and_session(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session)
            _insert_settled_bet(session)
            session.add(BetRecord(
                created_at="2024-06-01T11:00:00+00:00",
                race_id="202406010101",
                bet_type="複勝", combo="3", stake=100, source="manual",
            ))
            session.commit()

        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/bets?settled=false")
        assert resp.json()["total"] == 1
        assert resp.json()["items"][0]["settled_at"] is None

    def test_filter_by_source(self, app_with_temp_db: FastAPI) -> None:
        engine = _get_engine_and_session(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session)
            session.add(BetRecord(
                created_at="2024-06-01T10:00:00+00:00",
                race_id="202406010101",
                bet_type="単勝", combo="5", stake=100, source="recommendation",
            ))
            session.add(BetRecord(
                created_at="2024-06-01T10:01:00+00:00",
                race_id="202406010101",
                bet_type="単勝", combo="3", stake=100, source="manual",
            ))
            session.commit()

        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/bets?source=recommendation")
        assert resp.json()["total"] == 1
        assert resp.json()["items"][0]["source"] == "recommendation"


# ── GET /api/bets/{id} ────────────────────────────────────────────────────────

class TestGetBet:
    def test_get_existing(self, app_with_temp_db: FastAPI) -> None:
        engine = _get_engine_and_session(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session)
            bet = BetRecord(
                created_at="2024-06-01T10:00:00+00:00",
                race_id="202406010101",
                bet_type="単勝", combo="5", stake=1000, source="manual",
            )
            session.add(bet)
            session.commit()
            bet_id = bet.id

        with TestClient(app_with_temp_db) as client:
            resp = client.get(f"/api/bets/{bet_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == bet_id

    def test_get_not_found(self, app_with_temp_db: FastAPI) -> None:
        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/bets/99999")
        assert resp.status_code == 404


# ── PUT /api/bets/{id} ────────────────────────────────────────────────────────

class TestUpdateBet:
    def test_update_notes(self, app_with_temp_db: FastAPI) -> None:
        engine = _get_engine_and_session(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session)
            bet = BetRecord(
                created_at="2024-06-01T10:00:00+00:00",
                race_id="202406010101",
                bet_type="単勝", combo="5", stake=1000, source="manual",
            )
            session.add(bet)
            session.commit()
            bet_id = bet.id

        with TestClient(app_with_temp_db) as client:
            resp = client.put(f"/api/bets/{bet_id}", json={"notes": "テストメモ"})
        assert resp.status_code == 200
        assert resp.json()["notes"] == "テストメモ"

    def test_update_settled_bet_409(self, app_with_temp_db: FastAPI) -> None:
        """settled な bet の更新は 409。"""
        engine = _get_engine_and_session(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session)
            bet = _insert_settled_bet(session)
            bet_id = bet.id

        with TestClient(app_with_temp_db) as client:
            resp = client.put(f"/api/bets/{bet_id}", json={"notes": "変更試みる"})
        assert resp.status_code == 409

    def test_update_not_found(self, app_with_temp_db: FastAPI) -> None:
        with TestClient(app_with_temp_db) as client:
            resp = client.put("/api/bets/99999", json={"notes": "存在しない"})
        assert resp.status_code == 404


# ── DELETE /api/bets/{id} ─────────────────────────────────────────────────────

class TestDeleteBet:
    def test_delete_204(self, app_with_temp_db: FastAPI) -> None:
        engine = _get_engine_and_session(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session)
            bet = BetRecord(
                created_at="2024-06-01T10:00:00+00:00",
                race_id="202406010101",
                bet_type="単勝", combo="5", stake=1000, source="manual",
            )
            session.add(bet)
            session.commit()
            bet_id = bet.id

        with TestClient(app_with_temp_db) as client:
            resp = client.delete(f"/api/bets/{bet_id}")
        assert resp.status_code == 204

        # 削除後に GET すると 404
        with TestClient(app_with_temp_db) as client:
            resp = client.get(f"/api/bets/{bet_id}")
        assert resp.status_code == 404

    def test_delete_settled_409(self, app_with_temp_db: FastAPI) -> None:
        """settled な bet の削除は 409。"""
        engine = _get_engine_and_session(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session)
            bet = _insert_settled_bet(session)
            bet_id = bet.id

        with TestClient(app_with_temp_db) as client:
            resp = client.delete(f"/api/bets/{bet_id}")
        assert resp.status_code == 409

    def test_delete_not_found(self, app_with_temp_db: FastAPI) -> None:
        with TestClient(app_with_temp_db) as client:
            resp = client.delete("/api/bets/99999")
        assert resp.status_code == 404
