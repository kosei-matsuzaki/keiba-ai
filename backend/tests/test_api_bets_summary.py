"""Tests for /api/bets/summary, /api/bets/timeseries, /api/bets/breakdown, /api/bets/export.csv."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from db.models.bet_record import BetRecord
from db.models.race import Race

# ── Helpers ───────────────────────────────────────────────────────────────────

def _insert_race(session: Session, race_id: str = "202406010101", race_class: str | None = "G1") -> Race:
    race = Race(
        race_id=race_id,
        date=race_id[0:4] + "-" + race_id[4:6] + "-" + race_id[6:8],
        course="東京",
        surface="芝",
        distance=2400,
        race_class=race_class,
    )
    session.add(race)
    session.commit()
    return race


def _insert_bet(
    session: Session,
    *,
    race_id: str = "202406010101",
    bet_type: str = "単勝",
    stake: int = 1000,
    source: str = "manual",
    created_at: str = "2024-06-01T10:00:00+00:00",
    payout: int | None = None,
    profit: int | None = None,
    settled_at: str | None = None,
) -> BetRecord:
    bet = BetRecord(
        created_at=created_at,
        race_id=race_id,
        bet_type=bet_type,
        combo="5",
        stake=stake,
        source=source,
        settled_at=settled_at,
        payout=payout,
        profit=profit,
    )
    session.add(bet)
    session.commit()
    return bet


def _get_engine(app_with_temp_db: FastAPI):
    from core.paths import db_path
    from db.session import make_engine
    return make_engine(db_path())


# ── GET /api/bets/summary ─────────────────────────────────────────────────────

class TestBetSummary:
    def test_summary_empty(self, app_with_temp_db: FastAPI) -> None:
        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/bets/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_bets"] == 0
        assert data["settled_bets"] == 0
        assert data["total_invested"] == 0
        assert data["payback_rate"] == 0.0
        assert data["hit_rate"] == 0.0

    def test_summary_correct_totals(self, app_with_temp_db: FastAPI) -> None:
        """settled bet + unsettled bet の合計集計が正しい。"""
        engine = _get_engine(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session)
            # settled: 1000 invested, 2800 payout → 1800 profit
            _insert_bet(session, stake=1000, payout=2800, profit=1800, settled_at="2024-06-01T16:00:00+00:00")
            # settled: 500 invested, 0 payout → -500 profit (外れ)
            _insert_bet(session, stake=500, payout=0, profit=-500, settled_at="2024-06-01T16:01:00+00:00")
            # pending: 300 invested, no payout yet — not matched by settled_at filter
            _insert_bet(session, stake=300)

        with TestClient(app_with_temp_db) as client:
            # No date range → no settled_at filter applied → all 3 bets returned
            resp = client.get("/api/bets/summary")
        assert resp.status_code == 200
        data = resp.json()
        # pending bet has settled_at=NULL so it passes through when no date filter is set
        assert data["total_bets"] == 3
        assert data["settled_bets"] == 2
        assert data["pending_bets"] == 1
        assert data["total_invested"] == 1800
        assert data["total_payout"] == 2800
        assert data["total_profit"] == 1300  # 1800 + (-500)
        assert abs(data["payback_rate"] - 2800 / 1800) < 1e-6
        # 1 out of 2 settled bets had payout > 0
        assert abs(data["hit_rate"] - 0.5) < 1e-6

    def test_summary_filter_by_source(self, app_with_temp_db: FastAPI) -> None:
        engine = _get_engine(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session)
            _insert_bet(session, source="recommendation", stake=1000,
                        payout=3000, profit=2000, settled_at="2024-06-01T16:00:00+00:00")
            _insert_bet(session, source="manual", stake=500,
                        payout=0, profit=-500, settled_at="2024-06-01T16:01:00+00:00")

        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/bets/summary?source=recommendation")
        data = resp.json()
        assert data["total_bets"] == 1
        assert data["total_invested"] == 1000

    def test_summary_filter_by_date_range_uses_settled_at(self, app_with_temp_db: FastAPI) -> None:
        """期間フィルタは settled_at ベースで動作する。

        created_at が期間内でも settled_at が期間外なら除外される。
        settled_at が期間内なら created_at が期間外でも含まれる。
        """
        engine = _get_engine(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session)
            # created 5月、settled 6月 → settled_at が期間内なので含まれる
            _insert_bet(
                session, stake=500,
                created_at="2024-05-20T10:00:00+00:00",
                payout=0, profit=-500,
                settled_at="2024-06-15T16:00:00+00:00",
            )
            # created 6月、settled 7月 → settled_at が期間外なので除外される
            _insert_bet(
                session, stake=200,
                created_at="2024-06-20T10:00:00+00:00",
                payout=400, profit=200,
                settled_at="2024-07-01T16:00:00+00:00",
            )

        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/bets/summary?from=2024-06-01&to=2024-06-30")
        data = resp.json()
        # Only the first bet (settled in June) should be included
        assert data["total_bets"] == 1
        assert data["total_invested"] == 500
        assert data["range_from"] == "2024-06-01"
        assert data["range_to"] == "2024-06-30"

    def test_summary_filter_by_date_range(self, app_with_temp_db: FastAPI) -> None:
        """settled_at が期間内の bet のみ集計される（settled_at ベース確認）。"""
        engine = _get_engine(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session)
            # settled in May — outside the June range
            _insert_bet(
                session, stake=200,
                created_at="2024-05-01T10:00:00+00:00",
                payout=0, profit=-200,
                settled_at="2024-05-01T16:00:00+00:00",
            )
            # settled in June — inside the range
            _insert_bet(
                session, stake=500,
                created_at="2024-06-15T10:00:00+00:00",
                payout=1000, profit=500,
                settled_at="2024-06-15T16:00:00+00:00",
            )

        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/bets/summary?from=2024-06-01&to=2024-06-30")
        data = resp.json()
        assert data["total_bets"] == 1
        assert data["total_invested"] == 500
        assert data["range_from"] == "2024-06-01"
        assert data["range_to"] == "2024-06-30"


# ── GET /api/bets/timeseries ──────────────────────────────────────────────────

class TestBetTimeseries:
    def test_timeseries_empty(self, app_with_temp_db: FastAPI) -> None:
        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/bets/timeseries")
        assert resp.status_code == 200
        data = resp.json()
        assert data["bucket"] == "day"
        assert data["points"] == []

    def test_cumulative_profit_monotone(self, app_with_temp_db: FastAPI) -> None:
        """cumulative_profit は各点の profit の累積和と一致する。settled_at ベース。"""
        engine = _get_engine(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session)
            _insert_bet(session, created_at="2024-06-01T10:00:00+00:00", stake=1000,
                        payout=1500, profit=500, settled_at="2024-06-01T16:00:00+00:00")
            _insert_bet(session, created_at="2024-06-02T10:00:00+00:00", stake=1000,
                        payout=500, profit=-500, settled_at="2024-06-02T16:00:00+00:00")
            _insert_bet(session, created_at="2024-06-03T10:00:00+00:00", stake=500,
                        payout=1000, profit=500, settled_at="2024-06-03T16:00:00+00:00")

        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/bets/timeseries?bucket=day")
        data = resp.json()
        points = data["points"]
        assert len(points) == 3
        assert points[0]["cumulative_profit"] == 500
        assert points[1]["cumulative_profit"] == 0    # 500 + (-500)
        assert points[2]["cumulative_profit"] == 500  # 0 + 500

    def test_timeseries_bucket_month(self, app_with_temp_db: FastAPI) -> None:
        """bucket=month でグループ集約が正しい。settled_at ベース。"""
        engine = _get_engine(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session)
            _insert_bet(session, created_at="2024-06-01T10:00:00+00:00", stake=500,
                        payout=0, profit=-500, settled_at="2024-06-01T16:00:00+00:00")
            _insert_bet(session, created_at="2024-06-15T10:00:00+00:00", stake=500,
                        payout=1000, profit=500, settled_at="2024-06-15T16:00:00+00:00")
            _insert_bet(session, created_at="2024-07-01T10:00:00+00:00", stake=300,
                        payout=600, profit=300, settled_at="2024-07-01T16:00:00+00:00")

        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/bets/timeseries?bucket=month")
        data = resp.json()
        points = data["points"]
        assert len(points) == 2
        june = next(p for p in points if p["date"] == "2024-06")
        assert june["bets"] == 2
        assert june["invested"] == 1000
        assert june["profit"] == 0  # -500 + 500

    def test_timeseries_empty_buckets_filled_with_zero(self, app_with_temp_db: FastAPI) -> None:
        """期間内で bet が無い日が 0 で埋められ、cumulative_profit が持ち越される。"""
        engine = _get_engine(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session)
            # Jun 1 settled
            _insert_bet(session, stake=1000,
                        created_at="2024-06-01T10:00:00+00:00",
                        payout=1500, profit=500, settled_at="2024-06-01T16:00:00+00:00")
            # Jun 3 settled (Jun 2 has no bet)
            _insert_bet(session, stake=500,
                        created_at="2024-06-03T10:00:00+00:00",
                        payout=0, profit=-500, settled_at="2024-06-03T16:00:00+00:00")

        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/bets/timeseries?bucket=day&from=2024-06-01&to=2024-06-03")
        data = resp.json()
        points = data["points"]

        # All 3 days must be present (including the empty Jun 2)
        dates = [p["date"] for p in points]
        assert "2024-06-01" in dates
        assert "2024-06-02" in dates
        assert "2024-06-03" in dates
        assert len(points) == 3

        jun2 = next(p for p in points if p["date"] == "2024-06-02")
        assert jun2["bets"] == 0
        assert jun2["invested"] == 0
        assert jun2["profit"] == 0
        # cumulative_profit carries over from Jun 1 (500)
        assert jun2["cumulative_profit"] == 500

    def test_timeseries_cumulative_profit_nondecreasing_with_gaps(self, app_with_temp_db: FastAPI) -> None:
        """空 bucket では cumulative_profit が単調非減少（前 bucket 値を維持）。"""
        engine = _get_engine(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session)
            _insert_bet(session, stake=1000,
                        created_at="2024-06-01T10:00:00+00:00",
                        payout=2000, profit=1000, settled_at="2024-06-01T16:00:00+00:00")
            _insert_bet(session, stake=500,
                        created_at="2024-06-05T10:00:00+00:00",
                        payout=500, profit=0, settled_at="2024-06-05T16:00:00+00:00")

        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/bets/timeseries?bucket=day&from=2024-06-01&to=2024-06-05")
        data = resp.json()
        points = data["points"]

        # 5 days: Jun 1–5
        assert len(points) == 5

        cumulative_values = [p["cumulative_profit"] for p in points]
        # Jun 1: 1000, Jun 2–4: 1000 (carried), Jun 5: 1000 + 0 = 1000
        assert cumulative_values[0] == 1000
        assert cumulative_values[1] == 1000  # empty bucket, carry over
        assert cumulative_values[2] == 1000
        assert cumulative_values[3] == 1000
        assert cumulative_values[4] == 1000

    def test_timeseries_excludes_pending_bets(self, app_with_temp_db: FastAPI) -> None:
        """settled_at IS NULL の pending bet は timeseries に含まれない。"""
        engine = _get_engine(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session)
            _insert_bet(session, stake=1000,
                        created_at="2024-06-01T10:00:00+00:00",
                        payout=1500, profit=500, settled_at="2024-06-01T16:00:00+00:00")
            # pending bet — should not appear in timeseries
            _insert_bet(session, stake=500, created_at="2024-06-01T11:00:00+00:00")

        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/bets/timeseries?bucket=day")
        data = resp.json()
        points = data["points"]
        assert len(points) == 1
        assert points[0]["bets"] == 1
        assert points[0]["invested"] == 1000


# ── GET /api/bets/breakdown ───────────────────────────────────────────────────

class TestBetBreakdown:
    def test_breakdown_by_bet_type(self, app_with_temp_db: FastAPI) -> None:
        engine = _get_engine(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session)
            _insert_bet(session, bet_type="単勝", stake=1000,
                        payout=2000, profit=1000, settled_at="2024-06-01T16:00:00+00:00")
            _insert_bet(session, bet_type="単勝", stake=500,
                        payout=0, profit=-500, settled_at="2024-06-01T16:01:00+00:00")
            _insert_bet(session, bet_type="複勝", stake=300,
                        payout=450, profit=150, settled_at="2024-06-01T16:02:00+00:00")

        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/bets/breakdown?group_by=bet_type")
        data = resp.json()
        assert data["group_by"] == "bet_type"
        rows = {r["group_key"]: r for r in data["rows"]}
        assert len(rows) == 2
        tansho = rows["単勝"]
        assert tansho["bets"] == 2
        assert tansho["invested"] == 1500
        assert abs(tansho["payback_rate"] - 2000 / 1500) < 1e-6
        assert abs(tansho["hit_rate"] - 0.5) < 1e-6

    def test_breakdown_by_source(self, app_with_temp_db: FastAPI) -> None:
        engine = _get_engine(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session)
            _insert_bet(session, source="recommendation", stake=1000,
                        payout=1500, profit=500, settled_at="2024-06-01T16:00:00+00:00")
            _insert_bet(session, source="manual", stake=500)

        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/bets/breakdown?group_by=source")
        data = resp.json()
        rows = {r["group_key"]: r for r in data["rows"]}
        assert "recommendation" in rows
        # pending bet (manual) has no settled_at → passes through with no date filter
        assert "manual" in rows
        assert rows["recommendation"]["invested"] == 1000

    def test_breakdown_filter_by_source(self, app_with_temp_db: FastAPI) -> None:
        engine = _get_engine(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session)
            _insert_bet(session, source="recommendation", stake=1000)
            _insert_bet(session, source="manual", bet_type="複勝", stake=500)

        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/bets/breakdown?group_by=bet_type&source=recommendation")
        data = resp.json()
        # Only recommendation bets included
        assert all(r["invested"] <= 1000 for r in data["rows"])


# ── GET /api/bets/export.csv ──────────────────────────────────────────────────

class TestBetExportCsv:
    def test_export_empty(self, app_with_temp_db: FastAPI) -> None:
        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/bets/export.csv")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        # Only header row
        body = resp.content.decode("utf-8-sig")
        lines = [line for line in body.splitlines() if line.strip()]
        assert len(lines) == 1  # header only

    def test_export_has_bom(self, app_with_temp_db: FastAPI) -> None:
        """BOM 付き UTF-8 (0xEF 0xBB 0xBF) で返す。"""
        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/bets/export.csv")
        assert resp.content[:3] == b"\xef\xbb\xbf"

    def test_export_content_disposition(self, app_with_temp_db: FastAPI) -> None:
        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/bets/export.csv")
        assert "attachment" in resp.headers.get("content-disposition", "")

    def test_export_correct_columns(self, app_with_temp_db: FastAPI) -> None:
        engine = _get_engine(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session)
            _insert_bet(session, stake=1000, payout=2500, profit=1500,
                        settled_at="2024-06-01T16:00:00+00:00")

        with TestClient(app_with_temp_db) as client:
            # default range は直近 1 年なので 2024-06-01 を含む期間を明示
            resp = client.get("/api/bets/export.csv?from=2024-01-01&to=2024-12-31")
        body = resp.content.decode("utf-8-sig")
        lines = body.splitlines()
        header = lines[0].split(",")
        assert "id" in header
        assert "bet_type" in header
        assert "stake" in header
        assert "profit" in header
        assert "notes" in header
        # One data row
        assert len(lines) == 2

    def test_export_filter_by_source(self, app_with_temp_db: FastAPI) -> None:
        engine = _get_engine(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session)
            _insert_bet(session, source="recommendation", stake=1000,
                        settled_at="2024-06-01T16:00:00+00:00")
            _insert_bet(session, source="manual", stake=500,
                        settled_at="2024-06-01T16:01:00+00:00")

        with TestClient(app_with_temp_db) as client:
            resp = client.get(
                "/api/bets/export.csv?source=recommendation&from=2024-01-01&to=2024-12-31"
            )
        body = resp.content.decode("utf-8-sig")
        lines = [line for line in body.splitlines() if line.strip()]
        assert len(lines) == 2  # header + 1 row

    def test_export_notes_with_comma(self, app_with_temp_db: FastAPI) -> None:
        """notes にカンマが含まれても CSV エスケープされる。"""
        engine = _get_engine(app_with_temp_db)
        from db.session import session_scope
        with session_scope(engine) as session:
            _insert_race(session)
            bet = BetRecord(
                created_at="2024-06-01T10:00:00+00:00",
                race_id="202406010101",
                bet_type="単勝",
                combo="5",
                stake=1000,
                source="manual",
                notes="テスト,メモ",
                settled_at="2024-06-01T16:00:00+00:00",
            )
            session.add(bet)
            session.commit()

        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/bets/export.csv?from=2024-01-01&to=2024-12-31")
        body = resp.content.decode("utf-8-sig")
        # csv quoting ensures comma inside notes doesn't break column count
        import csv
        import io
        reader = csv.DictReader(io.StringIO(body))
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["notes"] == "テスト,メモ"

    def test_export_period_over_one_year_returns_400(self, app_with_temp_db: FastAPI) -> None:
        """期間 > 366 日の場合は 400 が返る。"""
        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/bets/export.csv?from=2022-01-01&to=2023-12-31")
        assert resp.status_code == 400
        assert "1 year" in resp.json()["detail"]

    def test_export_period_exactly_366_days_is_ok(self, app_with_temp_db: FastAPI) -> None:
        """期間がちょうど 366 日（上限）なら 200 が返る。"""
        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/bets/export.csv?from=2024-01-01&to=2025-01-01")
        assert resp.status_code == 200

    def test_export_limit_param_accepted(self, app_with_temp_db: FastAPI) -> None:
        """limit パラメータが受け付けられる。"""
        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/bets/export.csv?limit=100")
        assert resp.status_code == 200

    def test_export_limit_over_max_returns_422(self, app_with_temp_db: FastAPI) -> None:
        """limit > 50000 は FastAPI バリデーションで 422 が返る。"""
        with TestClient(app_with_temp_db) as client:
            resp = client.get("/api/bets/export.csv?limit=99999")
        assert resp.status_code == 422
