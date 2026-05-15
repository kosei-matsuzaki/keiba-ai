"""Tests for GET /api/scraper/recent_activity."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from db.models.scrape_log import ScrapeLog


def _add_log(session, fetched_at: datetime, race_id: str, status: str = "ok") -> None:
    session.add(
        ScrapeLog(
            url=f"https://db.netkeiba.com/race/{race_id}/",
            fetched_at=fetched_at.isoformat(),
            status=status,
        )
    )
    session.commit()


def test_recent_activity_empty_window(api_client: TestClient) -> None:
    """空 DB なら全カウントが 0、latest 系は None。"""
    resp = api_client.get("/api/scraper/recent_activity?minutes=10")
    assert resp.status_code == 200
    body = resp.json()
    assert body["window_minutes"] == 10
    assert body["total_fetched"] == 0
    assert body["ok_count"] == 0
    assert body["error_count"] == 0
    assert body["skipped_count"] == 0
    assert body["rate_per_min"] == 0.0
    assert body["latest_fetched_at"] is None
    assert body["latest_race_id"] is None


def test_recent_activity_counts_status_breakdown(
    app_with_temp_db,
    tmp_path,
) -> None:
    """ok / error / skipped が正しく分類される。"""
    from core.paths import db_path
    from db.session import make_engine, session_scope

    engine = make_engine(db_path())
    now = datetime.now(UTC)

    with session_scope(engine) as s:
        # 5 ok / 2 error / 1 skipped 全て直近
        for i in range(5):
            _add_log(s, now - timedelta(seconds=i), f"20240605090{i}", "ok")
        _add_log(s, now - timedelta(seconds=10), "202406050990", "error")
        _add_log(s, now - timedelta(seconds=11), "202406050991", "error")
        _add_log(s, now - timedelta(seconds=12), "202406050992", "skipped")

    with TestClient(app_with_temp_db) as client:
        resp = client.get("/api/scraper/recent_activity?minutes=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_fetched"] == 8
    assert body["ok_count"] == 5
    assert body["error_count"] == 2
    assert body["skipped_count"] == 1
    assert body["rate_per_min"] == 1.0  # 5 ok / 5 min


def test_recent_activity_excludes_old_entries(
    app_with_temp_db,
    tmp_path,
) -> None:
    """ウィンドウ外 (cutoff より古い) のログはカウントしない。"""
    from core.paths import db_path
    from db.session import make_engine, session_scope

    engine = make_engine(db_path())
    now = datetime.now(UTC)

    with session_scope(engine) as s:
        _add_log(s, now - timedelta(minutes=2), "202406050001")  # window 内
        _add_log(s, now - timedelta(minutes=20), "202406050002")  # window 外

    with TestClient(app_with_temp_db) as client:
        resp = client.get("/api/scraper/recent_activity?minutes=10")
    body = resp.json()
    assert body["total_fetched"] == 1
    assert body["latest_race_id"] == "202406050001"


def test_recent_activity_extracts_latest_race_id(
    app_with_temp_db,
    tmp_path,
) -> None:
    """latest_race_id は最新 fetched_at の URL の race_id を返す。"""
    from core.paths import db_path
    from db.session import make_engine, session_scope

    engine = make_engine(db_path())
    now = datetime.now(UTC)

    with session_scope(engine) as s:
        _add_log(s, now - timedelta(seconds=30), "202412280101")
        _add_log(s, now - timedelta(seconds=10), "202412280111")  # latest

    with TestClient(app_with_temp_db) as client:
        resp = client.get("/api/scraper/recent_activity?minutes=5")
    body = resp.json()
    assert body["latest_race_id"] == "202412280111"
    assert body["latest_fetched_at"] is not None


def test_recent_activity_minutes_bounds(api_client: TestClient) -> None:
    """`minutes` クエリパラメータは 1..1440 の範囲。"""
    # 範囲外は 422
    assert api_client.get("/api/scraper/recent_activity?minutes=0").status_code == 422
    assert api_client.get("/api/scraper/recent_activity?minutes=1500").status_code == 422
    # 範囲内
    assert api_client.get("/api/scraper/recent_activity?minutes=1").status_code == 200
    assert api_client.get("/api/scraper/recent_activity?minutes=1440").status_code == 200


def test_recent_activity_caps_rows_to_avoid_full_scan(
    app_with_temp_db,
    tmp_path,
) -> None:
    """ウィンドウ内に 2000 行を超える行があっても、SELECT 上限 (LIMIT 2000)
    が効いて応答時間が爆発しないことを確認する。Phase 2 ingest のピーク
    模擬。集計値は LIMIT 後の行に対する集計なので 2000 を超えない。
    """
    from core.paths import db_path
    from db.session import make_engine, session_scope

    engine = make_engine(db_path())
    now = datetime.now(UTC)

    # 2500 行を直近 5 分以内に分散して投入
    with session_scope(engine) as s:
        for i in range(2500):
            _add_log(
                s,
                now - timedelta(seconds=300 - (i / 2500.0) * 300),
                f"20240605{i:06d}"[-12:],
            )

    with TestClient(app_with_temp_db) as client:
        resp = client.get("/api/scraper/recent_activity?minutes=10")
    assert resp.status_code == 200
    body = resp.json()
    # LIMIT 2000 が効いて total_fetched は最大 2000
    assert body["total_fetched"] == 2000
    # 最新行の latest_fetched_at は拾える (order_by desc + limit なので)
    assert body["latest_fetched_at"] is not None
