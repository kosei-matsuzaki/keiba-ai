"""Tests for jobs/refill_payouts.py — retro-fill フローの冪等性検証。"""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest
from sqlalchemy import select

from keiba_ai.db.models.payout import Payout
from keiba_ai.db.models.race import Race
from keiba_ai.jobs.refill_payouts import _collect_cache_files, run_refill

FIXTURES = Path(__file__).parent / "fixtures"
ALL_PAYOUT_HTML = (FIXTURES / "race_result_all_payout_types.html").read_text(encoding="utf-8")
BASIC_HTML = (FIXTURES / "race_result_202406010101.html").read_text(encoding="utf-8")

_RACE_ID_ALL = "202406010101"  # 基本フィクスチャと揃えた 12 桁


def _write_cache(raw_root: Path, race_id: str, html: str) -> Path:
    """data/raw/<yyyy>/<mm>/<race_id>.html を書き込む。"""
    yyyy = race_id[:4]
    mm = race_id[4:6]
    cache_dir = raw_root / yyyy / mm
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{race_id}.html"
    path.write_text(html, encoding="utf-8")
    return path


def _insert_race(session, race_id: str) -> None:
    """races テーブルに最小限の行を挿入（FK 制約を満たすため）。"""
    from sqlalchemy.dialects.sqlite import insert as sqlite_insert
    stmt = sqlite_insert(Race).values(
        race_id=race_id,
        date=f"{race_id[:4]}-{race_id[4:6]}-{race_id[6:8]}",
        course="東京",
        surface="芝",
        distance=2000,
    ).on_conflict_do_nothing(index_elements=["race_id"])
    session.execute(stmt)
    session.commit()


# ── _collect_cache_files() ──────────────────────────────────────────────────

def test_collect_cache_files_finds_html(tmp_path):
    """キャッシュディレクトリの HTML を正しく列挙する。"""
    raw = tmp_path / "raw"
    _write_cache(raw, _RACE_ID_ALL, ALL_PAYOUT_HTML)

    files = _collect_cache_files(raw, start=None, end=None)
    assert len(files) == 1
    assert files[0][0] == _RACE_ID_ALL


def test_collect_cache_files_date_filter_start(tmp_path):
    """--start より前の race_id はスキップされる。"""
    raw = tmp_path / "raw"
    _write_cache(raw, "202401010101", BASIC_HTML)
    _write_cache(raw, "202406010101", ALL_PAYOUT_HTML)

    files = _collect_cache_files(raw, start=datetime.date(2024, 6, 1), end=None)
    race_ids = [f[0] for f in files]
    assert "202406010101" in race_ids
    assert "202401010101" not in race_ids


def test_collect_cache_files_date_filter_end(tmp_path):
    """--end より後の race_id はスキップされる。"""
    raw = tmp_path / "raw"
    _write_cache(raw, "202401010101", BASIC_HTML)
    _write_cache(raw, "202412310101", ALL_PAYOUT_HTML)

    files = _collect_cache_files(raw, start=None, end=datetime.date(2024, 1, 31))
    race_ids = [f[0] for f in files]
    assert "202401010101" in race_ids
    assert "202412310101" not in race_ids


def test_collect_cache_files_ignores_non_html(tmp_path):
    """HTML ファイル以外は無視される。"""
    raw = tmp_path / "raw"
    html_path = _write_cache(raw, _RACE_ID_ALL, ALL_PAYOUT_HTML)
    (html_path.parent / "note.txt").write_text("ignore me")

    files = _collect_cache_files(raw, start=None, end=None)
    assert len(files) == 1


def test_collect_cache_files_empty_dir(tmp_path):
    """キャッシュディレクトリが存在しない場合は空リストを返す。"""
    raw = tmp_path / "raw_nonexistent"
    files = _collect_cache_files(raw, start=None, end=None)
    assert files == []


# ── run_refill() ─────────────────────────────────────────────────────────────

def test_run_refill_inserts_payouts(db_session, tmp_path, monkeypatch):
    """races 行が存在するキャッシュに対して payouts を挿入する。"""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    raw = tmp_path / "raw"
    _write_cache(raw, _RACE_ID_ALL, ALL_PAYOUT_HTML)
    _insert_race(db_session, _RACE_ID_ALL)

    counters = run_refill(db_session)

    assert counters["processed"] == 1
    assert counters["errors"] == 0

    payouts = db_session.execute(select(Payout)).scalars().all()
    bet_types = {p.bet_type for p in payouts}
    assert {"単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "三連複", "三連単"} == bet_types


def test_run_refill_skips_missing_race_row(db_session, tmp_path, monkeypatch):
    """races 行が無いキャッシュは FK 制約回避のためスキップされる。"""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    raw = tmp_path / "raw"
    _write_cache(raw, _RACE_ID_ALL, ALL_PAYOUT_HTML)
    # races テーブルには何も挿入しない

    counters = run_refill(db_session)

    assert counters["skipped_no_race"] == 1
    assert counters["processed"] == 0

    payouts = db_session.execute(select(Payout)).scalars().all()
    assert payouts == []


def test_run_refill_idempotent(db_session, tmp_path, monkeypatch):
    """2 回実行しても payouts 行が重複しない（DELETE → INSERT の冪等性）。"""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    raw = tmp_path / "raw"
    _write_cache(raw, _RACE_ID_ALL, ALL_PAYOUT_HTML)
    _insert_race(db_session, _RACE_ID_ALL)

    run_refill(db_session)
    count_first = len(db_session.execute(select(Payout)).scalars().all())

    run_refill(db_session)
    count_second = len(db_session.execute(select(Payout)).scalars().all())

    assert count_first == count_second


def test_run_refill_limit(db_session, tmp_path, monkeypatch):
    """--limit N で処理ファイル数が制限される。"""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    raw = tmp_path / "raw"
    _write_cache(raw, "202406010101", ALL_PAYOUT_HTML)
    _write_cache(raw, "202406010102", ALL_PAYOUT_HTML)
    _insert_race(db_session, "202406010101")
    _insert_race(db_session, "202406010102")

    counters = run_refill(db_session, limit=1)
    assert counters["processed"] == 1


def test_run_refill_skips_no_payout_html(db_session, tmp_path, monkeypatch):
    """払戻テーブルが無い HTML は skipped_no_payouts としてカウントされる。"""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    raw = tmp_path / "raw"
    empty_html = "<html><body><p>no payout table</p></body></html>"
    _write_cache(raw, _RACE_ID_ALL, empty_html)
    _insert_race(db_session, _RACE_ID_ALL)

    counters = run_refill(db_session)
    assert counters["skipped_no_payouts"] == 1
    assert counters["processed"] == 0


def test_run_refill_date_range(db_session, tmp_path, monkeypatch):
    """--start / --end フィルタが run_refill に正しく伝わる。"""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    raw = tmp_path / "raw"
    _write_cache(raw, "202401010101", ALL_PAYOUT_HTML)
    _write_cache(raw, "202406010101", ALL_PAYOUT_HTML)
    _insert_race(db_session, "202401010101")
    _insert_race(db_session, "202406010101")

    counters = run_refill(
        db_session,
        start=datetime.date(2024, 6, 1),
        end=datetime.date(2024, 6, 30),
    )
    assert counters["processed"] == 1

    payouts = db_session.execute(
        select(Payout).where(Payout.race_id == "202401010101")
    ).scalars().all()
    assert payouts == [], "1月のレースは処理されていないはず"
