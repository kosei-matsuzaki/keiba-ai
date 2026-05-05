"""Tests for jobs/refill_race_meta.py — retro-fill race name / class."""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from keiba_ai.db.models.race import Race
from keiba_ai.jobs.refill_race_meta import run_refill_race_meta

FIXTURES = Path(__file__).parent / "fixtures"

G1_HTML = (FIXTURES / "race_result_real_db_netkeiba.html").read_text(encoding="utf-8")
MAIDEN_HTML = (FIXTURES / "race_result_maiden_db_netkeiba.html").read_text(encoding="utf-8")


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
    """races テーブルに最小限の行を挿入。"""
    stmt = sqlite_insert(Race).values(
        race_id=race_id,
        date=f"{race_id[:4]}-{race_id[4:6]}-{race_id[6:8]}",
        course="中山",
        surface="芝",
        distance=2500,
    ).on_conflict_do_nothing(index_elements=["race_id"])
    session.execute(stmt)
    session.commit()


# ── run_refill_race_meta() ─────────────────────────────────────────────────────

def test_refill_updates_g1_name_and_class(db_session, tmp_path, monkeypatch):
    """G1 レースの HTML から name='有馬記念', race_class='G1' に更新される。"""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    race_id = "202412220601"
    raw = tmp_path / "raw"
    _write_cache(raw, race_id, G1_HTML)
    _insert_race(db_session, race_id)

    counters = run_refill_race_meta(db_session)

    assert counters["processed"] == 1
    assert counters["errors"] == 0

    row = db_session.get(Race, race_id)
    assert row is not None
    assert row.name == "有馬記念"
    assert row.race_class == "G1"


def test_refill_updates_maiden_name_and_class(db_session, tmp_path, monkeypatch):
    """未勝利レースの HTML から name='3歳未勝利', race_class='未勝利' に更新される。"""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    race_id = "202402241001"
    raw = tmp_path / "raw"
    _write_cache(raw, race_id, MAIDEN_HTML)
    _insert_race(db_session, race_id)

    counters = run_refill_race_meta(db_session)

    assert counters["processed"] == 1
    assert counters["errors"] == 0

    row = db_session.get(Race, race_id)
    assert row is not None
    assert row.name == "3歳未勝利"
    assert row.race_class == "未勝利"


def test_refill_skips_missing_race_row(db_session, tmp_path, monkeypatch):
    """races テーブルに行が無いキャッシュはスキップされる。"""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    race_id = "202412220601"
    raw = tmp_path / "raw"
    _write_cache(raw, race_id, G1_HTML)
    # races テーブルには何も挿入しない

    counters = run_refill_race_meta(db_session)

    assert counters["skipped_no_race"] == 1
    assert counters["processed"] == 0


def test_refill_overwrites_existing_wrong_class(db_session, tmp_path, monkeypatch):
    """既存の race_class='OP'（誤検出バグ由来）を正しい値で上書きする。"""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    race_id = "202412220601"
    raw = tmp_path / "raw"
    _write_cache(raw, race_id, G1_HTML)

    # 誤った class で race を挿入
    stmt = sqlite_insert(Race).values(
        race_id=race_id,
        date="2024-12-22",
        course="中山",
        surface="芝",
        distance=2500,
        race_class="OP",  # 誤検出バグ由来
        name=None,
    ).on_conflict_do_nothing(index_elements=["race_id"])
    db_session.execute(stmt)
    db_session.commit()

    counters = run_refill_race_meta(db_session)
    assert counters["processed"] == 1

    row = db_session.get(Race, race_id)
    assert row is not None
    assert row.race_class == "G1"  # 正しい値に上書きされていること
    assert row.name == "有馬記念"


def test_refill_limit(db_session, tmp_path, monkeypatch):
    """--limit N で処理ファイル数が制限される。"""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    raw = tmp_path / "raw"
    _write_cache(raw, "202412220601", G1_HTML)
    _write_cache(raw, "202402241001", MAIDEN_HTML)
    _insert_race(db_session, "202412220601")
    _insert_race(db_session, "202402241001")

    counters = run_refill_race_meta(db_session, limit=1)
    assert counters["processed"] == 1


def test_refill_date_filter(db_session, tmp_path, monkeypatch):
    """--start / --end 日付フィルタで対象を絞り込める。"""
    monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
    raw = tmp_path / "raw"
    _write_cache(raw, "202412220601", G1_HTML)     # 2024-12-22
    _write_cache(raw, "202402241001", MAIDEN_HTML)  # 2024-02-24
    _insert_race(db_session, "202412220601")
    _insert_race(db_session, "202402241001")

    counters = run_refill_race_meta(
        db_session,
        start=datetime.date(2024, 12, 1),
        end=datetime.date(2024, 12, 31),
    )
    assert counters["processed"] == 1

    # 2月のレースは処理されていないこと
    row = db_session.get(Race, "202402241001")
    assert row is not None
    assert row.name is None  # 更新されていない
