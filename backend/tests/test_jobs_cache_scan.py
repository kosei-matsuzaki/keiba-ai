"""Tests for jobs/cache_scan.py — キャッシュ走査と期間の絞り込み。

肝は **race_id を日付として読まない**こと。netkeiba の race_id は
年(4) + 競馬場(2) + 回(2) + 日(2) + R(2) で、`race_id[4:6]` は月ではない。
以前この 2 つの refill ジョブは race_id を日付として parse していて、--start /
--end を付けるとほとんどのレースが黙って範囲外に落ちていた。ここのテストは
**race_id の見かけの日付と races.date がずれている**レースを使って、それを固定する。
"""

from __future__ import annotations

import datetime
from pathlib import Path

from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from db.models.race import Race
from jobs.cache_scan import collect_cache_files, race_dates, select_cached_races

# 2024年・第5場・2回・12日・11R。race_id を日付として読むと 2024-05-02 になるが、
# 実際の開催日は 2024-11-02。この食い違いがバグを踏むかどうかの分かれ目。
RACE_ID = "202405021211"
RACE_DATE = "2024-11-02"


def _write_cache(raw_root: Path, race_id: str, yyyy: str, mm: str) -> Path:
    cache_dir = raw_root / yyyy / mm
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{race_id}.html"
    path.write_text("<html></html>", encoding="utf-8")
    return path


def _insert_race(session, race_id: str, date: str) -> None:
    session.execute(
        sqlite_insert(Race)
        .values(race_id=race_id, date=date, course="東京", surface="芝", distance=2000)
        .on_conflict_do_nothing(index_elements=["race_id"])
    )
    session.commit()


# ── collect_cache_files() ────────────────────────────────────────────────────

def test_collect_finds_html(tmp_path):
    raw = tmp_path / "raw"
    _write_cache(raw, RACE_ID, "2024", "11")

    files = collect_cache_files(raw)
    assert [race_id for race_id, _ in files] == [RACE_ID]


def test_collect_ignores_non_html(tmp_path):
    raw = tmp_path / "raw"
    path = _write_cache(raw, RACE_ID, "2024", "11")
    (path.parent / "note.txt").write_text("ignore me")

    assert len(collect_cache_files(raw)) == 1


def test_collect_missing_dir_is_empty(tmp_path):
    assert collect_cache_files(tmp_path / "nope") == []


# ── race_dates() ─────────────────────────────────────────────────────────────

def test_race_dates_skips_unknown_ids(db_session):
    _insert_race(db_session, RACE_ID, RACE_DATE)

    dates = race_dates(db_session, [RACE_ID, "209912345678"])
    assert dates == {RACE_ID: RACE_DATE}


# ── select_cached_races() ────────────────────────────────────────────────────

def test_select_filters_by_races_date_not_race_id(db_session, tmp_path):
    """期間は races.date で判定する (race_id の見かけの日付ではない)。

    RACE_ID を日付として読むと 2024-05-02 になるので、旧実装はこの窓で 0 件に
    なっていた。実際の開催日 2024-11-02 で判定すれば入る。
    """
    raw = tmp_path / "raw"
    _write_cache(raw, RACE_ID, "2024", "11")
    _insert_race(db_session, RACE_ID, RACE_DATE)

    selected, skipped = select_cached_races(
        db_session, raw, start=datetime.date(2024, 10, 1), end=datetime.date(2024, 12, 31)
    )
    assert [race_id for race_id, _ in selected] == [RACE_ID]
    assert skipped == 0


def test_select_excludes_outside_window(db_session, tmp_path):
    raw = tmp_path / "raw"
    _write_cache(raw, RACE_ID, "2024", "11")
    _insert_race(db_session, RACE_ID, RACE_DATE)

    selected, _ = select_cached_races(
        db_session, raw, start=datetime.date(2025, 1, 1)
    )
    assert selected == []


def test_select_counts_races_missing_from_db(db_session, tmp_path):
    """races に行が無いレースは FK 制約に触るので、走らせる前に落として数える。"""
    raw = tmp_path / "raw"
    _write_cache(raw, RACE_ID, "2024", "11")

    selected, skipped = select_cached_races(db_session, raw)
    assert selected == []
    assert skipped == 1


def test_select_applies_limit_after_the_window(db_session, tmp_path):
    """limit は期間で絞ったあとに掛ける。

    先に limit を掛けると、範囲外のファイルで枠を使い切って 1 件も処理されない
    ことがある (ファイルは race_id 順に並ぶので、古い年から埋まる)。
    """
    raw = tmp_path / "raw"
    old_id = "202305021211"
    _write_cache(raw, old_id, "2023", "11")
    _write_cache(raw, RACE_ID, "2024", "11")
    _insert_race(db_session, old_id, "2023-11-04")
    _insert_race(db_session, RACE_ID, RACE_DATE)

    selected, _ = select_cached_races(
        db_session, raw, start=datetime.date(2024, 1, 1), limit=1
    )
    assert [race_id for race_id, _ in selected] == [RACE_ID]
