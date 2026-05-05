"""Retro-fill payouts from cached HTML.

既存の data/raw/<yyyy>/<mm>/<race_id>.html キャッシュを走査して
parse_payouts() を再実行し、payouts テーブルを更新する。

レースキャッシュが存在しても races テーブルに対応行が無い場合は FK 制約違反に
なるためスキップする。

Usage:
    uv run python -m keiba_ai.jobs.refill_payouts
    uv run python -m keiba_ai.jobs.refill_payouts --start 2024-01-01 --end 2024-12-31
    uv run python -m keiba_ai.jobs.refill_payouts --limit 100
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from keiba_ai.core.logging import configure_logging, get_logger
from keiba_ai.core.paths import db_path, raw_dir
from keiba_ai.db.base import Base
from keiba_ai.db.models.payout import Payout
from keiba_ai.db.models.race import Race
from keiba_ai.db.session import make_engine, session_scope
from keiba_ai.scraper.parsers.payout import parse_payouts

logger = get_logger(__name__)

# race_id はファイル名から導出（12 桁数字）
_RACE_ID_RE = re.compile(r"^(\d{12})\.html$")


def _collect_cache_files(
    raw: Path,
    start: datetime.date | None,
    end: datetime.date | None,
) -> list[tuple[str, Path]]:
    """data/raw/<yyyy>/<mm>/<race_id>.html を列挙して (race_id, path) リストを返す。

    start/end でフィルタリングする場合は race_id 先頭 8 桁（YYYYMMDD）を使う。
    """
    result: list[tuple[str, Path]] = []

    if not raw.exists():
        return result

    for yyyy_dir in sorted(raw.iterdir()):
        if not yyyy_dir.is_dir() or not yyyy_dir.name.isdigit():
            continue

        for mm_dir in sorted(yyyy_dir.iterdir()):
            if not mm_dir.is_dir() or not mm_dir.name.isdigit():
                continue

            for html_file in sorted(mm_dir.iterdir()):
                m = _RACE_ID_RE.match(html_file.name)
                if not m:
                    continue
                race_id = m.group(1)

                if start is not None or end is not None:
                    # race_id 先頭 8 桁が YYYYMMDD
                    try:
                        race_date = datetime.date(
                            int(race_id[:4]),
                            int(race_id[4:6]),
                            int(race_id[6:8]),
                        )
                    except ValueError:
                        continue
                    if start is not None and race_date < start:
                        continue
                    if end is not None and race_date > end:
                        continue

                result.append((race_id, html_file))

    return result


def _race_exists(session: Session, race_id: str) -> bool:
    row = session.execute(
        select(Race.race_id).where(Race.race_id == race_id).limit(1)
    ).first()
    return row is not None


def run_refill(
    session: Session,
    start: datetime.date | None = None,
    end: datetime.date | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    """payouts テーブルを retro-fill する。

    Returns:
        counters: {"processed": int, "skipped_no_race": int, "skipped_no_payouts": int, "errors": int}
    """
    raw = raw_dir()
    cache_files = _collect_cache_files(raw, start, end)

    if limit is not None:
        cache_files = cache_files[:limit]

    counters = {
        "processed": 0,
        "skipped_no_race": 0,
        "skipped_no_payouts": 0,
        "errors": 0,
    }

    for race_id, html_path in cache_files:
        if not _race_exists(session, race_id):
            logger.debug("Skipping %s: no races row (FK constraint)", race_id)
            counters["skipped_no_race"] += 1
            continue

        try:
            html = html_path.read_text(encoding="utf-8")
            payout_rows = parse_payouts(html)

            if not payout_rows:
                logger.debug("Skipping %s: parse_payouts returned no rows", race_id)
                counters["skipped_no_payouts"] += 1
                continue

            # DELETE → INSERT で冪等更新
            session.execute(
                Payout.__table__.delete().where(Payout.race_id == race_id)
            )
            for row in payout_rows:
                session.add(Payout(
                    race_id=race_id,
                    bet_type=row.bet_type,
                    combo=row.combo,
                    amount=row.amount,
                    popularity=row.popularity,
                ))
            session.commit()

            counters["processed"] += 1
            progress = {
                "race_id": race_id,
                "status": "done",
                "payouts": len(payout_rows),
            }
            print(json.dumps(progress), flush=True)

        except Exception as exc:
            logger.error("Error refilling payouts for %s: %s", race_id, exc)
            session.rollback()
            counters["errors"] += 1
            progress = {"race_id": race_id, "status": "error", "message": str(exc)}
            print(json.dumps(progress), flush=True)

    return counters


def main(args: argparse.Namespace) -> int:
    configure_logging()
    engine = make_engine(db_path())
    Base.metadata.create_all(engine)

    start = datetime.date.fromisoformat(args.start) if args.start else None
    end = datetime.date.fromisoformat(args.end) if args.end else None
    limit = args.limit

    with session_scope(engine) as session:
        counters = run_refill(session, start=start, end=end, limit=limit)

    logger.info(
        "Refill complete — processed=%d skipped_no_race=%d skipped_no_payouts=%d errors=%d",
        counters["processed"],
        counters["skipped_no_race"],
        counters["skipped_no_payouts"],
        counters["errors"],
    )
    summary = {"summary": counters}
    print(json.dumps(summary), flush=True)
    return 0 if counters["errors"] == 0 else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retro-fill payouts table from cached race HTML files"
    )
    parser.add_argument(
        "--start",
        default=None,
        metavar="YYYY-MM-DD",
        help="Start date (inclusive). Filters by race_id date prefix.",
    )
    parser.add_argument(
        "--end",
        default=None,
        metavar="YYYY-MM-DD",
        help="End date (inclusive). Filters by race_id date prefix.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of cache files to process (debug use).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(main(args))
