"""Retro-fill payouts from cached HTML.

既存の data/raw/<yyyy>/<mm>/<race_id>.html キャッシュを走査して
parse_payouts() を再実行し、payouts テーブルを更新する。

レースキャッシュが存在しても races テーブルに対応行が無い場合は FK 制約違反に
なるためスキップする。

Usage:
    uv run python -m jobs.refill_payouts
    uv run python -m jobs.refill_payouts --start 2024-01-01 --end 2024-12-31
    uv run python -m jobs.refill_payouts --limit 100
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys

from sqlalchemy.orm import Session

from core.logging import configure_logging, get_logger
from core.paths import db_path, raw_dir
from db.base import Base
from db.models.payout import Payout
from db.session import make_engine, session_scope
from jobs.cache_scan import add_range_args, select_cached_races
from scraper.parsers.payout import parse_payouts

logger = get_logger(__name__)



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
    cache_files, skipped_no_race = select_cached_races(
        session, raw_dir(), start=start, end=end, limit=limit
    )

    counters = {
        "processed": 0,
        # races 行が無いレースは FK 制約に触るので、走らせる前に落としてある。
        "skipped_no_race": skipped_no_race,
        "skipped_no_payouts": 0,
        "errors": 0,
    }

    for race_id, html_path in cache_files:
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
    parser = argparse.ArgumentParser(description="Retro-fill payouts table from cached race HTML files")
    add_range_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(main(args))
