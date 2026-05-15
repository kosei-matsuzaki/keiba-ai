"""Retro-fill race name and race_class from cached HTML.

既存の data/raw/<yyyy>/<mm>/<race_id>.html キャッシュを走査して
parse_race_result() を再実行し、races テーブルの name / race_class を
無条件上書きする（過去の誤検出バグ修正のため既存値は信用しない）。

完了後に race_class の分布を標準出力に JSON で出力する。

Usage:
    uv run python -m jobs.refill_race_meta
    uv run python -m jobs.refill_race_meta --start 2024-01-01 --end 2024-12-31
    uv run python -m jobs.refill_race_meta --limit 100
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import re
import sys
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from core.logging import configure_logging, get_logger
from core.paths import db_path, raw_dir
from db.base import Base
from db.models.race import Race
from db.session import make_engine, session_scope
from scraper.parsers.race_result import ParseError, parse_race_result

logger = get_logger(__name__)

# race_id はファイル名から導出（12 桁数字）
_RACE_ID_RE = re.compile(r"^(\d{12})\.html$")


def _collect_cache_files(
    raw: Path,
    start: datetime.date | None,
    end: datetime.date | None,
) -> list[tuple[str, Path]]:
    """data/raw/<yyyy>/<mm>/<race_id>.html を列挙して (race_id, path) リストを返す。"""
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


def run_refill_race_meta(
    session: Session,
    start: datetime.date | None = None,
    end: datetime.date | None = None,
    limit: int | None = None,
) -> dict[str, int | dict[str, int]]:
    """races テーブルの name / race_class を retro-fill する。

    Returns:
        counters: {
            "processed": int,
            "skipped_no_race": int,
            "skipped_parse_error": int,
            "errors": int,
            "class_distribution": {race_class: count},
        }
    """
    raw = raw_dir()
    cache_files = _collect_cache_files(raw, start, end)

    if limit is not None:
        cache_files = cache_files[:limit]

    processed = 0
    skipped_no_race = 0
    skipped_parse_error = 0
    errors = 0
    class_dist: collections.Counter[str] = collections.Counter()

    for race_id, html_path in cache_files:
        if not _race_exists(session, race_id):
            logger.debug("Skipping %s: no races row", race_id)
            skipped_no_race += 1
            continue

        try:
            html = html_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.error("Cannot read %s: %s", html_path, exc)
            errors += 1
            continue

        try:
            parsed = parse_race_result(html, race_id)
        except ParseError as exc:
            logger.debug("ParseError for %s: %s", race_id, exc)
            skipped_parse_error += 1
            continue
        except Exception as exc:
            logger.error("Unexpected parse error for %s: %s", race_id, exc)
            errors += 1
            continue

        try:
            session.execute(
                update(Race)
                .where(Race.race_id == race_id)
                .values(name=parsed.name, race_class=parsed.race_class)
            )
            session.commit()

            processed += 1
            class_key = parsed.race_class or "(None)"
            class_dist[class_key] += 1

            progress = {
                "race_id": race_id,
                "status": "done",
                "name": parsed.name,
                "race_class": parsed.race_class,
            }
            print(json.dumps(progress, ensure_ascii=False), flush=True)

        except Exception as exc:
            logger.error("DB error for %s: %s", race_id, exc)
            session.rollback()
            errors += 1

    return {
        "processed": processed,
        "skipped_no_race": skipped_no_race,
        "skipped_parse_error": skipped_parse_error,
        "errors": errors,
        "class_distribution": dict(class_dist.most_common()),
    }


def main(args: argparse.Namespace) -> int:
    configure_logging()
    engine = make_engine(db_path())
    Base.metadata.create_all(engine)

    start = datetime.date.fromisoformat(args.start) if args.start else None
    end = datetime.date.fromisoformat(args.end) if args.end else None

    with session_scope(engine) as session:
        counters = run_refill_race_meta(session, start=start, end=end, limit=args.limit)

    logger.info(
        "Refill race meta complete — processed=%d skipped_no_race=%d "
        "skipped_parse_error=%d errors=%d",
        counters["processed"],
        counters["skipped_no_race"],
        counters["skipped_parse_error"],
        counters["errors"],
    )
    logger.info("race_class distribution: %s", counters["class_distribution"])

    summary = {"summary": counters}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if counters["errors"] == 0 else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retro-fill races.name / race_class from cached race HTML files"
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
