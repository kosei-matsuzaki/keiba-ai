"""Retro-fill race meta (name / race_class / surface / distance / 馬場) from cached HTML.

既存の data/raw/<yyyy>/<mm>/<race_id>.html キャッシュを走査して
parse_race_result() を再実行し、races テーブルを更新する（過去の誤検出バグ修正の
ため既存値は信用しない）。**ネットワークには一切アクセスしない。**

name / race_class は無条件で上書きし、surface / distance / track_condition は
**解析できたときだけ**上書きする（直すためのジョブが、解析漏れで既存の値を
空にしてしまわないようにするため）。

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
import sys

from sqlalchemy import update
from sqlalchemy.orm import Session

from core.logging import configure_logging, get_logger
from core.paths import db_path, raw_dir
from db.base import Base
from db.models.race import Race
from db.session import make_engine, session_scope
from jobs.cache_scan import add_range_args, select_cached_races
from scraper.parsers.race_result import ParseError, parse_race_result

logger = get_logger(__name__)



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
    cache_files, skipped_no_race = select_cached_races(
        session, raw_dir(), start=start, end=end, limit=limit
    )

    processed = 0
    skipped_parse_error = 0
    errors = 0
    class_dist: collections.Counter[str] = collections.Counter()

    for race_id, html_path in cache_files:
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
            # **surface / distance も直す。** コース形状の解析漏れで
            # surface='' / distance=0 のまま保存された行が 964 件あった
            # (新潟の直線・障害の襷コース)。ただし解析できなかったときに
            # 既存の値を空で上書きしない — 直すためのジョブが壊す側に回る。
            values: dict = {"name": parsed.name, "race_class": parsed.race_class}
            if parsed.surface:
                values["surface"] = parsed.surface
            if parsed.distance:
                values["distance"] = parsed.distance
            if parsed.track_condition:
                values["track_condition"] = parsed.track_condition
            session.execute(
                update(Race).where(Race.race_id == race_id).values(**values)
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
    parser = argparse.ArgumentParser(description="Retro-fill races.name / race_class from cached race HTML files")
    add_range_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(main(args))
