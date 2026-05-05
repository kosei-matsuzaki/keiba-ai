"""全未確定 bet を一括確定する CLI ジョブ。

Usage:
    uv run python -m keiba_ai.jobs.settle_bets
    uv run python -m keiba_ai.jobs.settle_bets --dry-run
"""

from __future__ import annotations

import argparse

import sqlalchemy as sa
from sqlalchemy import select

from keiba_ai.core.logging import configure_logging, get_logger
from keiba_ai.core.paths import db_path
from keiba_ai.db.models.bet_record import BetRecord
from keiba_ai.db.session import make_engine, session_scope
from keiba_ai.services.bet_settlement import settle_all_pending

logger = get_logger(__name__)


def run(*, dry_run: bool = False) -> int:
    """未確定 bet を確定する。dry_run=True の場合は対象件数を返すだけで DB を変更しない。"""
    configure_logging()
    engine = make_engine(db_path())

    with session_scope(engine) as session:
        pending_count = session.scalar(
            select(sa.func.count()).select_from(BetRecord).where(
                BetRecord.settled_at.is_(None)
            )
        ) or 0

        if dry_run:
            logger.info("[dry-run] 未確定 bet 件数: %d", pending_count)
            return pending_count

        settled = settle_all_pending(session)
        logger.info("確定した bet 件数: %d / 未確定合計: %d", settled, pending_count)
        return settled


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="全未確定 bet_records を一括確定する")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="実行せず対象件数のみ表示する",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(dry_run=args.dry_run)
    raise SystemExit(0)
