"""Persistence helpers for SimulationRun.

シミュレーション結果を simulation_runs テーブルに保存し、上限 50 件を維持する。
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime

from sqlalchemy import desc, func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from ai.simulation import SimulationResult
from db.models.simulation_run import SimulationRun

log = logging.getLogger(__name__)

MAX_SAVED_RUNS: int = 50

# `database is locked` 用の retry パラメータ。busy_timeout=30s と合わせて、
# 並行する 重い ingest job の影響を吸収する。
_SAVE_MAX_RETRIES: int = 5
_SAVE_RETRY_BASE_SLEEP: float = 1.0  # exponential backoff: 1, 2, 4, 8, 16


def save_simulation_result(
    session: Session, result: SimulationResult, model_run_id: int
) -> SimulationRun:
    """SimulationResult を simulation_runs テーブルに保存する。

    Args:
        model_run_id: バックテストに使ったモデル (model_runs.id)。NOT NULL FK。

    保存後、件数が MAX_SAVED_RUNS (= 50) を超える場合は created_at 古い順に削除する。
    `database is locked` のときは exponential backoff で retry (最大 5 回)。
    Returns: 新規作成された SimulationRun (id 付き)。
    """
    d = result.as_dict()
    now = datetime.now(UTC).isoformat()

    payload = dict(
        created_at=now,
        model_run_id=model_run_id,
        budget=d["budget"],
        strategy=d["strategy"],
        window_start=d["window"]["start"],
        window_end=d["window"]["end"],
        model_path=d["model_path"],
        n_races=d["n_races"],
        n_settled_races=d["n_settled_races"],
        final_bankroll=d["final_bankroll"],
        peak_bankroll=d["peak_bankroll"],
        summary_json=json.dumps(d["summary"], ensure_ascii=False),
        by_bet_type_json=json.dumps(d["by_bet_type"], ensure_ascii=False),
        by_race_class_json=json.dumps(d["by_race_class"], ensure_ascii=False),
        by_course_json=json.dumps(d["by_course"], ensure_ascii=False),
        bankroll_timeseries_json=json.dumps(
            d["bankroll_timeseries"], ensure_ascii=False
        ),
    )

    last_exc: Exception | None = None
    for attempt in range(1, _SAVE_MAX_RETRIES + 1):
        try:
            run = SimulationRun(**payload)
            session.add(run)
            session.flush()  # id を確定
            _prune_old_runs(session)
            session.commit()
            return run
        except OperationalError as exc:
            # database is locked 系のみ retry。それ以外は即 raise。
            if "database is locked" not in str(exc):
                raise
            last_exc = exc
            session.rollback()
            sleep_sec = _SAVE_RETRY_BASE_SLEEP * (2 ** (attempt - 1))
            log.warning(
                "save_simulation_result: db locked (attempt %d/%d), retry after %.1fs",
                attempt, _SAVE_MAX_RETRIES, sleep_sec,
            )
            time.sleep(sleep_sec)
    # fall through: 全 retry 失敗
    assert last_exc is not None
    raise last_exc


def _prune_old_runs(session: Session) -> int:
    """件数が MAX_SAVED_RUNS を超えていれば created_at 古い順に削除。

    Returns: 削除した件数。
    """
    n = session.scalar(select(func.count()).select_from(SimulationRun)) or 0
    if n <= MAX_SAVED_RUNS:
        return 0

    n_to_delete = n - MAX_SAVED_RUNS
    old_ids = list(
        session.scalars(
            select(SimulationRun.id)
            .order_by(SimulationRun.created_at.asc())
            .limit(n_to_delete)
        )
    )
    for run in session.scalars(
        select(SimulationRun).where(SimulationRun.id.in_(old_ids))
    ):
        session.delete(run)
    return len(old_ids)


def list_simulation_runs(
    session: Session, limit: int = 50, model_run_id: int | None = None
) -> list[SimulationRun]:
    """新しい順に最大 limit 件を返す (一覧表示用)。

    model_run_id を渡すとそのモデルの実行のみに絞る (モデル詳細画面用)。
    """
    stmt = select(SimulationRun)
    if model_run_id is not None:
        stmt = stmt.where(SimulationRun.model_run_id == model_run_id)
    stmt = stmt.order_by(desc(SimulationRun.created_at)).limit(limit)
    return list(session.scalars(stmt))


def get_simulation_run(session: Session, run_id: int) -> SimulationRun | None:
    """id 指定で 1 件取得。存在しないとき None。"""
    return session.get(SimulationRun, run_id)


def delete_simulation_run(session: Session, run_id: int) -> bool:
    """id 指定で削除。削除した場合 True、見つからなければ False。"""
    run = session.get(SimulationRun, run_id)
    if run is None:
        return False
    session.delete(run)
    session.commit()
    return True
