"""シミュレーションエンドポイント。

- GET  /api/simulation/active_model      シンクロ実行 (3 分以内の小さい window 用、後方互換)
- POST /api/simulation/start             バックグラウンドジョブで実行 (大きい window OK)
- GET  /api/simulation/runs              保存済み実行 一覧
- GET  /api/simulation/runs/{id}         保存済み実行 詳細
- DELETE /api/simulation/runs/{id}       保存済み実行 削除
"""

from __future__ import annotations

import asyncio
import json
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from keiba_ai.ai.registry import get_active
from keiba_ai.ai.simulation import (
    STRATEGY_PRESETS,
    SimulationResult,
    simulate_active_model,
)
from keiba_ai.ai.simulation_persistence import (
    delete_simulation_run,
    get_simulation_run,
    list_simulation_runs,
    save_simulation_result,
)
from keiba_ai.api.deps import get_engine, get_job_registry, get_session
from keiba_ai.api.jobs import JobRegistry
from keiba_ai.api.schemas import JobAccepted
from keiba_ai.core.logging import get_logger
from keiba_ai.db.models.simulation_run import SimulationRun
from keiba_ai.db.session import session_scope

logger = get_logger(__name__)

router = APIRouter()

# 1 年 ≒ 3000 race で逐次 predict + settle すると 5 分以上かかり frontend HTTP
# timeout に当たる。実用上は 6 か月 (~1500 race) が上限の目安。
MAX_WINDOW_DAYS: int = 186


class GroupStatsResponse(BaseModel):
    label: str
    n_bets: int
    invested: int
    payout: int
    payback_rate: float
    hit_rate: float


class BankrollPointResponse(BaseModel):
    """日次の資産推移ポイント (グラフ表示用)。"""
    date: str          # YYYY-MM-DD
    bankroll: int      # その日の最終 race 後の残高
    invested: int      # その日の累計 stake
    payout: int        # その日の累計 payout (整数化)
    n_bets: int


class SimulationWindow(BaseModel):
    start: str | None
    end: str | None


class SimulationResponse(BaseModel):
    """シミュレーション完全結果。実行直後のレスポンスと、保存済み run 詳細の
    両方で使う。run_id は実行直後のみセット (保存済み詳細は別レスポンス) 。"""
    window: SimulationWindow
    model_path: str
    strategy: str
    budget: int
    n_races: int
    n_settled_races: int
    final_bankroll: int
    peak_bankroll: int
    summary: GroupStatsResponse
    by_bet_type: list[GroupStatsResponse]
    by_race_class: list[GroupStatsResponse]
    by_course: list[GroupStatsResponse]
    bankroll_timeseries: list[BankrollPointResponse]
    # 実行直後にバックエンドが保存した row の id。再呼び出しで詳細を取得可能。
    run_id: int | None = None


class SimulationRunSummary(BaseModel):
    """保存済み実行の一覧表示用 (重い json は含めない)。"""
    id: int
    created_at: str
    budget: int
    strategy: str
    window_start: str | None
    window_end: str | None
    n_races: int
    n_settled_races: int
    final_bankroll: int
    peak_bankroll: int


class SimulationRunListResponse(BaseModel):
    runs: list[SimulationRunSummary]
    total: int


def _result_to_response(
    r: SimulationResult, run_id: int | None = None
) -> SimulationResponse:
    """Convert SimulationResult dataclass to pydantic response model."""
    d = r.as_dict()
    return SimulationResponse(
        window=SimulationWindow(**d["window"]),
        model_path=d["model_path"],
        strategy=d["strategy"],
        budget=d["budget"],
        n_races=d["n_races"],
        n_settled_races=d["n_settled_races"],
        final_bankroll=d["final_bankroll"],
        peak_bankroll=d["peak_bankroll"],
        summary=GroupStatsResponse(**d["summary"]),
        by_bet_type=[GroupStatsResponse(**g) for g in d["by_bet_type"]],
        by_race_class=[GroupStatsResponse(**g) for g in d["by_race_class"]],
        by_course=[GroupStatsResponse(**g) for g in d["by_course"]],
        bankroll_timeseries=[
            BankrollPointResponse(**p) for p in d["bankroll_timeseries"]
        ],
        run_id=run_id,
    )


def _row_to_response(row: SimulationRun) -> SimulationResponse:
    """保存済み SimulationRun row → SimulationResponse (json を decode)."""
    return SimulationResponse(
        window=SimulationWindow(start=row.window_start, end=row.window_end),
        model_path=row.model_path,
        strategy=row.strategy,
        budget=row.budget,
        n_races=row.n_races,
        n_settled_races=row.n_settled_races,
        final_bankroll=row.final_bankroll,
        peak_bankroll=row.peak_bankroll,
        summary=GroupStatsResponse(**json.loads(row.summary_json)),
        by_bet_type=[
            GroupStatsResponse(**g) for g in json.loads(row.by_bet_type_json)
        ],
        by_race_class=[
            GroupStatsResponse(**g) for g in json.loads(row.by_race_class_json)
        ],
        by_course=[GroupStatsResponse(**g) for g in json.loads(row.by_course_json)],
        bankroll_timeseries=[
            BankrollPointResponse(**p)
            for p in json.loads(row.bankroll_timeseries_json)
        ],
        run_id=row.id,
    )


def _row_to_summary(row: SimulationRun) -> SimulationRunSummary:
    return SimulationRunSummary(
        id=row.id,
        created_at=row.created_at,
        budget=row.budget,
        strategy=row.strategy,
        window_start=row.window_start,
        window_end=row.window_end,
        n_races=row.n_races,
        n_settled_races=row.n_settled_races,
        final_bankroll=row.final_bankroll,
        peak_bankroll=row.peak_bankroll,
    )


def _validate_request(
    start: str | None,
    end: str | None,
    strategy: str,
) -> None:
    """戦略 / 期間の妥当性を確認。違反は HTTPException(400) を投げる。

    sync run と async start で共通利用。
    """
    if strategy not in STRATEGY_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown strategy {strategy!r}. Choose from {list(STRATEGY_PRESETS)}.",
        )

    if start is not None and end is not None:
        try:
            d_start = date.fromisoformat(start)
            d_end = date.fromisoformat(end)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"start / end は YYYY-MM-DD 形式で指定してください: {exc}",
            ) from exc
        if d_end < d_start:
            raise HTTPException(
                status_code=400,
                detail="end は start 以降の日付を指定してください。",
            )
        if (d_end - d_start).days > MAX_WINDOW_DAYS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"期間が長すぎます (max {MAX_WINDOW_DAYS} 日 ≒ 6 か月)。"
                    " 1 年規模だと逐次 predict + settle が数分かかります。"
                    " 6 か月以内で分割実行するか、それを超える window が必要なら"
                    " バックグラウンドジョブ (POST /api/simulation/start) を使ってください。"
                ),
            )


@router.get(
    "/simulation/active_model",
    response_model=SimulationResponse,
)
def run_simulation(
    session: Annotated[Session, Depends(get_session)],
    start: Annotated[str | None, Query(description="窓の開始日 YYYY-MM-DD")] = None,
    end: Annotated[str | None, Query(description="窓の終了日 YYYY-MM-DD")] = None,
    budget: Annotated[
        int,
        Query(
            ge=1000,
            le=100_000_000,
            description="初期資産 (円)。各 race ごとに残資産 (= budget + 累計 profit) を "
            "bankroll として Kelly stake を計算する (compounding wealth)。"
            "payout は次 race の bet 余力に加算される。資産尽きたら以降は実質 bet しない。",
        ),
    ] = 100_000,
    strategy: Annotated[
        Literal["conservative", "balanced", "aggressive"],
        Query(description="戦略プリセット"),
    ] = "balanced",
    max_stake_per_race_yen: Annotated[
        int | None,
        Query(
            ge=0,
            le=10_000_000,
            description="1 race の累計 stake 絶対上限 (円)。0 / 未指定で無効 "
            "(% cap のみ)。compounding wealth で bankroll が増えても各 race の "
            "投資額をこの値で頭打ちにできる。",
        ),
    ] = None,
) -> SimulationResponse:
    """Run end-to-end backtest with active model on the given window.

    動作:
      1. アクティブなモデルを load (binary + calibrator 含む)
      2. 期間内の全レースに対して predict + recommendation を生成
      3. 実 finish_position と payouts で settle
      4. bet_type / race_class / course でアグリゲート

    所要時間: 800 race で ~30-60 秒。レスポンスはキャッシュされない。
    """
    _validate_request(start, end, strategy)

    active_path = get_active(session)
    if active_path is None:
        raise HTTPException(
            status_code=503,
            detail="アクティブなモデルがありません。Models 画面でモデルを active 化してください。",
        )

    logger.info(
        "Simulation request: window=%s..%s, budget=%d, strategy=%s",
        start, end, budget, strategy,
    )

    result = simulate_active_model(
        session=session,
        model_path=Path(active_path),
        start=start,
        end=end,
        budget=budget,
        strategy=strategy,  # type: ignore[arg-type]
        max_stake_per_race_yen=max_stake_per_race_yen,
    )

    # 自動保存 (上限 50 件、超過したら古い順に削除)
    saved = save_simulation_result(session, result)
    logger.info("Simulation result saved as run id=%d", saved.id)

    return _result_to_response(result, run_id=saved.id)


# ---------------------------------------------------------------------------
# Saved runs (list / detail / delete)
# ---------------------------------------------------------------------------


@router.get(
    "/simulation/runs",
    response_model=SimulationRunListResponse,
)
def list_runs(
    session: Annotated[Session, Depends(get_session)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> SimulationRunListResponse:
    """保存済みシミュレーション実行の一覧を新しい順で返す (重い json は含まない)。"""
    runs = list_simulation_runs(session, limit=limit)
    return SimulationRunListResponse(
        runs=[_row_to_summary(r) for r in runs],
        total=len(runs),
    )


@router.get(
    "/simulation/runs/{run_id}",
    response_model=SimulationResponse,
)
def get_run(
    run_id: int,
    session: Annotated[Session, Depends(get_session)],
) -> SimulationResponse:
    """保存済みシミュレーション実行の詳細を返す (グラフ + 全テーブル含む)。"""
    row = get_simulation_run(session, run_id)
    if row is None:
        raise HTTPException(
            status_code=404, detail=f"simulation run id={run_id} が見つかりません",
        )
    return _row_to_response(row)


@router.delete("/simulation/runs/{run_id}")
def delete_run(
    run_id: int,
    session: Annotated[Session, Depends(get_session)],
) -> dict:
    """保存済みシミュレーション実行を削除する。"""
    if not delete_simulation_run(session, run_id):
        raise HTTPException(
            status_code=404, detail=f"simulation run id={run_id} が見つかりません",
        )
    return {"deleted": run_id}


# ---------------------------------------------------------------------------
# Background job (long-running simulation)
# ---------------------------------------------------------------------------


# Background job 用は MAX_WINDOW_DAYS の cap を緩める (1 年まで)。HTTP timeout を
# 気にしなくて良いので、もう少し長くても OK。
MAX_BG_WINDOW_DAYS: int = 366


def _validate_request_bg(
    start: str | None,
    end: str | None,
    strategy: str,
) -> None:
    """background job 用 validation。期間 cap は MAX_BG_WINDOW_DAYS まで。"""
    if strategy not in STRATEGY_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown strategy {strategy!r}. Choose from {list(STRATEGY_PRESETS)}.",
        )
    if start is not None and end is not None:
        try:
            d_start = date.fromisoformat(start)
            d_end = date.fromisoformat(end)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=f"start / end は YYYY-MM-DD 形式で指定してください: {exc}",
            ) from exc
        if d_end < d_start:
            raise HTTPException(
                status_code=400,
                detail="end は start 以降の日付を指定してください。",
            )
        if (d_end - d_start).days > MAX_BG_WINDOW_DAYS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"期間が長すぎます (max {MAX_BG_WINDOW_DAYS} 日 ≒ 1 年)。"
                ),
            )


@router.post(
    "/simulation/start",
    response_model=JobAccepted,
)
async def start_simulation_job(
    session: Annotated[Session, Depends(get_session)],
    registry: Annotated[JobRegistry, Depends(get_job_registry)],
    engine: Annotated[Engine, Depends(get_engine)],
    start: Annotated[str | None, Query(description="YYYY-MM-DD")] = None,
    end: Annotated[str | None, Query(description="YYYY-MM-DD")] = None,
    budget: Annotated[
        int,
        Query(ge=1000, le=100_000_000, description="初期資産 (円)"),
    ] = 100_000,
    strategy: Annotated[
        Literal["conservative", "balanced", "aggressive"],
        Query(description="戦略プリセット"),
    ] = "balanced",
    max_stake_per_race_yen: Annotated[
        int | None,
        Query(
            ge=0, le=10_000_000,
            description="1 race の累計 stake 絶対上限 (円)。0 / 未指定で無効。",
        ),
    ] = None,
) -> JobAccepted:
    """シミュレーションをバックグラウンド job として実行する。

    HTTP timeout を気にせず長い window (最大 1 年) を扱える。
    完了後 job.result.run_id に保存済み run の id が入るので、UI は
    /api/simulation/runs/{run_id} で詳細を取得すれば良い。

    NOTE: async def で宣言する必要がある (registry.start 内部で
    asyncio.create_task を呼ぶため、event loop 上で動かす必要がある)。
    """
    _validate_request_bg(start, end, strategy)
    active_path = get_active(session)
    if active_path is None:
        raise HTTPException(
            status_code=503,
            detail="アクティブなモデルがありません。",
        )

    logger.info(
        "Simulation job submit: window=%s..%s, budget=%d, strategy=%s",
        start, end, budget, strategy,
    )

    # asyncio.create_task の中で session を作るため、Engine だけを capture。
    # request 由来の session を job loop 内で使うと scope が合わない。
    captured_engine = engine
    captured_path = Path(active_path)

    def _run_simulation_blocking() -> int:
        """Worker thread: open new session + run + save。Returns saved run id."""
        with session_scope(captured_engine) as bg_session:
            result = simulate_active_model(
                session=bg_session,
                model_path=captured_path,
                start=start,
                end=end,
                budget=budget,
                strategy=strategy,  # type: ignore[arg-type]
                max_stake_per_race_yen=max_stake_per_race_yen,
            )
            saved = save_simulation_result(bg_session, result)
            saved_id = saved.id
        return saved_id

    async def _coro() -> dict:
        # Heavy CPU/IO work は別スレッドで (event loop を block しない)
        run_id = await asyncio.to_thread(_run_simulation_blocking)
        logger.info("Simulation job completed: saved run_id=%d", run_id)
        return {"run_id": run_id}

    info = registry.start("simulation", _coro)
    return JobAccepted(
        job_id=info.job_id,
        status=info.status,
        started_at=info.started_at,
    )
