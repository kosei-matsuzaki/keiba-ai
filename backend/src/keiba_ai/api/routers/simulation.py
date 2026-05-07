"""GET /api/simulation/active_model — backtest using active model.

Used by the Ledger 「シミュレーション」 tab.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
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
from keiba_ai.api.deps import get_session
from keiba_ai.core.logging import get_logger
from keiba_ai.db.models.simulation_run import SimulationRun

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
) -> SimulationResponse:
    """Run end-to-end backtest with active model on the given window.

    動作:
      1. アクティブなモデルを load (binary + calibrator 含む)
      2. 期間内の全レースに対して predict + recommendation を生成
      3. 実 finish_position と payouts で settle
      4. bet_type / race_class / course でアグリゲート

    所要時間: 800 race で ~30-60 秒。レスポンスはキャッシュされない。
    """
    if strategy not in STRATEGY_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"unknown strategy {strategy!r}. Choose from {list(STRATEGY_PRESETS)}.",
        )

    # 期間の上限 check (frontend HTTP timeout を未然防止)。
    # 1 年だと数分かかり HTTP timeout に当たるため 6 か月で頭打ちにする。
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
                    " 1 年規模だと逐次 predict + settle が数分かかり HTTP timeout"
                    " するので、現状は 6 か月までで分割実行してください。"
                ),
            )

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
