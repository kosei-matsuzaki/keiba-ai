"""GET /api/simulation/active_model — backtest using active model.

Used by the Ledger 「シミュレーション」 tab.
"""

from __future__ import annotations

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
from keiba_ai.api.deps import get_session
from keiba_ai.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


class GroupStatsResponse(BaseModel):
    label: str
    n_bets: int
    invested: int
    payout: int
    payback_rate: float
    hit_rate: float


class SimulationWindow(BaseModel):
    start: str | None
    end: str | None


class SimulationResponse(BaseModel):
    window: SimulationWindow
    model_path: str
    strategy: str
    budget: int
    n_races: int
    n_settled_races: int
    summary: GroupStatsResponse
    by_bet_type: list[GroupStatsResponse]
    by_race_class: list[GroupStatsResponse]
    by_course: list[GroupStatsResponse]


def _result_to_response(r: SimulationResult) -> SimulationResponse:
    """Convert SimulationResult dataclass to pydantic response model."""
    d = r.as_dict()
    return SimulationResponse(
        window=SimulationWindow(**d["window"]),
        model_path=d["model_path"],
        strategy=d["strategy"],
        budget=d["budget"],
        n_races=d["n_races"],
        n_settled_races=d["n_settled_races"],
        summary=GroupStatsResponse(**d["summary"]),
        by_bet_type=[GroupStatsResponse(**g) for g in d["by_bet_type"]],
        by_race_class=[GroupStatsResponse(**g) for g in d["by_race_class"]],
        by_course=[GroupStatsResponse(**g) for g in d["by_course"]],
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
        Query(ge=1000, le=100_000_000, description="期間全体の予算 (円)"),
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

    return _result_to_response(result)
