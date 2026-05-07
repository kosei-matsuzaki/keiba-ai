"""GET /api/simulation/active_model — backtest using active model.

Used by the Ledger 「シミュレーション」 tab.
"""

from __future__ import annotations

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
from keiba_ai.api.deps import get_session
from keiba_ai.core.logging import get_logger

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
        Query(
            ge=1000,
            le=100_000_000,
            description="Kelly 計算用の元手 (円)。1 race ごとの stake 上限は "
            "元手 × max_stake_per_race_pct (= 5%) で決まる。"
            "累計支出のキャップではないため、race 数が増えると累計 invested は増える。",
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

    return _result_to_response(result)
