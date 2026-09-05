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
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from ai.model.registry import _resolve_model_path
from ai.simulation.engine import (
    SimulationResult,
    simulate_active_model,
)
from ai.simulation.persistence import (
    delete_simulation_run,
    get_simulation_run,
    list_simulation_runs,
    save_simulation_result,
)
from api.deps import (
    get_engine,
    get_job_registry,
    get_session,
    get_settings_store,
)
from api.jobs import JobRegistry
from api.schemas import JobAccepted
from core.bet_types import supported_bet_types
from core.logging import get_logger
from core.settings_store import SettingsStore, resolve_betting_settings
from db.models.model_run import ModelRun
from db.models.simulation_run import SimulationRun
from db.session import session_scope

logger = get_logger(__name__)

router = APIRouter()

# 1 年 ≒ 3000 race で逐次 predict + settle すると 5 分以上かかり frontend HTTP
# timeout に当たる。実用上は 6 か月 (~1500 race) が上限の目安。
MAX_WINDOW_DAYS: int = 186


def _resolve_target_model(
    session: Session, model_id: int | None
) -> tuple[Path, int]:
    """シミュレーション対象モデルを解決し (resolved_path, model_run_id) を返す。

    model_id 指定時はその ModelRun を引く (404 if not found)。未指定時は active
    モデルにフォールバック (503 if none)。シミュレーションは必ずどれか 1 モデルに
    紐づくため、いずれの経路でも model_run_id を確定して返す。
    """
    if model_id is not None:
        run = session.get(ModelRun, model_id)
        if run is None:
            raise HTTPException(
                status_code=404, detail=f"model id={model_id} が見つかりません",
            )
        return _resolve_model_path(run.model_path), run.id

    active = session.query(ModelRun).filter(ModelRun.is_active == 1).first()
    if active is None:
        raise HTTPException(
            status_code=503,
            detail="アクティブなモデルがありません。Models 画面でモデルを active 化するか、model_id を指定してください。",
        )
    return _resolve_model_path(active.model_path), active.id


class GroupStatsResponse(BaseModel):
    label: str
    n_bets: int
    invested: int
    payout: int
    payback_rate: float
    hit_rate: float


class ProfitPointResponse(BaseModel):
    """日次の損益推移ポイント (グラフ表示用)。0 から始まる累計損益。"""
    date: str          # YYYY-MM-DD
    profit: int        # その日の最終 race 後の累計損益 (マイナスもある)
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
    # バックテストに使ったモデル (model_runs.id)。
    model_run_id: int | None = None
    #: 1 レースに使ってよい上限 (円)。使い切る目標ではない。
    race_budget: int
    n_races: int
    n_settled_races: int
    #: 0 から始まる累計損益。マイナスならその期間はトータル負け。
    final_profit: int
    peak_profit: int
    summary: GroupStatsResponse
    by_bet_type: list[GroupStatsResponse]
    by_race_class: list[GroupStatsResponse]
    by_course: list[GroupStatsResponse]
    profit_timeseries: list[ProfitPointResponse]
    #: この run が**どの条件で走ったか** (確率モデルの有無・複勝の確信度の
    #: しきい値・券種・1 点あたりの金額・連系の下限など)。
    #: 0013 より前に保存された run では None = 「条件の記録なし」。
    conditions: dict | None = None
    #: 期間中の累計損益の最小値 (マイナスになりうる)。
    trough_profit: int = 0
    #: 途中で止まらずに回すのに必要だった資金 (= −trough_profit、下限 0)。
    required_capital: int = 0
    # 実行直後にバックエンドが保存した row の id。再呼び出しで詳細を取得可能。
    run_id: int | None = None


class SimulationRunSummary(BaseModel):
    """保存済み実行の一覧表示用 (重い json は含めない)。"""
    id: int
    created_at: str
    model_run_id: int | None
    race_budget: int
    window_start: str | None
    window_end: str | None
    n_races: int
    n_settled_races: int
    final_profit: int
    peak_profit: int


class SimulationRunListResponse(BaseModel):
    runs: list[SimulationRunSummary]
    total: int


def _result_to_response(
    r: SimulationResult,
    run_id: int | None = None,
    model_run_id: int | None = None,
) -> SimulationResponse:
    """Convert SimulationResult dataclass to pydantic response model."""
    d = r.as_dict()
    return SimulationResponse(
        window=SimulationWindow(**d["window"]),
        model_path=d["model_path"],
        model_run_id=model_run_id,
        race_budget=d["race_budget"],
        n_races=d["n_races"],
        n_settled_races=d["n_settled_races"],
        final_profit=d["final_profit"],
        peak_profit=d["peak_profit"],
        summary=GroupStatsResponse(**d["summary"]),
        by_bet_type=[GroupStatsResponse(**g) for g in d["by_bet_type"]],
        by_race_class=[GroupStatsResponse(**g) for g in d["by_race_class"]],
        by_course=[GroupStatsResponse(**g) for g in d["by_course"]],
        conditions=d.get("conditions") or None,
        trough_profit=int(d.get("trough_profit") or 0),
        required_capital=int(d.get("required_capital") or 0),
        profit_timeseries=[
            ProfitPointResponse(**p) for p in d["profit_timeseries"]
        ],
        run_id=run_id,
    )


def _row_to_response(row: SimulationRun) -> SimulationResponse:
    """保存済み SimulationRun row → SimulationResponse (json を decode)."""
    return SimulationResponse(
        window=SimulationWindow(start=row.window_start, end=row.window_end),
        model_path=row.model_path,
        model_run_id=row.model_run_id,
        race_budget=row.race_budget,
        n_races=row.n_races,
        n_settled_races=row.n_settled_races,
        final_profit=row.final_profit,
        peak_profit=row.peak_profit,
        summary=GroupStatsResponse(**json.loads(row.summary_json)),
        by_bet_type=[
            GroupStatsResponse(**g) for g in json.loads(row.by_bet_type_json)
        ],
        by_race_class=[
            GroupStatsResponse(**g) for g in json.loads(row.by_race_class_json)
        ],
        by_course=[GroupStatsResponse(**g) for g in json.loads(row.by_course_json)],
        conditions=json.loads(row.conditions_json) if row.conditions_json else None,
        profit_timeseries=[
            ProfitPointResponse(**p)
            for p in json.loads(row.profit_timeseries_json)
        ],
        run_id=row.id,
    )


def _row_to_summary(row: SimulationRun) -> SimulationRunSummary:
    return SimulationRunSummary(
        id=row.id,
        created_at=row.created_at,
        model_run_id=row.model_run_id,
        race_budget=row.race_budget,
        window_start=row.window_start,
        window_end=row.window_end,
        n_races=row.n_races,
        n_settled_races=row.n_settled_races,
        final_profit=row.final_profit,
        peak_profit=row.peak_profit,
    )


def _validate_request(
    start: str | None,
    end: str | None,
    max_days: int = MAX_WINDOW_DAYS,
    too_long_hint: str = (
        " 1 年規模だと逐次 predict + settle が数分かかります。"
        " 6 か月以内で分割実行するか、それを超える window が必要なら"
        " バックグラウンドジョブ (POST /api/simulation/start) を使ってください。"
    ),
) -> None:
    """期間の妥当性を確認。違反は HTTPException(400) を投げる。

    sync run と background job で共通。**違いは上限日数と、超えたときの案内文
    だけ**なので引数にした (以前は 2 本に分かれていて、docstring だけが「共通
    利用」と言っていた)。
    """
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
        if (d_end - d_start).days > max_days:
            raise HTTPException(
                status_code=400,
                detail=f"期間が長すぎます (max {max_days} 日)。{too_long_hint}",
            )


@router.get(
    "/simulation/active_model",
    response_model=SimulationResponse,
)
def run_simulation(
    session: Annotated[Session, Depends(get_session)],
    settings_store: Annotated[SettingsStore, Depends(get_settings_store)],
    start: Annotated[str | None, Query(description="窓の開始日 YYYY-MM-DD")] = None,
    end: Annotated[str | None, Query(description="窓の終了日 YYYY-MM-DD")] = None,
    race_budget: Annotated[
        int,
        Query(
            ge=100,
            le=1_000_000,
            description="1 レースに使ってよい上限 (円)。**使い切る目標ではない。**"
            "実際に賭ける額は複勝の確信度と連系の的中確率の下限が決めるので、"
            "レースごとに変わる。連系の点数の上限もこの額から決まる。",
        ),
    ] = 5_000,
    model_id: Annotated[
        int | None,
        Query(description="対象モデル (model_runs.id)。未指定で active モデル。"),
    ] = None,
) -> SimulationResponse:
    """Run end-to-end backtest on the given window.

    動作:
      1. 対象モデル (model_id 指定 or active) を load (binary + calibrator 含む)
      2. 期間内の全レースに対して predict + recommendation を生成
      3. 実 finish_position と payouts で settle
      4. bet_type / race_class / course でアグリゲート

    所要時間: 800 race で ~30-60 秒。レスポンスはキャッシュされない。
    """
    _validate_request(start, end)

    model_path, model_run_id = _resolve_target_model(session, model_id)

    settings = settings_store.load()
    # **券種は絞らない。** どの券種が効くかはレースによって違い、買うかどうかは
    # 確信度 (的中確率の下限) が決める。
    enabled_bet_types = None
    # 買い方は推奨 API と同じ設定から解決する (アプリと数字を揃えるため)。
    bet_settings = resolve_betting_settings(settings)

    logger.info(
        "Simulation request: model_run_id=%d, window=%s..%s, race_budget=%d, "
        "enabled_bet_types=%s",
        model_run_id, start, end, race_budget, enabled_bet_types,
    )

    result = simulate_active_model(
        session=session,
        model_path=model_path,
        start=start,
        end=end,
        race_budget=race_budget,
        probability_model_path=bet_settings.probability_model_path,
        place_min_confidence=bet_settings.place_min_confidence,
        min_hit_prob_by_bet_type=bet_settings.combo_min_hit_prob,
        enabled_bet_types=enabled_bet_types,
        win_min_odds=bet_settings.win_min_odds,
    )

    # 自動保存 (上限 50 件、超過したら古い順に削除)
    saved = save_simulation_result(session, result, model_run_id)
    logger.info("Simulation result saved as run id=%d", saved.id)

    return _result_to_response(result, run_id=saved.id, model_run_id=model_run_id)


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
    model_id: Annotated[
        int | None,
        Query(description="このモデル (model_runs.id) の実行のみに絞る。"),
    ] = None,
) -> SimulationRunListResponse:
    """保存済みシミュレーション実行の一覧を新しい順で返す (重い json は含まない)。

    model_id を指定するとそのモデルの実行のみ返す (モデル詳細画面用)。
    """
    runs = list_simulation_runs(session, limit=limit, model_run_id=model_id)
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


@router.post(
    "/simulation/start",
    response_model=JobAccepted,
)
async def start_simulation_job(
    session: Annotated[Session, Depends(get_session)],
    registry: Annotated[JobRegistry, Depends(get_job_registry)],
    engine: Annotated[Engine, Depends(get_engine)],
    settings_store: Annotated[SettingsStore, Depends(get_settings_store)],
    start: Annotated[str | None, Query(description="YYYY-MM-DD")] = None,
    end: Annotated[str | None, Query(description="YYYY-MM-DD")] = None,
    race_budget: Annotated[
        int,
        Query(
            ge=100,
            le=1_000_000,
            description="1 レースに使ってよい上限 (円)。使い切る目標ではない。",
        ),
    ] = 5_000,
    model_id: Annotated[
        int | None,
        Query(description="対象モデル (model_runs.id)。未指定で active モデル。"),
    ] = None,
    bet_types: Annotated[
        str | None,
        Query(description="このシミュレーションだけの対象券種 (カンマ区切り)。未指定なら設定値"),
    ] = None,
) -> JobAccepted:
    """シミュレーションをバックグラウンド job として実行する。

    HTTP timeout を気にせず長い window (最大 1 年) を扱える。
    完了後 job.result.run_id に保存済み run の id が入るので、UI は
    /api/simulation/runs/{run_id} で詳細を取得すれば良い。

    NOTE: async def で宣言する必要がある (registry.start 内部で
    asyncio.create_task を呼ぶため、event loop 上で動かす必要がある)。
    """
    _validate_request(start, end, max_days=MAX_BG_WINDOW_DAYS, too_long_hint="")
    # 対象モデルは submit 時点で確定する (job 実行時に active が変わっても、
    # 投入時に選んだモデルでバックテストする)。
    model_path, model_run_id = _resolve_target_model(session, model_id)

    settings = settings_store.load()
    # 券種は「この実行だけ」上書きできる。未指定なら全券種。
    enabled_bet_types = (
        supported_bet_types([b.strip() for b in bet_types.split(",") if b.strip()])
        if bet_types
        else None
    )
    # 買い方は推奨 API と同じ設定から解決する (アプリと数字を揃えるため)。
    bet_settings = resolve_betting_settings(settings)

    logger.info(
        "Simulation job submit: model_run_id=%d, window=%s..%s, race_budget=%d, "
        "enabled_bet_types=%s",
        model_run_id, start, end, race_budget, enabled_bet_types,
    )

    # asyncio.create_task の中で session を作るため、Engine だけを capture。
    # request 由来の session を job loop 内で使うと scope が合わない。
    captured_engine = engine
    captured_path = model_path
    captured_model_run_id = model_run_id
    captured_bet_types = enabled_bet_types

    def _run_simulation_blocking() -> int:
        """Worker thread: open new session + run + save。Returns saved run id."""
        with session_scope(captured_engine) as bg_session:
            result = simulate_active_model(
                session=bg_session,
                model_path=captured_path,
                start=start,
                end=end,
                race_budget=race_budget,
                probability_model_path=bet_settings.probability_model_path,
                place_min_confidence=bet_settings.place_min_confidence,
                min_hit_prob_by_bet_type=bet_settings.combo_min_hit_prob,
                enabled_bet_types=captured_bet_types,
                win_min_odds=bet_settings.win_min_odds,
            )
            saved = save_simulation_result(
                bg_session, result, captured_model_run_id
            )
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
