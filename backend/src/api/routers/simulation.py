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

from ai.model.registry import _resolve_model_path
from ai.simulation.engine import (
    STRATEGY_PRESETS,
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
from core.settings_store import SettingsStore, resolve_model_path
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
    # バックテストに使ったモデル (model_runs.id)。
    model_run_id: int | None = None
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
    #: この run が**どの条件で走ったか** (確率モデルの有無・複勝の確信度の
    #: しきい値・履歴の無いレースの除外・券種・1 点あたりの金額など)。
    #: 0013 より前に保存された run では None = 「条件の記録なし」。
    conditions: dict | None = None
    #: 資金不足で 1 点も買えなかったレース数。0 でなければ、その run の回収率は
    #: 「破産するまでの期間」しか測っていない。
    n_races_broke: int = 0
    #: 期間中の資産の最小値。定額ではマイナスになりうる。
    trough_bankroll: int = 0
    #: この戦略を最後まで回すのに必要だった資金。定額で「いくら用意すれば
    #: 途中で止まらずに済んだか」を表す。
    required_capital: int = 0
    # 実行直後にバックエンドが保存した row の id。再呼び出しで詳細を取得可能。
    run_id: int | None = None


class SimulationRunSummary(BaseModel):
    """保存済み実行の一覧表示用 (重い json は含めない)。"""
    id: int
    created_at: str
    model_run_id: int | None
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
        conditions=d.get("conditions") or None,
        n_races_broke=int(d.get("n_races_broke") or 0),
        trough_bankroll=int(d.get("trough_bankroll") or 0),
        required_capital=int(d.get("required_capital") or 0),
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
        model_run_id=row.model_run_id,
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
        conditions=json.loads(row.conditions_json) if row.conditions_json else None,
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
        model_run_id=row.model_run_id,
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
    settings_store: Annotated[SettingsStore, Depends(get_settings_store)],
    start: Annotated[str | None, Query(description="窓の開始日 YYYY-MM-DD")] = None,
    end: Annotated[str | None, Query(description="窓の終了日 YYYY-MM-DD")] = None,
    budget: Annotated[
        int,
        Query(
            ge=1000,
            le=100_000_000,
            description="初期資産 (円)。各 race ごとに残資産 (= budget + 累計 profit) を "
            "その race の予算として 1 点定額で賭ける (compounding wealth)。"
            "payout は次 race の bet 余力に加算される。資産尽きたら以降は実質 bet しない。",
        ),
    ] = 100_000,
    strategy: Annotated[
        Literal["conservative", "balanced", "aggressive"],
        Query(description="戦略プリセット"),
    ] = "balanced",
    exclude_low_information: Annotated[
        bool,
        Query(
            description="履歴の無いレース (新馬戦など) を除外する。出走馬全員が初出走だと"
            "モデルの履歴特徴が全滅し、枠順・馬体重・騎手・血統・オッズだけの予想になるため、"
            "同じモデルでも入力の質が別物になる。",
        ),
    ] = False,
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
    staking: Annotated[
        Literal["flat", "compound"],
        Query(
            description="賭け金の決め方。flat=1 レースの予算を固定 (既定)。"
            "compound=残資産の一定割合。compound は払戻 1.0 未満の券種を数百レース"
            "買うと破産し、以降を実質評価しなくなるため、回収率の測定には flat を推奨。"
        ),
    ] = "flat",
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
    _validate_request(start, end, strategy)

    model_path, model_run_id = _resolve_target_model(session, model_id)

    # Settings の馬券種ターゲットを simulation でも反映する
    settings = settings_store.load()
    enabled_bet_types = supported_bet_types(settings.get("enabled_bet_types"))
    # 券種ごとの 1 点あたり金額。推奨 API (AI 予想) と同じ配分でシミュレーションする
    # ため、Settings の stake_units をそのまま渡す (docs/ai-model.md「賭け金の配分」)。
    stake_units = {k: int(v) for k, v in (settings.get("stake_units") or {}).items()} or None
    # 複勝の確信度フィルタも推奨 API と同じ設定を使う (アプリと数字を揃えるため)
    prob_model_path = resolve_model_path(settings.get("probability_model_path"))
    place_min_confidence = float(settings.get("place_min_hit_prob", 0.60))

    logger.info(
        "Simulation request: model_run_id=%d, window=%s..%s, budget=%d, "
        "strategy=%s, enabled_bet_types=%s",
        model_run_id, start, end, budget, strategy, enabled_bet_types,
    )

    result = simulate_active_model(
        session=session,
        model_path=model_path,
        start=start,
        end=end,
        budget=budget,
        strategy=strategy,  # type: ignore[arg-type]
        max_stake_per_race_yen=max_stake_per_race_yen,
        stake_unit_by_bet_type=stake_units,
            probability_model_path=prob_model_path,
            place_min_confidence=place_min_confidence,
            staking=staking,
        exclude_low_information=exclude_low_information,
        enabled_bet_types=enabled_bet_types,
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
    settings_store: Annotated[SettingsStore, Depends(get_settings_store)],
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
    exclude_low_information: Annotated[
        bool,
        Query(
            description="履歴の無いレース (新馬戦など) を除外する。出走馬全員が初出走だと"
            "モデルの履歴特徴が全滅し、枠順・馬体重・騎手・血統・オッズだけの予想になるため、"
            "同じモデルでも入力の質が別物になる。",
        ),
    ] = False,
    max_stake_per_race_yen: Annotated[
        int | None,
        Query(
            ge=0, le=10_000_000,
            description="1 race の累計 stake 絶対上限 (円)。0 / 未指定で無効。",
        ),
    ] = None,
    staking: Annotated[
        Literal["flat", "compound"],
        Query(
            description="賭け金の決め方。flat=1 レースの予算を固定 (既定)。"
            "compound=残資産の一定割合。compound は払戻 1.0 未満の券種を数百レース"
            "買うと破産し、以降を実質評価しなくなるため、回収率の測定には flat を推奨。"
        ),
    ] = "flat",
    model_id: Annotated[
        int | None,
        Query(description="対象モデル (model_runs.id)。未指定で active モデル。"),
    ] = None,
    bet_types: Annotated[
        str | None,
        Query(description="このシミュレーションだけの対象券種 (カンマ区切り)。未指定なら設定値"),
    ] = None,
    top_n_horses: Annotated[
        int | None,
        Query(ge=1, le=18, description="連系を組む上位頭数。未指定なら戦略プリセット"),
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
    # 対象モデルは submit 時点で確定する (job 実行時に active が変わっても、
    # 投入時に選んだモデルでバックテストする)。
    model_path, model_run_id = _resolve_target_model(session, model_id)

    settings = settings_store.load()
    # 券種は RACE 画面と同じく「この実行だけ」上書きできる。未指定なら設定値。
    enabled_bet_types = supported_bet_types(
        [b.strip() for b in bet_types.split(",") if b.strip()]
        if bet_types
        else settings.get("enabled_bet_types")
    )
    # 券種ごとの 1 点あたり金額。推奨 API (AI 予想) と同じ配分でシミュレーションする
    # ため、Settings の stake_units をそのまま渡す (docs/ai-model.md「賭け金の配分」)。
    stake_units = {k: int(v) for k, v in (settings.get("stake_units") or {}).items()} or None
    # 複勝の確信度フィルタも推奨 API と同じ設定を使う (アプリと数字を揃えるため)
    prob_model_path = resolve_model_path(settings.get("probability_model_path"))
    place_min_confidence = float(settings.get("place_min_hit_prob", 0.60))

    logger.info(
        "Simulation job submit: model_run_id=%d, window=%s..%s, budget=%d, "
        "strategy=%s, enabled_bet_types=%s",
        model_run_id, start, end, budget, strategy, enabled_bet_types,
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
                budget=budget,
                strategy=strategy,  # type: ignore[arg-type]
                max_stake_per_race_yen=max_stake_per_race_yen,
                stake_unit_by_bet_type=stake_units,
                **({"top_n_horses": top_n_horses} if top_n_horses else {}),
            probability_model_path=prob_model_path,
            place_min_confidence=place_min_confidence,
            staking=staking,
                exclude_low_information=exclude_low_information,
                enabled_bet_types=captured_bet_types,
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
