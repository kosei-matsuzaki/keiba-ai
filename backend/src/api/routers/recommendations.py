"""GET /api/recommendations/{race_id} — recommended bet candidates with flat stakes."""

from __future__ import annotations

import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ai.betting.odds import compute_race_odds_with_sources
from ai.betting.strategy import STAKE_UNIT, TOP_N_HORSES, recommend_for_race
from ai.inference.confidence import (
    is_place_worth_buying,
    pick_confidence,
    points_for_confidence,
)
from ai.inference.predict import (
    _combinations_from_base,
    _predict_race_nn,
    merge_combination_sources,
    predict_race,
    predict_race_with_combinations,
)
from ai.model.registry import get_active, load_model_full
from api.deps import (
    build_inference_frame_or_404,
    get_odds_session,
    get_session,
    get_settings_store,
)
from core.bet_types import COMBINATION_BET_TYPES, DEFAULT_COMBO_MIN_HIT_PROB
from core.logging import get_logger
from core.settings_store import SettingsStore, resolve_model_path

logger = get_logger(__name__)

router = APIRouter()


class RecommendationCandidate(BaseModel):
    bet_type: str
    combo: str
    pattern: str
    prob: float
    #: **確信度 = 確率モデルから見た「この買い目が当たる確率」。**
    #:
    #: 券種をまたいで同じ意味にしてある: 単勝なら 1 着になる確率、複勝なら
    #: 3 着以内に入る確率、連系なら組合せの的中確率。`prob` は単複だと active
    #: (買う馬を決めるモデル) の確率で、確率としての精度は保証されない
    #: (本命の win_prob と勝敗の相関は 0.073)。判断に使うのはこちら。
    #: 確率モデル未設定なら None。
    confidence: float | None = None
    est_odds: float | None
    est_odds_source: Literal["confirmed", "scraped", "implied", "unknown"] = "unknown"
    ev: float | None
    stake: int
    post_positions: list[int]


class RecommendationsResponse(BaseModel):
    race_id: str
    race_budget: int
    #: 1 点あたりの金額 (円)。**固定値**。賭け金は必ずこの倍数で、
    #: `stake / stake_unit` がその買い目の点数になる。
    stake_unit: int = STAKE_UNIT
    candidates: list[RecommendationCandidate]
    odds_source: Literal["live", "past", "unknown"] = "unknown"
    #: 確率モデルが AI の本命に与えた単勝確率。確率モデル未設定なら None。
    place_confidence: float | None = None
    #: 複勝を買う確信度のしきい値 (place_confidence がこれ未満なら複勝は見送る)。
    place_confidence_threshold: float | None = None


def _resolve_odds_source(
    session: Session,
    odds_session: Session | None,
    race_id: str,
) -> tuple[
    dict[str, dict[str, float]] | None,
    dict[str, dict[str, str]] | None,
    Literal["live", "past", "unknown"],
]:
    """Determine odds + per-combo source labels.

    High-level label は日付ベース: 過去レース → "past"、当日レース → "live"、
    オッズ皆無 → "unknown"。当日レースは payouts 確定前のため単勝(entries.odds_win)
    由来の市場オッズが中心だが、UI 上は "live"(=現時点の市場オッズ)扱いとする。
    Per-combo source label は "confirmed" / "implied" (compute_race_odds_with_sources
    が付与)。取得できない combo は dict から欠落する。

    Returns:
        (race_odds, sources, odds_source_label).
        race_odds / sources are None when no data is available at all.
    """
    odds, sources = compute_race_odds_with_sources(
        session, race_id, odds_session=odds_session
    )
    if not odds:
        return None, None, "unknown"

    # 過去レース判定は date < today。当日レースは市場(単勝)オッズ由来でも "live" 扱い。
    from sqlalchemy import select as sa_select

    from db.models.race import Race as RaceModel

    race_row = session.execute(
        sa_select(RaceModel.date).where(RaceModel.race_id == race_id)
    ).first()

    today_str = datetime.date.today().isoformat()
    is_past = race_row is not None and race_row.date < today_str

    label: Literal["live", "past", "unknown"] = "past" if is_past else "live"
    return odds, sources, label


@router.get("/recommendations/{race_id}", response_model=RecommendationsResponse)
def get_recommendations(
    race_id: str,
    session: Annotated[Session, Depends(get_session)],
    odds_session: Annotated[Session, Depends(get_odds_session)],
    store: Annotated[SettingsStore, Depends(get_settings_store)],
    top_k: Annotated[int, Query(ge=1, le=200, description="Combination upper limit per bet type (1-200)")] = 50,
    # ── このレースだけ Settings を上書きするための任意パラメータ ──
    # 未指定なら Settings の値を使う。全レース共通の既定値は Settings 側に置き、
    # 「このレースだけ予算を絞る / 券種を単複に限る」といった判断をここで通す。
    race_budget: Annotated[
        int | None,
        Query(ge=100, description="このレースに使う上限 (円)。未指定なら設定値"),
    ] = None,
    bet_types: Annotated[
        str | None,
        Query(description="このレースだけの対象券種 (カンマ区切り。未指定なら設定値)"),
    ] = None,
) -> RecommendationsResponse:
    """Return recommended bet candidates for a race.

    Flow:
    1. Resolve active model (503 if none).
    2. Build inference frame for race_id (404 if not found or empty).
    3. Run predict_race to get win_prob / place_prob per horse.
    4. Resolve race odds: live → past → unknown.
    5. Run predict_race_with_combinations for combination EVs.
    6. Load Settings (race_budget, stake_unit, etc.) and call recommend_for_race.
    7. Return RecommendationsResponse.
    """
    active_path = get_active(session)
    if active_path is None:
        raise HTTPException(
            status_code=503,
            detail="No active model. Train and activate a model first.",
        )

    frame = build_inference_frame_or_404(session, race_id)

    bundle = load_model_full(active_path)

    # Step 3: win_prob / place_prob per horse
    # bundle 経由で推論 (temperature scaler 等は内部で適用)
    predictions = predict_race(bundle, frame, session=session)

    # Join post_position from frame so recommend_for_race can build top_pps.
    # predict_race returns horse_id-indexed rows without post_position.
    pp_map = dict(zip(frame["horse_id"].values, frame["post_position"].values, strict=True))
    predictions["post_position"] = predictions["horse_id"].map(pp_map)

    # Step 4: resolve confirmed + scraped(実オッズ) + implied odds + per-combo source
    race_odds, race_odds_sources, odds_source = _resolve_odds_source(
        session, odds_session, race_id
    )
    if odds_source == "unknown":
        logger.warning(
            "No confirmed odds available for race %s — est_odds will be null", race_id
        )

    # Step 5: combination probabilities (capped by top_k for performance)
    # 確率専用モデルが設定されていれば、**連系の確率はそちらから出す**。
    # 連系確率はスコアから解析的 PL で導出するので、スコアが PL の強度パラメータで
    # あることを前提にしている。active は回収率で学習しており その保証が無い。
    # 買う馬・買い目の脚は active のまま (predictions がそれを決める)。
    settings_early = store.load()
    prob_model_path = resolve_model_path(settings_early.get("probability_model_path"))
    prob_bundle = None
    if prob_model_path is not None:
        try:
            prob_bundle = load_model_full(prob_model_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("probability model load failed (%s): %s", prob_model_path, exc)
    combinations_by_type = predict_race_with_combinations(
        bundle,
        frame,
        session=session,
        top_k_combinations=top_k,
        race_odds=race_odds,
        race_odds_sources=race_odds_sources,
    )
    if prob_bundle is not None:
        # 連系だけ確率モデル由来に差し替える (単勝・複勝の候補は active のまま)
        combinations_by_type = merge_combination_sources(
            combinations_by_type,
            _combinations_from_base(
                base_df=_predict_race_nn(prob_bundle, frame, session=session),
                frame=frame,
                n_samples=10_000,
                rng=None,
                top_k_combinations=top_k,
                race_odds=race_odds,
                race_odds_sources=race_odds_sources,
            ),
        )

    # Step 6: load settings and run recommendation logic
    # クエリで渡された分だけ、このレースに限って設定を上書きする。
    settings = store.load()
    # 連系の点数は**的中確率の下限だけ**で決まる。線を超えた買い目を全部買うので、
    # 確信度の高いレースほど点数が増え、低いレースでは 0 点になる。
    # 券種ごとの点数上限は持たない (ワイドが効くレース・三連単が効くレースを
    # 一律の点数で潰さないため)。
    _floors = settings.get("combo_min_hit_prob")
    eff_combo_floors: dict[str, float] = (
        {k: float(v) for k, v in _floors.items()}
        if isinstance(_floors, dict)
        else dict(DEFAULT_COMBO_MIN_HIT_PROB)
    )

    eff_budget: int = (
        race_budget if race_budget is not None else int(settings.get("race_budget", 5_000))
    )
    # 単勝のオッズ下限
    eff_win_min_odds: float = float(settings.get("win_min_odds", 1.1))
    if bet_types is not None:
        requested = [t.strip() for t in bet_types.split(",") if t.strip()]
        unknown = [t for t in requested if t not in COMBINATION_BET_TYPES]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown bet_types: {', '.join(unknown)}",
            )
        if not requested:
            raise HTTPException(status_code=422, detail="bet_types must not be empty")
        eff_bet_types = requested
    else:
        # **券種を設定で絞らない。** どの券種が効くかはレースによって違い、
        # 買うかどうかは確信度 (的中確率の下限) が決める。
        eff_bet_types = list(COMBINATION_BET_TYPES)

    # 複勝の確信度フィルタ。確率専用モデル (proper scoring rule で学習) が設定されて
    # いれば、AI の本命に対するその確率がしきい値未満のレースでは複勝を見送る。
    # 買う馬は変えない — 確率モデルに馬を選ばせると人気馬に寄って回収率が落ちる。
    # 実測 (前進検証 4.5 年): しきい値 0.30 で複勝回収率 0.866 → 0.907
    # (ai/inference/confidence.py に根拠)。未設定なら何もしない。
    confidence: float | None = None
    win_confidence: float | None = None
    conf_threshold: float | None = None
    if prob_bundle is not None and not predictions.empty:
        # 単勝の確信度 = 確率モデルの 1 着確率。**買う/買わないには使わない**
        # (実測で回収率が反応しない) が、画面には同じ意味の数字として出す。
        win_confidence = pick_confidence(
            prob_bundle, frame, predictions.iloc[0]["horse_id"],
            session=session, bet_type="単勝",
        )
    if prob_bundle is not None and "複勝" in eff_bet_types and not predictions.empty:
        confidence = pick_confidence(
            prob_bundle, frame, predictions.iloc[0]["horse_id"], session=session
        )
        conf_threshold = float(settings.get("place_min_hit_prob", 0.60))
        if not is_place_worth_buying(confidence, conf_threshold):
            eff_bet_types = [b for b in eff_bet_types if b != "複勝"]
            logger.info(
                "race %s: 複勝 skipped (confidence %.3f < %.2f)",
                race_id, confidence, conf_threshold,
            )
    # **点数は確信度から決める。1 点 = 100 円。**
    # 連系は 1 組合せ = 1 点で、何点買うかは上の下限が決める。
    eff_points: dict[str, int] = {
        "単勝": points_for_confidence("単勝", win_confidence),
        "複勝": points_for_confidence("複勝", confidence),
    }

    result = recommend_for_race(
        predictions=predictions,
        combinations_by_type=combinations_by_type,
        race_id=race_id,
        race_budget=eff_budget,
        points_by_bet_type=eff_points,
        win_min_odds=eff_win_min_odds,
        top_n_horses=TOP_N_HORSES,
        enabled_bet_types=eff_bet_types,
        min_hit_prob_by_bet_type=eff_combo_floors,
    )

    def _confidence_for(c) -> float | None:  # noqa: ANN001
        """券種横断の確信度。

        連系の `prob` はもともと確率モデル由来 (`merge_combination_sources`) なので
        そのまま使える。単複は active の確率なので、確率モデルに引き直した値を返す。
        """
        if c.bet_type == "単勝":
            return win_confidence
        if c.bet_type == "複勝":
            return confidence
        return c.prob

    candidates = [
        RecommendationCandidate(
            bet_type=c.bet_type,
            combo=c.combo,
            pattern=c.pattern,
            prob=c.prob,
            confidence=_confidence_for(c),
            est_odds=c.est_odds,
            est_odds_source=c.est_odds_source,
            ev=c.ev,
            stake=c.stake,
            post_positions=list(c.post_positions),
        )
        for c in result.candidates
    ]

    return RecommendationsResponse(
        race_id=result.race_id,
        race_budget=result.race_budget,
        stake_unit=STAKE_UNIT,
        candidates=candidates,
        odds_source=odds_source,
        place_confidence=confidence,
        place_confidence_threshold=conf_threshold,
    )
