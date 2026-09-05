"""End-to-end backtest simulation for the active model.

Loops over all races in a given window, runs predict + recommendations
+ settle (using actual finish_position and confirmed payouts) and
aggregates ROI / hit-rate by bet_type / race_class / course.

This is the engine behind the Ledger 「シミュレーション」 tab.

買い方と入力は `simulate_active_model` の docstring。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session

from ai.betting.odds import (
    compute_past_race_odds,
    compute_race_odds_with_sources,
)
from ai.betting.strategy import (
    DEFAULT_RACE_BUDGET,
    TOP_N_HORSES,
    recommend_for_race,
)
from ai.inference.confidence import (
    is_place_worth_buying,
    pick_confidence,
    points_for_confidence,
)
from ai.inference.predict import (
    DEFAULT_TOP_K_COMBINATIONS,
    _combinations_from_base,
    _predict_race_nn,
    merge_combination_sources,
    predict_race,
    predict_race_with_combinations,
)
from ai.model.registry import ModelBundle, load_model_full
from core.bet_types import COMBINATION_BET_TYPES, DEFAULT_COMBO_MIN_HIT_PROB
from core.logging import get_logger
from db.odds_db import init_odds_db, make_odds_engine
from features.builder import build_training_frame

log = get_logger(__name__)





# 単勝 / 複勝 / 連系 すべての券種を simulation 対象とする
DEFAULT_BET_TYPES: list[str] = list(COMBINATION_BET_TYPES)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GroupStats:
    """Aggregated stats for a single group (bet_type / race_class / course)."""

    label: str
    n_bets: int = 0
    invested: int = 0
    payout: float = 0.0
    hits: int = 0

    @property
    def payback_rate(self) -> float:
        """payout / invested. 0 when no bets."""
        return float(self.payout) / float(self.invested) if self.invested > 0 else 0.0

    @property
    def hit_rate(self) -> float:
        """hits / n_bets. 0 when no bets."""
        return self.hits / self.n_bets if self.n_bets > 0 else 0.0

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "n_bets": self.n_bets,
            "invested": self.invested,
            "payout": round(self.payout),
            "payback_rate": round(self.payback_rate, 4),
            "hit_rate": round(self.hit_rate, 4),
        }


@dataclass
class ProfitPoint:
    """損益推移グラフ用の 1 日分のスナップショット。

    **0 から始まる累計損益**を持つ。元手の額はシミュレーションに要らない
    (賭け金は 1 レースの予算と確信度だけで決まり、資金繰りに依存しない) ので、
    見たいのは「プラスかマイナスか」だけである。
    """

    date: str           # YYYY-MM-DD
    profit: int         # その日の最終 race 後の累計損益 (0 スタート)
    invested: int       # その日の累計 stake
    payout: float       # その日の累計 payout
    n_bets: int         # その日の bet 件数


@dataclass
class SimulationResult:
    """Top-level simulation result.

    n_races: total races within window (including ones where no bets fired)
    n_settled_races: subset where finish_position was available (i.e. past)
    final_profit: 期間終了時の累計損益 (0 スタート。マイナスもそのまま持つ)
    peak_profit / trough_profit: 期間中の最大・最小の累計損益
    profit_timeseries: 日次の損益推移 (グラフ用)
    """

    window_start: str | None
    window_end: str | None
    model_path: str
    race_budget: int
    n_races: int = 0
    n_settled_races: int = 0
    final_profit: int = 0
    peak_profit: int = 0
    summary: GroupStats = field(default_factory=lambda: GroupStats(label="all"))
    by_bet_type: list[GroupStats] = field(default_factory=list)
    by_race_class: list[GroupStats] = field(default_factory=list)
    by_course: list[GroupStats] = field(default_factory=list)
    profit_timeseries: list[ProfitPoint] = field(default_factory=list)
    #: この run が**どの条件で走ったか**。設定を変えて回し直したとき、過去の run が
    #: 何の条件だったか分からなくなるのを防ぐ (確率モデルの有無・確信度のしきい値・
    #: 履歴の無いレースの除外・券種・1 点あたりの金額は、いずれも結果を大きく変える)。
    conditions: dict = field(default_factory=dict)
    #: 期間中の累計損益の最小値 (マイナスになりうる)。
    trough_profit: int = 0
    #: 途中で止まらずに回すのに必要だった資金 (= −trough_profit、下限 0)。
    #: 賭け金は資金繰りに依存しないので破産は起きないが、「どれだけ沈む時期が
    #: あったか」は運用上の情報として残す。
    required_capital: int = 0

    def as_dict(self) -> dict:
        return {
            "window": {"start": self.window_start, "end": self.window_end},
            "model_path": self.model_path,
            "conditions": self.conditions,
            "trough_profit": self.trough_profit,
            "required_capital": self.required_capital,
            "race_budget": self.race_budget,
            "n_races": self.n_races,
            "n_settled_races": self.n_settled_races,
            "final_profit": self.final_profit,
            "peak_profit": self.peak_profit,
            "summary": self.summary.as_dict(),
            "by_bet_type": [g.as_dict() for g in self.by_bet_type],
            "by_race_class": [g.as_dict() for g in self.by_race_class],
            "by_course": [g.as_dict() for g in self.by_course],
            "profit_timeseries": [
                {
                    "date": p.date,
                    "profit": p.profit,
                    "invested": p.invested,
                    "payout": round(p.payout),
                    "n_bets": p.n_bets,
                }
                for p in self.profit_timeseries
            ],
        }


# ---------------------------------------------------------------------------
# Settlement helpers
# ---------------------------------------------------------------------------


# 連系 (馬連 / ワイド / 馬単 / 三連複 / 三連単) の miss を最大何件 log する か。
# KEIBA_DEBUG_SIM_MISSES=1 のときのみ有効。0% hit_rate の根本原因が
# combo 表記不一致なのか pure miss なのかを切り分けるための診断ログ。
_DEBUG_MISSES_LIMIT = 20
_debug_misses_emitted = 0


def _settle_candidates(
    candidates: list,
    race_id: str,
    finish_to_pp: dict[int, int],
    past_odds: dict[str, dict[str, float]],
) -> list[dict]:
    """Determine hit/miss + payout for each recommended candidate.

    Args:
        candidates: list[BetCandidate] from recommend_for_race (stake > 0).
        race_id: target race.
        finish_to_pp: {finish_position: post_position}. Only contains finished horses.
        past_odds: {bet_type: {combo: confirmed_odds_multiplier}} from
            compute_past_race_odds (winners only for 連系; all horses for 単勝/複勝).

    Returns:
        list[dict] with keys: bet_type, stake, payout, hit (0/1)
    """
    global _debug_misses_emitted
    debug_misses = os.environ.get("KEIBA_DEBUG_SIM_MISSES", "0") == "1"

    winner_pp = finish_to_pp.get(1)
    top3 = {finish_to_pp.get(p) for p in (1, 2, 3) if finish_to_pp.get(p) is not None}

    settlements: list[dict] = []
    tan_odds = past_odds.get("単勝", {})
    fuku_odds = past_odds.get("複勝", {})

    for cand in candidates:
        if cand.stake <= 0:
            continue

        hit = False
        payout = 0.0

        if cand.bet_type == "単勝":
            # combo は post_position 文字列。winner と一致したら hit
            if winner_pp is not None and cand.combo == str(winner_pp):
                hit = True
                odds = tan_odds.get(cand.combo, 0.0)
                payout = cand.stake * odds
        elif cand.bet_type == "複勝":
            # combo は post_position 文字列。top-3 にいたら hit
            try:
                pp = int(cand.combo)
            except (ValueError, TypeError):
                pp = None
            if pp is not None and pp in top3:
                hit = True
                odds = fuku_odds.get(cand.combo, 0.0)
                payout = cand.stake * odds
        else:
            # 連系: payouts dict に combo が登録されていれば hit
            confirmed = past_odds.get(cand.bet_type, {}).get(cand.combo)
            if confirmed is not None:
                hit = True
                payout = cand.stake * confirmed
            elif debug_misses and _debug_misses_emitted < _DEBUG_MISSES_LIMIT:
                # combo 表記の不一致 vs 純粋な miss を切り分ける診断 log。
                # past_odds[bet_type] の登録 combo を最大 3 件並べて、cand.combo
                # がそれと比較して妥当かどうかを目視できるようにする。
                _debug_misses_emitted += 1
                bet_keys = list(past_odds.get(cand.bet_type, {}).keys())
                top3_pps = [finish_to_pp.get(p) for p in (1, 2, 3)]
                log.info(
                    "[SIM_DEBUG_MISS] race=%s bet_type=%s cand.combo=%r "
                    "past_keys=%r (sample) top3_pps=%r",
                    race_id, cand.bet_type, cand.combo,
                    bet_keys[:3], top3_pps,
                )

        settlements.append({
            "bet_type": cand.bet_type,
            "stake": int(cand.stake),
            "payout": float(payout),
            "hit": 1 if hit else 0,
            "source": getattr(cand, "est_odds_source", "unknown"),
        })

    return settlements


# ---------------------------------------------------------------------------
# Main simulation entrypoint
# ---------------------------------------------------------------------------


def simulate_active_model(
    session: Session,
    model_path: Path,
    start: str | None,
    end: str | None,
    race_budget: int = DEFAULT_RACE_BUDGET,
    enabled_bet_types: list[str] | None = None,
    min_hit_prob_by_bet_type: dict[str, float] | None = None,
    probability_model_path: Path | None = None,
    place_min_confidence: float = 0.60,
    win_min_odds: float = 1.1,
    top_k_combinations: int | None = DEFAULT_TOP_K_COMBINATIONS,
    *,
    bundle: ModelBundle | None = None,
    bet_sink: list[dict] | None = None,
) -> SimulationResult:
    """**RACE 画面の予想を全レースでやったらどうなるか**を測る。

    買い方は推奨買目 API と同一 (`recommend_for_race`)。賭け金は資金繰りに
    依存させない:

      - 入力は **1 レースに使う上限** (`race_budget`) だけ。初期資産・賭け金の
        決め方・戦略プリセットは持たない
      - 予算は上限であって使い切る目標ではない。実際に賭ける額は複勝の確信度と
        連系の的中確率の下限が決める (レースごとに変わる)
      - 資産ではなく **0 から始まる累計損益**を返す

    以前は初期資産から複利で回していたが、払戻 1.0 未満の券種を数百レース買うと
    資産が尽きて賭け金が下限に張り付き、**以降を実質評価しなくなる**。回収率は
    Σpayout/Σstake = 賭け金の重み付き平均なので、破産すると「早い時期の大きい
    賭け金」に偏った数字になる。実際にこれで「連系は点数が少なく測定不能」と
    誤って結論していた (定額で測り直したら十分測れた)。

    Args:
        session: SQLAlchemy session bound to the keiba DB.
        model_path: Path to an NN model directory (model.pt + meta.json,
            optionally preprocessor.pkl / temperature_scaler.pkl).
        start / end: window date range (YYYY-MM-DD), inclusive. Both optional.
        race_budget: 1 レースに使ってよい上限 (円)。**使い切る目標ではない。**
        enabled_bet_types: 対象券種。None なら全種。
        min_hit_prob_by_bet_type: 連系を買う的中確率の下限 (券種ごと)。
            None なら既定値。**連系の点数はこれだけで決まる。**
        probability_model_path: 確率専用モデル (proper scoring rule で学習) の
            ディレクトリ。指定すると (a) 本命の 3 着内率が place_min_confidence
            未満のレースでは**複勝を買わない**、(b) 買うレースでは確信度に応じて
            複勝の点数を変える、(c) **連系の確率もそこから導出する**。
            買う馬は変えない (`ai/inference/confidence.py` に実測の根拠)。
        place_min_confidence: 上のしきい値 (既定 0.60 = 3 着内率)。
        win_min_odds: 単勝を買うオッズの下限。**呼び出し側が settings の値を
            渡すこと。** 既定のまま呼ぶと RACE 画面より低い線で買うので、
            シミュレーションだけが Settings に追従しない。
        top_k_combinations: 券種ごとに候補へ残す通り数 (EV 順に打ち切る)。
            RACE 画面と同じ値にしておく (`DEFAULT_TOP_K_COMBINATIONS`)。

    Returns:
        SimulationResult with summary, by_bet_type, by_race_class, by_course,
        final_profit, peak_profit, and profit_timeseries (日次推移).
    """
    eff_top_n = TOP_N_HORSES
    types = enabled_bet_types or DEFAULT_BET_TYPES

    # Allow callers (notably ad-hoc experiments) to pass a pre-built bundle
    # so they can attach ensemble fields or override calibrators without
    # writing the changes back to disk.  When omitted we load from disk.
    if bundle is None:
        log.info(
            "Loading active model bundle from %s (race_budget=%d)", model_path, race_budget
        )
        bundle = load_model_full(model_path)
    else:
        log.info(
            "Using pre-built bundle (model_dir=%s, race_budget=%d)",
            getattr(bundle, "model_dir", model_path), race_budget,
        )

    prob_bundle = None
    if probability_model_path is not None:
        try:
            prob_bundle = load_model_full(probability_model_path)
            log.info("Confidence model loaded from %s", probability_model_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("confidence model load failed (%s): %s", probability_model_path, exc)

    log.info("Building feature frame for window %s..%s", start, end)
    frame = build_training_frame(session, train_start=start, train_end=end)

    result = SimulationResult(
        window_start=start,
        window_end=end,
        model_path=str(model_path),
        race_budget=race_budget,
        conditions={
            "probability_model": (
                Path(probability_model_path).name if probability_model_path else None
            ),
            "place_min_confidence": (
                place_min_confidence if probability_model_path else None
            ),
            "enabled_bet_types": list(types),
            "race_budget": race_budget,
            "win_min_odds": win_min_odds,
            "top_k_combinations": top_k_combinations,
            "combo_min_hit_prob": dict(
                min_hit_prob_by_bet_type
                if min_hit_prob_by_bet_type is not None
                else DEFAULT_COMBO_MIN_HIT_PROB
            ),
        },
    )

    if frame.empty:
        log.warning("No races in window — returning empty simulation result")
        return result

    # Aggregation buckets (keyed by group label)
    bet_type_groups: dict[str, GroupStats] = {}
    race_class_groups: dict[str, GroupStats] = {}
    course_groups: dict[str, GroupStats] = {}

    race_ids = list(frame["race_id"].unique())
    result.n_races = len(race_ids)
    log.info("Simulating %d races...", result.n_races)

    # 累計損益を 0 から積む。賭け金は残高に依存しないので破産は起きない
    # (= 途中で評価が止まらない)。マイナスの最小値は「途中で止まらずに回すのに
    # 必要だった資金」として残す。
    n_skipped_place = 0
    trough_profit = 0
    current_profit = 0
    peak_profit = 0
    # 日次バケット: その日の累計 stake / payout / 最後の race 終了時の累計損益。
    daily_buckets: dict[str, dict[str, float | int]] = {}

    # odds.db の実オッズで EV 選択を実測ベースにする。未 backfill のレースは
    # load_race_odds が {} を返し、従来の Plackett-Luce 推定へフォールバックする
    # （後方互換）。読み取り専用なので close は loop 後にまとめて行う。
    odds_engine = make_odds_engine()
    init_odds_db(odds_engine)
    odds_session = Session(bind=odds_engine)

    n_settled = 0
    for race_id in race_ids:
        race_frame = frame[frame["race_id"] == race_id]
        if race_frame.empty or len(race_frame) < 2:
            continue

        # Predictions (NN bundle 経由)
        try:
            preds = predict_race(bundle, race_frame, session=session)
        except Exception as exc:  # noqa: BLE001
            log.warning("predict_race failed for %s: %s", race_id, exc)
            continue

        # Attach post_position (recommend_for_race needs it)
        pp_map = dict(zip(race_frame["horse_id"].values, race_frame["post_position"].values, strict=True))
        preds["post_position"] = preds["horse_id"].map(pp_map)

        # 確信度 (確率専用モデルが指定されているときだけ)。単勝は買う/買わないの
        # 判定には使わず、点数だけを動かす。複勝は可否と厚みの両方に使う。
        race_types = types
        place_conf: float | None = None
        win_conf: float | None = None
        if prob_bundle is not None and not preds.empty:
            win_conf = pick_confidence(
                prob_bundle, race_frame, preds.iloc[0]["horse_id"],
                session=session, bet_type="単勝",
            )
        if prob_bundle is not None and "複勝" in types:
            conf = pick_confidence(
                prob_bundle, race_frame, preds.iloc[0]["horse_id"], session=session
            )
            if not is_place_worth_buying(conf, place_min_confidence):
                race_types = [b for b in types if b != "複勝"]
                n_skipped_place += 1
            else:
                # 買うと決めた後の厚み。確信度の高いレースに厚く賭ける。
                place_conf = conf

        # Combination predictions + odds (with implied fill)
        race_odds, race_odds_sources = compute_race_odds_with_sources(
            session, race_id, odds_session=odds_session
        )
        try:
            combos_by_type = predict_race_with_combinations(
                bundle,
                race_frame,
                session=session,
                top_k_combinations=top_k_combinations,
                race_odds=race_odds,
                race_odds_sources=race_odds_sources,
            )
            if prob_bundle is not None:
                # 連系だけ確率モデル由来に差し替える (単勝・複勝の候補は active のまま)
                combos_by_type = merge_combination_sources(
                    combos_by_type,
                    _combinations_from_base(
                        base_df=_predict_race_nn(prob_bundle, race_frame, session=session),
                        frame=race_frame,
                        n_samples=10_000,
                        rng=None,
                        top_k_combinations=top_k_combinations,
                        race_odds=race_odds,
                        race_odds_sources=race_odds_sources,
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("predict_race_with_combinations failed for %s: %s", race_id, exc)
            continue

        # **点数は確信度から決める** (RACE 画面と同じ `points_for_confidence`)。
        # 1 点 = 100 円。連系は 1 組合せ 1 点で、何点買うかは的中確率の下限が決める。
        points = {
            "単勝": points_for_confidence("単勝", win_conf),
            "複勝": points_for_confidence("複勝", place_conf),
        }

        rec = recommend_for_race(
            predictions=preds,
            combinations_by_type=combos_by_type,
            race_id=race_id,
            race_budget=race_budget,
            points_by_bet_type=points,
            win_min_odds=win_min_odds,
            top_n_horses=eff_top_n,
            enabled_bet_types=race_types,
            min_hit_prob_by_bet_type=min_hit_prob_by_bet_type,
        )

        # Determine finish_position map (only finished races settle)
        finished_rows = race_frame[race_frame["finish_position"].notna()]
        if finished_rows.empty:
            continue
        finish_to_pp: dict[int, int] = {}
        for _, row in finished_rows.iterrows():
            try:
                fp = int(row["finish_position"])
                pp = int(row["post_position"])
                finish_to_pp[fp] = pp
            except (ValueError, TypeError):
                continue
        if not finish_to_pp:
            continue

        n_settled += 1

        past_odds = compute_past_race_odds(session, race_id)

        # Aggregate per-race attributes
        race_class = (
            race_frame["race_class"].dropna().iloc[0]
            if "race_class" in race_frame.columns
               and not race_frame["race_class"].dropna().empty
            else "unknown"
        )
        course = (
            race_frame["course"].dropna().iloc[0]
            if "course" in race_frame.columns
               and not race_frame["course"].dropna().empty
            else "unknown"
        )

        settlements = _settle_candidates(
            rec.candidates, race_id, finish_to_pp, past_odds
        )

        # 累計損益の更新。0 スタートなので、そのままプラス / マイナスを表す。
        # NaN / Inf ガード: odds が壊れた値だと payout が NaN になり得るので 0 に丸める。
        race_invested = sum(int(s["stake"]) for s in settlements)
        race_payout_raw = sum(float(s["payout"]) for s in settlements)
        race_payout = race_payout_raw if math.isfinite(race_payout_raw) else 0.0
        current_profit = current_profit - race_invested + int(round(race_payout))
        trough_profit = min(trough_profit, current_profit)
        peak_profit = max(peak_profit, current_profit)

        # 日次バケット update (race の date 単位で集約)
        race_date_str = (
            str(race_frame["date"].iloc[0])
            if "date" in race_frame.columns and not race_frame.empty
            else ""
        )
        if race_date_str:
            bucket = daily_buckets.setdefault(
                race_date_str,
                {"invested": 0, "payout": 0.0, "n_bets": 0, "profit_at_end": current_profit},
            )
            bucket["invested"] = int(bucket["invested"]) + race_invested
            bucket["payout"] = float(bucket["payout"]) + race_payout
            bucket["n_bets"] = int(bucket["n_bets"]) + len(settlements)
            # 同一日内の race は順次処理されるので、最後の race 後の損益が残る
            bucket["profit_at_end"] = current_profit

        for s in settlements:
            # NaN を 0 として扱う (集計 / pydantic int 化で落ちないため)
            s_payout = (
                float(s["payout"])
                if math.isfinite(float(s["payout"]))
                else 0.0
            )
            # Optional per-bet record sink (CI / source-coverage analysis).
            if bet_sink is not None:
                bet_sink.append({
                    "race_id": race_id,
                    "date": race_date_str,
                    "bet_type": s["bet_type"],
                    "stake": int(s["stake"]),
                    "payout": s_payout,
                    "hit": int(s["hit"]),
                    "source": s.get("source", "unknown"),
                })
            # global summary
            result.summary.n_bets += 1
            result.summary.invested += s["stake"]
            result.summary.payout += s_payout
            result.summary.hits += s["hit"]
            # by bet_type
            grp = bet_type_groups.setdefault(
                s["bet_type"], GroupStats(label=s["bet_type"])
            )
            grp.n_bets += 1
            grp.invested += s["stake"]
            grp.payout += s_payout
            grp.hits += s["hit"]
            # by race_class
            cls_grp = race_class_groups.setdefault(
                race_class, GroupStats(label=str(race_class))
            )
            cls_grp.n_bets += 1
            cls_grp.invested += s["stake"]
            cls_grp.payout += s_payout
            cls_grp.hits += s["hit"]
            # by course
            crs_grp = course_groups.setdefault(
                course, GroupStats(label=str(course))
            )
            crs_grp.n_bets += 1
            crs_grp.invested += s["stake"]
            crs_grp.payout += s_payout
            crs_grp.hits += s["hit"]

    odds_session.close()

    if n_skipped_place:
        log.info(
            "skipped 複勝 in %d races (confidence < %.2f)", n_skipped_place, place_min_confidence
        )
    # 累計損益の最小値 (= その額だけ沈んだ時期があった)。
    result.trough_profit = trough_profit
    result.required_capital = max(0, -trough_profit)
    result.n_settled_races = n_settled
    result.final_profit = current_profit
    result.peak_profit = peak_profit
    # Sort groups by invested desc for predictable display order
    result.by_bet_type = sorted(
        bet_type_groups.values(), key=lambda g: g.invested, reverse=True
    )
    result.by_race_class = sorted(
        race_class_groups.values(), key=lambda g: g.invested, reverse=True
    )
    result.by_course = sorted(
        course_groups.values(), key=lambda g: g.invested, reverse=True
    )
    # 日次の損益推移を date 昇順で list 化 (グラフ用)
    result.profit_timeseries = [
        ProfitPoint(
            date=d,
            profit=int(v["profit_at_end"]),
            invested=int(v["invested"]),
            payout=float(v["payout"]),
            n_bets=int(v["n_bets"]),
        )
        for d, v in sorted(daily_buckets.items())
    ]

    log.info(
        "Done. %d settled races, %d bets, payback=%.3f, hit_rate=%.3f, "
        "profit=%+d (peak=%+d, trough=%+d)",
        n_settled, result.summary.n_bets,
        result.summary.payback_rate, result.summary.hit_rate,
        result.final_profit, result.peak_profit, result.trough_profit,
    )
    return result
