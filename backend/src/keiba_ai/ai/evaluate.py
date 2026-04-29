"""CLI: Backtest evaluation — NDCG, hit rates, and ROI.

Usage:
    uv run python -m keiba_ai.ai.evaluate --model <path>
                                           [--db PATH]
                                           [--start YYYY-MM-DD]
                                           [--end YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from keiba_ai.ai.labels import assign_relevance
from keiba_ai.ai.predict import predict_race
from keiba_ai.ai.registry import load_model
from keiba_ai.core.paths import db_path
from keiba_ai.db.models import ModelRun  # noqa: F401
from keiba_ai.db.session import make_engine, session_scope
from keiba_ai.features.builder import build_training_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

WIN_EV_THRESHOLD = 1.1   # Expected value threshold for win bet
PLACE_EV_THRESHOLD = 1.05  # Expected value threshold for place bet


def _parse_payout_place(json_str: str | None) -> dict[int, int]:
    """Parse payout_place JSON string into {finish_position: payout_yen} dict.

    Expected format: '{"1": 120, "2": 240, "3": 180}' where values are
    payout per 100 yen bet (Japanese convention).
    Returns empty dict if json_str is None or unparsable.
    """
    if not json_str:
        return {}
    try:
        raw = json.loads(json_str)
        return {int(k): int(v) for k, v in raw.items()}
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}


def evaluate(
    model_path: Path,
    db: Path | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """Run backtest evaluation and return metrics dict."""
    resolved_db = db or db_path()
    engine = make_engine(resolved_db)

    model = load_model(model_path)

    log.info("Building evaluation frame from %s", resolved_db)
    with session_scope(engine) as session:
        frame = build_training_frame(session, train_start=start, train_end=end)

    if frame.empty:
        log.warning("No evaluation data found.")
        return {}

    frame["relevance"] = frame["finish_position"].map(assign_relevance)

    # Per-race metrics
    ndcg1_list: list[float] = []
    ndcg3_list: list[float] = []
    top1_hits: list[int] = []
    place_hits: list[int] = []

    # Betting simulation — payback rate convention (回収率): gross_payout / invested
    # 1.00 = break-even, 1.10 = 10% profit, 0.80 = 20% loss
    win_bets = 0
    win_gross_payout = 0.0  # 払戻金合計（賭け金は含まない）
    win_invested = 0.0      # 賭け金合計

    # Place betting simulation (複勝)
    place_bets = 0
    place_gross_payout = 0.0
    place_invested = 0.0

    race_ids = frame["race_id"].unique()
    for race_id in race_ids:
        race_frame = frame[frame["race_id"] == race_id].copy()
        if len(race_frame) < 2:
            continue

        preds = predict_race(model, race_frame)
        # Merge actual finish positions
        actual = race_frame[["horse_id", "finish_position", "odds_win", "relevance"]].copy()
        preds = preds.merge(actual, on="horse_id", how="left")

        # NDCG
        true_rel = race_frame["relevance"].values.reshape(1, -1)
        # Align scores to same order as race_frame
        score_map = dict(zip(preds["horse_id"], preds["score"], strict=False))
        pred_scores = np.array([score_map.get(h, 0.0) for h in race_frame["horse_id"]]).reshape(1, -1)
        ndcg1_list.append(float(ndcg_score(true_rel, pred_scores, k=1)))
        ndcg3_list.append(float(ndcg_score(true_rel, pred_scores, k=3)))

        # Top-1 hit: does the horse ranked #1 by model actually finish 1st?
        top_horse = preds.iloc[0]  # sorted by score desc
        top1_hits.append(1 if top_horse["finish_position"] == 1 else 0)

        # Place hit: is at least one of top-3 model picks in actual top-3?
        top3_horses = set(preds.iloc[:3]["horse_id"])
        actual_top3 = set(
            actual[actual["finish_position"].notna() & (actual["finish_position"] <= 3)]["horse_id"]
        )
        place_hits.append(1 if top3_horses & actual_top3 else 0)

        # Win betting: bet if win_prob × odds_win > WIN_EV_THRESHOLD
        for _, row in preds.iterrows():
            odds = row.get("odds_win")
            if odds is None or pd.isna(odds):
                continue
            ev = row["win_prob"] * odds
            if ev > WIN_EV_THRESHOLD:
                win_bets += 1
                win_invested += 100
                if row.get("finish_position") == 1:
                    win_gross_payout += odds * 100

        # Place betting (複勝): requires payout_place data on the race frame
        # race_frame may carry payout_place if the training frame includes it.
        # Look it up from the race_frame column if present.
        payout_place_raw: str | None = None
        if "payout_place" in race_frame.columns:
            vals = race_frame["payout_place"].dropna()
            if not vals.empty:
                payout_place_raw = vals.iloc[0]

        payout_place_map = _parse_payout_place(payout_place_raw)
        if payout_place_map:
            # Determine the minimum payout across 1st/2nd/3rd place for EV calculation.
            # Using min payout gives a conservative estimate of expected return.
            min_payout = min(payout_place_map.values())
            # min_payout is in yen per 100 yen bet, so odds = min_payout / 100
            min_odds = min_payout / 100.0

            for _, row in preds.iterrows():
                ev = row["place_prob"] * min_odds
                if ev > PLACE_EV_THRESHOLD:
                    place_bets += 1
                    place_invested += 100
                    finish_pos = row.get("finish_position")
                    if (
                        finish_pos is not None
                        and not pd.isna(finish_pos)
                        and int(finish_pos) in payout_place_map
                    ):
                        place_gross_payout += payout_place_map[int(finish_pos)]

    n_races = len(ndcg1_list)
    metrics = {
        "n_races": n_races,
        "ndcg1": float(np.mean(ndcg1_list)) if ndcg1_list else float("nan"),
        "ndcg3": float(np.mean(ndcg3_list)) if ndcg3_list else float("nan"),
        "top1_hit": float(np.mean(top1_hits)) if top1_hits else float("nan"),
        # 上位 3 推奨のうち少なくとも 1 頭が実際に 3 着以内に入ったレース割合
        "place_hit": float(np.mean(place_hits)) if place_hits else float("nan"),
        "win_bets": win_bets,
        "win_invested": win_invested,
        "win_gross_payout": win_gross_payout,
        # 回収率 = 払戻金合計 / 賭け金合計（1.00 が損益分岐）
        "payback_win": (win_gross_payout / win_invested) if win_invested > 0 else float("nan"),
        # 複勝回収率
        "place_bets": place_bets,
        "place_invested": place_invested,
        "place_gross_payout": place_gross_payout,
        "payback_place": (
            (place_gross_payout / place_invested) if place_invested > 0 else float("nan")
        ),
    }

    log.info("Evaluation metrics: %s", metrics)
    return metrics


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate keiba-ai model via backtest")
    parser.add_argument("--model", type=Path, required=True, help="Path to model directory")
    parser.add_argument("--db", type=Path, default=None, help="Path to SQLite DB")
    parser.add_argument("--start", default=None, help="Evaluation start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Evaluation end date YYYY-MM-DD")
    args = parser.parse_args()

    metrics = evaluate(
        model_path=args.model,
        db=args.db,
        start=args.start,
        end=args.end,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
