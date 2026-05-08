"""連系 馬券 (馬連 / ワイド / 馬単 / 三連複 / 三連単) の確率 calibration 診断。

仕組み:
  - 期間内の各 race で predict_race_with_combinations を回し、
    各 combo の (predicted prob, actual hit) を集める
  - 馬券種ごとに predicted-prob 10 等分位 bucket で actual_rate と比較
  - mean_pred / actual_rate の ratio が 1x から大きく外れる bucket を可視化

Goal:
  単勝 calibration は単勝 isotonic calibrator で補正済だが、その下流の
  Plackett-Luce 由来の連系 combo 確率がどれだけズレているかを定量化する。
  ratio >> 1x が低 prob 帯で続くなら、連系 EV 計算に系統的バイアス
  (= 過大評価) が乗っており、payback < 1.0 の根本原因として強く示唆される。

CLI:
  uv run python -m keiba_ai.ai.combo_calibration_diagnosis \
      --model data/models/<timestamp> \
      --start 2024-10-01 --end 2024-12-31
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sqlalchemy import select

from keiba_ai.ai.predict import predict_race_with_combinations
from keiba_ai.ai.registry import load_model_full
from keiba_ai.core.logging import get_logger
from keiba_ai.core.paths import db_path
from keiba_ai.db.models.entry import Entry
from keiba_ai.db.session import make_engine, session_scope
from keiba_ai.features.builder import build_training_frame

log = get_logger(__name__)


_BET_TYPES = ["馬連", "ワイド", "馬単", "三連複", "三連単"]


def _get_top3(session, race_id: str) -> list[tuple[int, int]]:
    """Top-3 finisher の (finish_position, post_position) を返す。
    1〜3 着が揃っていなければ空リスト。"""
    rows = session.execute(
        select(Entry.finish_position, Entry.post_position)
        .where(Entry.race_id == race_id)
        .where(Entry.finish_position.in_([1, 2, 3]))
        .where(Entry.post_position.is_not(None))
        .order_by(Entry.finish_position)
    ).all()
    out = [(int(r.finish_position), int(r.post_position)) for r in rows]
    if len(out) < 3:
        return []
    return out


def _is_hit(bet_type: str, combo: str, top3: list[tuple[int, int]]) -> bool:
    """combo 文字列が実 top-3 と一致するかを判定する。"""
    by_finish = {fp: pp for fp, pp in top3}
    pp1 = by_finish.get(1)
    pp2 = by_finish.get(2)
    pp3 = by_finish.get(3)

    try:
        if bet_type == "馬連":
            pps = sorted(int(x) for x in combo.split("-"))
            return pps == sorted([pp1, pp2])
        if bet_type == "ワイド":
            pps = {int(x) for x in combo.split("-")}
            return pps.issubset({pp1, pp2, pp3})
        if bet_type == "馬単":
            parts = [int(x) for x in combo.split("→")]
            return parts == [pp1, pp2]
        if bet_type == "三連複":
            pps = sorted(int(x) for x in combo.split("-"))
            return pps == sorted([pp1, pp2, pp3])
        if bet_type == "三連単":
            parts = [int(x) for x in combo.split("→")]
            return parts == [pp1, pp2, pp3]
    except (ValueError, TypeError):
        return False
    return False


def diagnose_combo_calibration(
    model_path: Path,
    db: Path | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """Compute combo-prob calibration diagnostics for 連系 5 種。

    Returns dict with per-bet-type buckets and Brier score.
    """
    resolved_db = db or db_path()
    engine = make_engine(resolved_db)
    bundle = load_model_full(model_path)
    model = bundle.lambdarank

    log.info("Building evaluation frame %s..%s", start, end)
    with session_scope(engine) as session:
        frame = build_training_frame(session, train_start=start, train_end=end)

    if frame.empty:
        log.warning("No rows in evaluation window")
        return {"n_races": 0, "results": {}, "model_path": str(model_path)}

    # bet_type ごとに (predicted_prob, hit, est_odds) を蓄積
    records: dict[str, list[tuple[float, int]]] = {bt: [] for bt in _BET_TYPES}

    n_races = 0
    n_skipped_no_top3 = 0
    n_skipped_pp_nan = 0

    with session_scope(engine) as session:
        for race_id, race_frame in frame.groupby("race_id"):
            if len(race_frame) < 4:
                continue
            if race_frame["post_position"].isna().any():
                n_skipped_pp_nan += 1
                continue
            top3 = _get_top3(session, race_id)
            if not top3:
                n_skipped_no_top3 += 1
                continue
            try:
                combo_map = predict_race_with_combinations(
                    model, race_frame,
                    binary_model=bundle.binary,
                    calibrator=bundle.calibrator,
                    combo_calibrators=bundle.combo_calibrators,
                )
            except Exception as exc:
                log.warning("predict_race_with_combinations failed for %s: %s",
                            race_id, exc)
                continue

            n_races += 1
            for bt in _BET_TYPES:
                for cp in combo_map.get(bt, []):
                    records[bt].append(
                        (float(cp.prob), 1 if _is_hit(bt, cp.combo, top3) else 0)
                    )

    log.info(
        "Diagnosed %d races (skipped no_top3=%d pp_nan=%d)",
        n_races, n_skipped_no_top3, n_skipped_pp_nan,
    )

    results: dict[str, dict] = {}
    for bt, recs in records.items():
        if not recs:
            results[bt] = {"n_combos": 0, "buckets": [], "brier": None}
            continue
        df = pd.DataFrame(recs, columns=["pred_prob", "hit"])
        # 10 等分位 bucket
        try:
            df["bucket"] = pd.qcut(df["pred_prob"], q=10, duplicates="drop")
        except ValueError:
            # 全 prob が同値の場合
            df["bucket"] = "all"
        agg = df.groupby("bucket", observed=True).agg(
            n=("hit", "size"),
            mean_pred=("pred_prob", "mean"),
            actual_rate=("hit", "mean"),
        )
        buckets = []
        for bucket_label, row in agg.iterrows():
            mean_p = float(row["mean_pred"])
            act = float(row["actual_rate"])
            ratio = mean_p / act if act > 0 else None
            buckets.append({
                "bucket": str(bucket_label),
                "n": int(row["n"]),
                "mean_pred": round(mean_p, 6),
                "actual_rate": round(act, 6),
                "ratio_pred_over_actual": round(ratio, 2) if ratio is not None else None,
            })
        brier = float(((df["pred_prob"] - df["hit"]) ** 2).mean())
        results[bt] = {
            "n_combos": len(df),
            "buckets": buckets,
            "brier": round(brier, 6),
        }

    return {
        "model_path": str(model_path),
        "window": {"start": start, "end": end},
        "n_races": n_races,
        "n_skipped_no_top3": n_skipped_no_top3,
        "n_skipped_pp_nan": n_skipped_pp_nan,
        "results": results,
    }


def _format_report(diag: dict) -> str:
    lines: list[str] = []
    lines.append("=== Combo prob calibration diagnosis ===")
    lines.append(f"model:    {diag['model_path']}")
    w = diag.get("window", {})
    lines.append(f"window:   {w.get('start')} 〜 {w.get('end')}  ({diag['n_races']} races)")
    if diag.get("n_skipped_no_top3"):
        lines.append(f"skipped:  no_top3={diag['n_skipped_no_top3']} pp_nan={diag['n_skipped_pp_nan']}")
    lines.append("")
    for bt, info in diag["results"].items():
        if info["n_combos"] == 0:
            continue
        lines.append(f"--- {bt}  ({info['n_combos']} combos, Brier {info['brier']}) ---")
        lines.append(f"  {'mean_pred':>10}  {'actual':>10}  {'ratio':>8}  {'n':>6}")
        for b in info["buckets"]:
            ratio = (
                f"{b['ratio_pred_over_actual']:.2f}x"
                if b["ratio_pred_over_actual"] is not None
                else "  inf"
            )
            lines.append(
                f"  {b['mean_pred']:>10.5f}  {b['actual_rate']:>10.5f}  "
                f"{ratio:>8}  {b['n']:>6}"
            )
        lines.append("")
    return "\n".join(lines)


def _main():
    parser = argparse.ArgumentParser(
        description="Diagnose combo-bet probability calibration"
    )
    parser.add_argument("--model", type=Path, required=True, help="Model directory")
    parser.add_argument("--db", type=Path, help="Path to SQLite DB")
    parser.add_argument("--start", help="Eval start date YYYY-MM-DD")
    parser.add_argument("--end", help="Eval end date YYYY-MM-DD")
    parser.add_argument("--json", action="store_true", help="Emit raw JSON")
    args = parser.parse_args()

    diag = diagnose_combo_calibration(
        model_path=args.model, db=args.db, start=args.start, end=args.end
    )
    if args.json:
        print(json.dumps(diag, ensure_ascii=False, indent=2))
    else:
        print(_format_report(diag))


if __name__ == "__main__":
    _main()
