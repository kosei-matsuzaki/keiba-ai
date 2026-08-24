"""履歴の無いレース (新馬戦など) を除くと回収率が上がるかを測る。

`features/race_info.py` の判定 (出走馬の過去走ゼロ率) でレースを 2 群に分け、
実運用と同じ買い方 (単勝・複勝ともモデルの本命 1 頭) での回収率を比べる。

Usage:
  PYTHONIOENCODING=utf-8 PYTHONPATH=src python -m scripts.low_information_races \\
      --start 2024-11-02 --end 2026-05-31
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai.evaluation.backtest import _parse_payout_place
from ai.inference.predict import predict_race
from ai.model.registry import get_active, load_model_full
from core.paths import db_path
from db.session import make_engine, session_scope
from features.builder import build_training_frame
from features.race_info import race_info_coverage


def _summarise(rows: pd.DataFrame, label: str) -> dict:
    if rows.empty:
        return {"group": label, "races": 0}
    win_ret = np.where(rows["won"], rows["odds_win"], 0.0)
    placed = rows["place_odds"].notna()
    place_ret = np.where(placed, rows["place_odds"].fillna(0.0), 0.0)
    return {
        "group": label,
        "races": int(len(rows)),
        "win_hit": round(float(rows["won"].mean()), 4),
        "win_payback": round(float(win_ret.mean()), 4),
        "place_hit": round(float(placed.mean()), 4),
        "place_payback": round(float(place_ret.mean()), 4),
        "mean_starts": round(float(rows["mean_starts"].mean()), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    engine = make_engine(db_path())
    with session_scope(engine) as s0:
        model_path = get_active(s0)
    if model_path is None:
        raise SystemExit("no active model")
    bundle = load_model_full(model_path)
    print(f"model: {model_path}", flush=True)

    recs: list[dict] = []
    with session_scope(engine) as session:
        frame = build_training_frame(session, train_start=args.start, train_end=args.end)
        race_ids = frame["race_id"].unique()
        print(f"{len(race_ids)} races", flush=True)
        for i, rid in enumerate(race_ids):
            rf = frame[frame["race_id"] == rid]
            if len(rf) < 2:
                continue
            cov = race_info_coverage(rf)
            preds = predict_race(bundle, rf, session=session)
            top = preds.iloc[0]
            row = rf[rf["horse_id"] == top["horse_id"]]
            if row.empty:
                continue
            r = row.iloc[0]
            odds = r.get("odds_win")
            fin = r.get("finish_position")
            if odds is None or pd.isna(odds) or pd.isna(fin):
                continue

            payout_map = {}
            if "payout_place" in rf.columns:
                vals = rf["payout_place"].dropna()
                if not vals.empty:
                    payout_map = _parse_payout_place(vals.iloc[0])
            place_odds = None
            if payout_map and float(fin) == int(fin) and int(fin) in payout_map:
                place_odds = payout_map[int(fin)] / 100.0

            recs.append({
                "race_id": rid,
                "race_class": r.get("race_class"),
                "is_low_information": cov.is_low_information,
                "debut_ratio": cov.debut_ratio,
                "mean_starts": cov.mean_starts,
                "won": bool(fin == 1),
                "odds_win": float(odds),
                "place_odds": place_odds,
            })
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(race_ids)}", flush=True)

    d = pd.DataFrame(recs)
    out = {
        "model": str(model_path),
        "window": [args.start, args.end],
        "all": _summarise(d, "全レース"),
        "low_information": _summarise(d[d["is_low_information"]], "情報が少ない (除外候補)"),
        "normal": _summarise(d[~d["is_low_information"]], "情報あり (除外後)"),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
