"""One-shot value-betting analysis for an odds-free model.

Predicts each OOS race once, dumps a per-horse table, then sweeps win-EV
thresholds and popularity bands to find any positive-ROI region.

The model's win_prob is independent of the market odds (trained with
KEIBA_EXCLUDE_ODDS_FEATURES), so `win_prob * odds_win > T` is a genuine
"model disagrees with the market in this horse's favour" (value) signal.

Usage:
  PYTHONPATH=src python -m scripts.value_bet_sweep --model <dir> --start ... --end ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ai.predict import predict_race
from ai.registry import load_model_full
from core.paths import db_path
from db.session import make_engine, session_scope
from features.builder import build_training_frame


def collect(model_path: Path, start: str, end: str) -> pd.DataFrame:
    engine = make_engine(db_path())
    bundle = load_model_full(model_path)
    with session_scope(engine) as session:
        frame = build_training_frame(session, train_start=start, train_end=end)
    rows = []
    for race_id, rf in frame.groupby("race_id"):
        if len(rf) < 2:
            continue
        preds = predict_race(bundle, rf)
        m = preds.merge(
            rf[["horse_id", "finish_position", "odds_win", "popularity"]],
            on="horse_id", how="left",
        )
        m["race_id"] = race_id
        # model predicted rank (1 = top score)
        m = m.sort_values("score", ascending=False).reset_index(drop=True)
        m["model_rank"] = np.arange(1, len(m) + 1)
        rows.append(m)
    return pd.concat(rows, ignore_index=True)


def sweep(df: pd.DataFrame) -> None:
    d = df.dropna(subset=["odds_win", "win_prob"]).copy()
    d = d[d["odds_win"] > 0]
    d["ev"] = d["win_prob"] * d["odds_win"]
    d["won"] = (d["finish_position"] == 1).astype(float)
    d["ret"] = np.where(d["won"] > 0, d["odds_win"], 0.0)  # payout per 1 unit staked

    n_races = d["race_id"].nunique()
    print(f"\n=== WIN value-bet sweep — {n_races} races, {len(d)} horse-rows ===")
    print(f"{'EV>':>6} {'bets':>7} {'bets/race':>9} {'hit%':>7} {'payback':>8} {'avg_odds':>8}")
    for T in [1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]:
        sel = d[d["ev"] > T]
        if len(sel) == 0:
            print(f"{T:>6.1f} {0:>7}")
            continue
        pb = sel["ret"].sum() / len(sel)
        hit = sel["won"].mean() * 100
        print(f"{T:>6.1f} {len(sel):>7} {len(sel)/n_races:>9.2f} {hit:>7.1f} {pb:>8.3f} {sel['odds_win'].mean():>8.1f}")

    # By popularity band (no EV filter) — bet model's top pick per race
    print("\n=== model TOP-1 pick payback by popularity band ===")
    top = d[d["model_rank"] == 1].copy()
    print(f"{'pop band':>10} {'bets':>7} {'hit%':>7} {'payback':>8}")
    for lo, hi, lbl in [(1,1,"1"),(2,3,"2-3"),(4,6,"4-6"),(7,9,"7-9"),(10,18,"10+")]:
        sel = top[(top["popularity"] >= lo) & (top["popularity"] <= hi)]
        if len(sel) == 0:
            continue
        pb = sel["ret"].sum() / len(sel)
        print(f"{lbl:>10} {len(sel):>7} {sel['won'].mean()*100:>7.1f} {pb:>8.3f}")

    # 2-D: EV threshold x popularity band
    print("\n=== payback grid: rows=EV>, cols=popularity band ===")
    bands = [(1,3,"1-3"),(4,6,"4-6"),(7,9,"7-9"),(10,18,"10+")]
    header = "EV>".rjust(6) + "".join(f"{lbl:>14}" for _,_,lbl in bands)
    print(header)
    for T in [1.0, 1.2, 1.5, 2.0]:
        line = f"{T:>6.1f}"
        for lo, hi, _ in bands:
            sel = d[(d["ev"] > T) & (d["popularity"] >= lo) & (d["popularity"] <= hi)]
            if len(sel) == 0:
                line += f"{'-':>14}"
            else:
                pb = sel["ret"].sum() / len(sel)
                line += f"{f'{pb:.2f}({len(sel)})':>14}"
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--dump", type=Path, default=None, help="optional pkl dump of the per-horse table")
    args = ap.parse_args()
    df = collect(args.model, args.start, args.end)
    if args.dump:
        df.to_pickle(args.dump)
        print(f"dumped {len(df)} rows to {args.dump}")
    sweep(df)


if __name__ == "__main__":
    main()
