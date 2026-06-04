"""active モデルで simulate_active_model を回し、ROI を券種別 + source 別に出す。

combo 校正の前後比較用。bet_sink で per-bet record を集め、(bet_type, source)
別の払戻率も出す（どの source がエッジ崩壊の主因かを見る）。

Usage:
  KEIBA_DATA_DIR=/tmp/keiba-snap PYTHONPATH=src \
    python -m scripts.run_sim_roi --model <dir> --start 2026-01-01 --end 2026-12-31 \
      [--budget 1000000] [--strategy balanced]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from sqlalchemy.orm import Session

from ai.simulation import simulate_active_model
from core.paths import db_path
from db.session import make_engine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--budget", type=int, default=1_000_000)
    ap.add_argument("--strategy", default="balanced")
    args = ap.parse_args()

    engine = make_engine(db_path())
    sink: list[dict] = []
    with Session(bind=engine) as session:
        res = simulate_active_model(
            session, args.model, args.start, args.end,
            budget=args.budget, strategy=args.strategy,
            bet_sink=sink,
        )

    s = res.summary
    print(f"\n=== SUMMARY ({args.start}..{args.end}, {args.strategy}, budget={args.budget}) ===")
    print(f"races={res.n_races} settled={res.n_settled_races}")
    print(f"bets={s.n_bets}  invested={s.invested:,}  payout={round(s.payout):,}")
    print(f"ROI(payback)={s.payback_rate:.4f}  hit_rate={s.hit_rate:.4f}")
    print(f"final_bankroll={res.final_bankroll:,}  peak={res.peak_bankroll:,}")

    print("\n=== by bet_type ===")
    print(f"{'bet_type':>8} {'bets':>7} {'invested':>12} {'payout':>12} {'ROI':>6} {'hit%':>6}")
    for g in res.by_bet_type:
        print(f"{g.label:>8} {g.n_bets:>7} {g.invested:>12,} {round(g.payout):>12,} "
              f"{g.payback_rate:>6.2f} {g.hit_rate*100:>6.1f}")

    # Flat-stake ROI from sink (path-independent: every bet weighted equally).
    # payout/stake == realized return multiple (odds if hit, else 0), so the
    # mean over bets is the flat-stake payback — removes compounding bankroll
    # path-dependence that makes the headline (Kelly) ROI noisy across runs.
    flat: dict[str, list[float]] = defaultdict(list)  # bet_type -> [ret_multiple,...]
    flat_all: list[float] = []
    for b in sink:
        st = b["stake"]
        if st <= 0:
            continue
        r = b["payout"] / st
        flat[b["bet_type"]].append(r)
        flat_all.append(r)
    print("\n=== FLAT-STAKE ROI (path-independent) ===")
    print(f"{'bet_type':>8} {'bets':>7} {'flatROI':>8} {'hit%':>6}")
    for bt, rs in sorted(flat.items(), key=lambda kv: -len(kv[1])):
        roi = sum(rs) / len(rs)
        hit = sum(1 for r in rs if r > 0) / len(rs) * 100
        print(f"{bt:>8} {len(rs):>7} {roi:>8.3f} {hit:>6.1f}")
    if flat_all:
        print(f"{'ALL':>8} {len(flat_all):>7} {sum(flat_all)/len(flat_all):>8.3f} "
              f"{sum(1 for r in flat_all if r>0)/len(flat_all)*100:>6.1f}")


if __name__ == "__main__":
    main()
