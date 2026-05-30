"""Backtest all bet types (incl. exotics) over a window and print ROI by type.

Tests whether exotic (連系/3連系) markets are inefficient enough for the
model's combo probabilities to find positive ROI, where simple win/place
cannot beat the ~20% takeout.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ai.simulation import simulate_active_model
from core.paths import db_path
from db.session import make_engine, session_scope


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--strategy", default="conservative")
    ap.add_argument("--budget", type=int, default=1_000_000)
    args = ap.parse_args()

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        res = simulate_active_model(
            session,
            model_path=args.model,
            start=args.start,
            end=args.end,
            budget=args.budget,
            strategy=args.strategy,
        )
    import json
    d = res.as_dict()
    Path("/tmp/exotic_bt.json").write_text(json.dumps(d, ensure_ascii=False, indent=2))
    s = d["summary"]
    print(f"\n=== OVERALL ({args.strategy}) {args.start}..{args.end} ===")
    print(f"  n_races={d['n_races']} settled={d['n_settled_races']}")
    print(f"  n_bets={s['n_bets']} invested={s['invested']} payout={s['payout']} "
          f"payback={s['payback_rate']:.3f} hit={s['hit_rate']:.3f}")
    print(f"  final_bankroll={d.get('final_bankroll')} peak={d.get('peak_bankroll')}")
    print("\n=== BY BET TYPE ===")
    print(f"{'type':>8} {'n_bets':>7} {'invested':>10} {'payout':>10} {'payback':>8} {'hit%':>6}")
    for g in sorted(d["by_bet_type"], key=lambda x: -x["invested"]):
        print(f"{g['label']:>8} {g['n_bets']:>7} {g['invested']:>10} {g['payout']:>10} "
              f"{g['payback_rate']:>8.3f} {g['hit_rate']*100:>6.1f}")


if __name__ == "__main__":
    main()
