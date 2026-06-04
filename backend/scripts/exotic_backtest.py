"""Backtest all bet types (incl. exotics) over a window and print ROI by type.

Tests whether exotic (連系/3連系) markets are inefficient enough for the
model's combo probabilities to find positive ROI, where simple win/place
cannot beat the ~20% takeout.

Beyond the point-estimate payback, this prints two things needed to actually
trust the numbers:
  - **race-level bootstrap 95% CI** on payback per bet type. Exotic bets within
    a race share the same outcome, so the race is the unit of independence;
    a payback point estimate over a few hundred correlated bets is otherwise
    uninterpretable (the famous "馬連 0.93" might be pure noise).
  - **odds source coverage**: what fraction of each bet type's selection EV was
    priced by real scraped odds (odds.db) vs the Plackett-Luce `implied`
    fallback. A real-odds verdict is only valid where coverage is ~100% scraped.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ai.simulation import simulate_active_model
from core.paths import db_path
from db.session import make_engine, session_scope

# JRA takeout → break-even payback (1 - takeout) per bet type, for reference.
_BREAK_EVEN: dict[str, float] = {
    "単勝": 0.80, "複勝": 0.80, "枠連": 0.775, "馬連": 0.775, "ワイド": 0.775,
    "馬単": 0.75, "三連複": 0.75, "三連単": 0.725,
}


def _race_level_payback_ci(
    bets: list[dict], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """Race-level bootstrap CI for payback = Σpayout / Σstake.

    Resamples *races* (not bets) with replacement so correlated within-race bets
    move together. Returns (payback, lo2.5%, hi97.5%). Empty → (0,0,0).
    """
    per_race: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])  # race -> [stake, payout]
    for b in bets:
        per_race[b["race_id"]][0] += b["stake"]
        per_race[b["race_id"]][1] += b["payout"]
    races = list(per_race.values())
    if not races:
        return 0.0, 0.0, 0.0
    arr = np.array(races, dtype=np.float64)  # (R, 2): stake, payout
    total_stake = arr[:, 0].sum()
    point = arr[:, 1].sum() / total_stake if total_stake > 0 else 0.0

    rng = np.random.default_rng(seed)
    n = len(arr)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = arr[idx, 0].sum()
        boots[i] = arr[idx, 1].sum() / s if s > 0 else 0.0
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--strategy", default="conservative")
    ap.add_argument("--budget", type=int, default=1_000_000)
    args = ap.parse_args()

    bet_sink: list[dict] = []
    engine = make_engine(db_path())
    with session_scope(engine) as session:
        res = simulate_active_model(
            session,
            model_path=args.model,
            start=args.start,
            end=args.end,
            budget=args.budget,
            strategy=args.strategy,
            bet_sink=bet_sink,
        )

    d = res.as_dict()
    Path("/tmp/exotic_bt.json").write_text(json.dumps(d, ensure_ascii=False, indent=2))
    s = d["summary"]
    print(f"\n=== OVERALL ({args.strategy}) {args.start}..{args.end} ===")
    print(f"  n_races={d['n_races']} settled={d['n_settled_races']}")
    print(f"  n_bets={s['n_bets']} invested={s['invested']} payout={s['payout']} "
          f"payback={s['payback_rate']:.3f} hit={s['hit_rate']:.3f}")
    print(f"  final_bankroll={d.get('final_bankroll')} peak={d.get('peak_bankroll')}")

    # Per-bet-type: bootstrap CI + source coverage from the per-bet sink.
    by_type: dict[str, list[dict]] = defaultdict(list)
    for b in bet_sink:
        by_type[b["bet_type"]].append(b)

    print("\n=== BY BET TYPE (race-level bootstrap 95% CI; scraped% = real-odds coverage) ===")
    print(f"{'type':>6} {'n_bets':>7} {'n_race':>6} {'payback':>8} "
          f"{'95% CI':>16} {'break':>6} {'>1?':>4} {'scraped%':>8} {'hit%':>6}")
    for bt in sorted(by_type, key=lambda t: -sum(x["stake"] for x in by_type[t])):
        bets = by_type[bt]
        pb, lo, hi = _race_level_payback_ci(bets)
        n_race = len({b["race_id"] for b in bets})
        n_scraped = sum(1 for b in bets if b["source"] == "scraped")
        scraped_pct = 100.0 * n_scraped / len(bets) if bets else 0.0
        hit_pct = 100.0 * sum(b["hit"] for b in bets) / len(bets) if bets else 0.0
        be = _BREAK_EVEN.get(bt, 0.8)
        # Is the bet type plausibly +EV? Only if the CI lower bound clears 1.0.
        # Guard against tiny samples where a few longshot hits fake a high payback
        # (a 2-race bootstrap CI is meaningless).
        if n_race < 30:
            verdict = "n/a"
        elif lo > 1.0:
            verdict = "YES"
        elif hi > 1.0:
            verdict = "~"
        else:
            verdict = "no"
        print(f"{bt:>6} {len(bets):>7} {n_race:>6} {pb:>8.3f} "
              f"[{lo:>5.2f},{hi:>5.2f}] {be:>6.3f} {verdict:>4} "
              f"{scraped_pct:>7.1f}% {hit_pct:>6.1f}")


if __name__ == "__main__":
    main()
