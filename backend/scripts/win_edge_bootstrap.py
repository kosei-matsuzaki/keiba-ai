"""Race-level bootstrap CI for candidate WIN-betting filters.

Loads the per-horse dump produced by value_bet_sweep --dump, defines several
candidate betting filters, and computes a race-level bootstrap CI on payback
(回収率 = total_payout / total_invested). Race-level resampling is the correct
unit because bets within a race are correlated and the high-odds longshot wins
that inflate point-estimate payback are concentrated in a few races.

A filter only constitutes a real edge if the 2.5%% CI bound > 1.0.

Usage:
  PYTHONPATH=src python -m scripts.win_edge_bootstrap --dump /tmp/active_perhorse.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["odds_win", "win_prob"]).copy()
    d = d[d["odds_win"] > 0]
    d["ev"] = d["win_prob"] * d["odds_win"]
    d["won"] = (d["finish_position"] == 1).astype(float)
    d["payout"] = np.where(d["won"] > 0, d["odds_win"] * 100.0, 0.0)  # 100 yen stake
    d["stake"] = 100.0
    return d


def race_agg(sel: pd.DataFrame, all_races: np.ndarray) -> pd.DataFrame:
    """Per-race invested/payout, reindexed over ALL evaluated races (0-filled)."""
    g = sel.groupby("race_id").agg(inv=("stake", "sum"), pay=("payout", "sum"))
    g = g.reindex(all_races, fill_value=0.0)
    return g


def bootstrap_payback(
    g: pd.DataFrame, n_iter: int = 5000, seed: int = 42
) -> tuple[float, float, float]:
    inv = g["inv"].to_numpy()
    pay = g["pay"].to_numpy()
    n = len(inv)
    rng = np.random.default_rng(seed)
    point = pay.sum() / inv.sum() if inv.sum() > 0 else float("nan")
    samples = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        si = inv[idx].sum()
        samples[i] = pay[idx].sum() / si if si > 0 else np.nan
    lo, hi = np.nanpercentile(samples, [2.5, 97.5])
    return point, lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--iters", type=int, default=5000)
    args = ap.parse_args()

    df = prep(pd.read_pickle(args.dump))
    all_races = np.sort(df["race_id"].unique())
    n_races = len(all_races)
    print(f"races={n_races}  horse-rows={len(df)}\n")

    fav = df[df["popularity"] == 1]
    top1 = df.sort_values("score", ascending=False).groupby("race_id").head(1)

    filters: list[tuple[str, pd.DataFrame]] = [
        ("favorite win (pop1)", fav),
        ("model top-1 pick", top1),
        ("EV>1.1 (all)", df[df["ev"] > 1.1]),
        ("EV>1.3 (all)", df[df["ev"] > 1.3]),
        ("EV>1.5 (all)", df[df["ev"] > 1.5]),
        ("EV>1.5 & pop>=10", df[(df["ev"] > 1.5) & (df["popularity"] >= 10)]),
        ("EV>2.0 & pop>=10", df[(df["ev"] > 2.0) & (df["popularity"] >= 10)]),
        ("EV>1.0 & pop7-9", df[(df["ev"] > 1.0) & (df["popularity"].between(7, 9))]),
    ]

    print(f"{'filter':>22} {'bets':>6} {'hit%':>6} {'payback':>8}  {'95% CI (race-level boot)':>26}")
    for label, sel in filters:
        if len(sel) == 0:
            print(f"{label:>22} {0:>6}")
            continue
        g = race_agg(sel, all_races)
        point, lo, hi = bootstrap_payback(g, n_iter=args.iters)
        hit = sel["won"].mean() * 100
        edge = "  <-- CI>1.0" if lo > 1.0 else ""
        print(
            f"{label:>22} {len(sel):>6} {hit:>6.1f} {point:>8.3f}  [{lo:>6.3f}, {hi:>6.3f}]{edge}"
        )


if __name__ == "__main__":
    main()
