"""NN vs tree (GBDT) compared on ROI, not just accuracy.

Same no-odds features, same 2-year OOS, same value-betting policy. Each model
produces a per-race WIN probability; we bet 単勝 on horses with EV = p*odds > thr
and settle with the real win outcome (odds_win in keiba.db — pre-race, no leak).
ROI reported with race-level bootstrap 95% CI at several EV thresholds, for:

  - GBDT  : LightGBM binary is_winner head, per-race renormalised.
  - NN    : MLP conditional-logit (per-race softmax).
  - favorite reference (bet the lowest-odds horse).

ROI depends on beating the market price, which neither model does — so the
expected (and honest) outcome is GBDT ≈ NN ≈ <1: model CLASS does not change ROI.

Controlled split (train<2023-11). Run with KEIBA_EXCLUDE_ODDS_FEATURES=1.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
from roi_objective_experiment import build_races  # same dir (scripts/)

from core.paths import db_path
from db.session import make_engine, session_scope
from features.builder import build_training_frame, get_active_features

TRAIN_END, VALID_END = "2023-11-01", "2024-05-01"
OOS_START, OOS_END = "2024-05-01", "2026-04-30"
THRESHOLDS = (1.0, 1.1, 1.3)


def race_winprobs_gbdt(model, races, mu, sd):
    """Per-race win prob from a binary booster, renormalised within race."""
    out = []
    for X, o, won in races:
        raw = model.predict((X - mu) / sd)
        p = raw / raw.sum() if raw.sum() > 0 else np.full(len(raw), 1 / len(raw))
        out.append((p, o, won))
    return out


@torch.no_grad()
def race_winprobs_mlp(net, races, mu, sd, device):
    out = []
    for X, o, won in races:
        s = net(torch.from_numpy((X - mu) / sd).float().to(device)).squeeze(-1)
        p = torch.softmax(s, dim=0).cpu().numpy()
        out.append((p, o, won))
    return out


def value_bet_roi(winprobs, thr):
    """Bet 単勝 when p*odds > thr. Returns (roi, n_bets, per_race list, avg_odds)."""
    per_race = defaultdict(lambda: [0.0, 0.0])
    n = 0
    osum = 0.0
    for ri, (p, o, won) in enumerate(winprobs):
        ev = p * o
        bets = ev > thr
        if not bets.any():
            continue
        per_race[ri][0] += float(bets.sum())
        per_race[ri][1] += float((o * won)[bets].sum())
        n += int(bets.sum())
        osum += float(o[bets].sum())
    return per_race, n, osum


def boot_ci(per_race, n_boot=2000, seed=0):
    arr = np.array(list(per_race.values()), dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), 0.0, float("nan"), float("nan")
    roi = arr[:, 1].sum() / arr[:, 0].sum()
    rng = np.random.default_rng(seed)
    b = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(arr), len(arr))
        s = arr[idx, 0].sum()
        b[i] = arr[idx, 1].sum() / s if s > 0 else 0.0
    lo, hi = np.percentile(b, [2.5, 97.5])
    return roi, len(arr), float(lo), float(hi)


def report(name, winprobs):
    print(f"\n  {name}:")
    print(f"    {'EV>=':>5} {'n_bets':>7} {'n_race':>6} {'ROI':>7} {'95% CI':>16} {'avgOdds':>8}")
    for thr in THRESHOLDS:
        per_race, n, osum = value_bet_roi(winprobs, thr)
        roi, nr, lo, hi = boot_ci(per_race)
        avgo = osum / n if n else 0.0
        print(f"    {thr:>5.1f} {n:>7} {nr:>6} {roi:>7.3f} [{lo:>5.2f},{hi:>5.2f}] {avgo:>8.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    feats = get_active_features()
    engine = make_engine(db_path())
    with session_scope(engine) as s:
        frame = build_training_frame(s, train_start="2015-01-01", train_end=OOS_END)

    def races_of(lo, hi, inc):
        m = (frame["date"] >= lo) & ((frame["date"] <= hi) if inc else (frame["date"] < hi))
        return build_races(frame[m], feats)

    train, feat_used = races_of("0000", TRAIN_END, False)
    valid, _ = races_of(TRAIN_END, VALID_END, False)
    oos, _ = races_of(OOS_START, OOS_END, True)
    mu = np.concatenate([r[0] for r in train]).mean(0)
    sd = np.concatenate([r[0] for r in train]).std(0) + 1e-6
    print(f"races: train={len(train)} valid={len(valid)} oos={len(oos)} | no-odds feats={len(feat_used)}")

    # ── GBDT (binary is_winner) ────────────────────────────────────────────
    Xtr = np.concatenate([(r[0] - mu) / sd for r in train])
    ytr = np.concatenate([r[2] for r in train])
    Xva = np.concatenate([(r[0] - mu) / sd for r in valid])
    yva = np.concatenate([r[2] for r in valid])
    booster = lgb.train(
        {"objective": "binary", "metric": "binary_logloss", "num_leaves": 127,
         "learning_rate": 0.03, "min_data_in_leaf": 50, "feature_fraction": 0.85,
         "bagging_fraction": 0.85, "bagging_freq": 5, "verbose": -1},
        lgb.Dataset(Xtr, label=ytr), num_boost_round=2000,
        valid_sets=[lgb.Dataset(Xva, label=yva)],
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)],
    )

    # ── NN (MLP conditional logit) ─────────────────────────────────────────
    net = nn.Sequential(
        nn.Linear(len(feat_used), 128), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    tr_t = [(torch.from_numpy((X - mu) / sd).float().to(device),
             int(won.argmax())) for X, _o, won in train]
    rng = np.random.default_rng(0)
    for _ep in range(args.epochs):
        net.train()
        opt.zero_grad()
        acc = torch.zeros((), device=device)
        for j, k in enumerate(rng.permutation(len(tr_t))):
            X, wi = tr_t[k]
            acc = acc - torch.log_softmax(net(X).squeeze(-1), dim=0)[wi]
            if (j + 1) % 256 == 0 or j == len(tr_t) - 1:
                (acc / 256).backward()
                opt.step()
                opt.zero_grad()
                acc = torch.zeros((), device=device)

    print("\n=== 単勝 value-bet ROI (no-odds models vs market), 2yr OOS ===")
    report("GBDT (LightGBM)", race_winprobs_gbdt(booster, oos, mu, sd))
    report("NN (MLP cond-logit)", race_winprobs_mlp(net, oos, mu, sd, device))
    # favorite reference
    fav = [(np.where(o == o.min(), 1.0, 0.0), o, won) for _X, o, won in oos]
    report("favorite (市場)", [(p / p.sum(), o, won) for p, o, won in fav])


if __name__ == "__main__":
    main()
