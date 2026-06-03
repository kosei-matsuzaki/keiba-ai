"""Train a model whose loss IS betting ROI, and test whether it generalises.

The user's claim: ROI should be the training objective, not decoupled from it.
This implements exactly that — a no-odds MLP trained with a DIFFERENTIABLE
betting objective: per race, scores -> softmax win-probs p_i; EV_i = p_i * O_i
(O_i = market win odds); a smooth bet weight w_i = sigmoid(k*(EV_i - 1)) lets the
model choose to bet; realised profit = Σ_i w_i * (O_i*won_i - 1). The loss is
-mean(profit), so SGD directly maximises realised betting return on the data.

Odds are used ONLY in the loss (and to bet), never as features — so the model
must learn a market-INDEPENDENT ability that finds horses the market mispriced.

Why this is the decisive test of "train on ROI": if the market is inefficient,
this objective will find real edge that generalises. If it is efficient, the
population gradient of expected ROI is ~0 and the objective can only fit
in-sample noise -> train ROI > 1 but OOS ROI < 1 (overfitting to a zero-signal
target). We report hard-policy ROI (bet when EV>1) on train / valid / OOS with
a race-level bootstrap CI on OOS.

Controlled split identical to the other experiments; OOS held out.
Run with KEIBA_EXCLUDE_ODDS_FEATURES=1.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from core.paths import db_path
from db.session import make_engine, session_scope
from features.builder import build_training_frame, get_active_features

TRAIN_END, VALID_END = "2025-01-05", "2025-07-05"
OOS_START, OOS_END = "2025-10-01", "2026-04-30"


def build_races(frame, feats):
    """List of races -> (X[n,F] float32, odds[n], won[n]). Drops races missing odds."""
    races = []
    num = frame[feats].select_dtypes(include="number").columns.tolist()
    feat_used = num  # numeric-only for the bare MLP (categoricals dropped for simplicity)
    for _rid, g in frame.groupby("race_id"):
        if len(g) < 2:
            continue
        odds = g["odds_win"].to_numpy(dtype=np.float64)
        won = (g["finish_position"].to_numpy() == 1).astype(np.float64)
        if np.isnan(odds).any() or won.sum() != 1:
            continue
        X = np.nan_to_num(g[feat_used].to_numpy(dtype=np.float64), nan=0.0)
        races.append((X.astype(np.float32), odds.astype(np.float32), won.astype(np.float32)))
    return races, feat_used


class MLP(nn.Module):
    def __init__(self, f, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(f, h), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 1),
        )

    def forward(self, x):  # x [n, F] -> [n]
        return self.net(x).squeeze(-1)


def race_profit(scores, odds, won, k=8.0):
    """Differentiable realised profit for one race. softmax win-prob -> soft bet."""
    p = torch.softmax(scores, dim=0)
    ev = p * odds
    w = torch.sigmoid(k * (ev - 1.0))          # smooth bet indicator
    return (w * (odds * won - 1.0)).sum()       # realised profit (unit stakes)


@torch.no_grad()
def hard_roi(model, races, device):
    """ROI of the hard policy: stake 1 on every horse with EV = p*odds > 1."""
    stake = 0.0
    payout = 0.0
    per_race = []
    for X, odds, won in races:
        s = model(torch.from_numpy(X).to(device))
        p = torch.softmax(s, dim=0).cpu().numpy()
        ev = p * odds
        bets = ev > 1.0
        st = float(bets.sum())
        pay = float((odds * won * bets).sum())
        stake += st
        payout += pay
        if st > 0:
            per_race.append((st, pay))
    roi = payout / stake if stake > 0 else float("nan")
    return roi, stake, per_race


def boot_ci(per_race, n_boot=2000, seed=0):
    if not per_race:
        return float("nan"), float("nan")
    arr = np.array(per_race, dtype=np.float64)
    rng = np.random.default_rng(seed)
    b = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(arr), len(arr))
        s = arr[idx, 0].sum()
        b[i] = arr[idx, 1].sum() / s if s > 0 else 0.0
    return tuple(np.percentile(b, [2.5, 97.5]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    feats = get_active_features()
    engine = make_engine(db_path())
    with session_scope(engine) as s:
        frame = build_training_frame(s, train_start="2015-01-01", train_end=OOS_END)
    print(f"no-odds features: {len(feats)} | odds_win in features: {'odds_win' in feats}")

    tr_f = frame[frame["date"] < TRAIN_END]
    va_f = frame[(frame["date"] >= TRAIN_END) & (frame["date"] < VALID_END)]
    oos_f = frame[(frame["date"] >= OOS_START) & (frame["date"] <= OOS_END)]
    train, feat_used = build_races(tr_f, feats)
    valid, _ = build_races(va_f, feats)
    oos, _ = build_races(oos_f, feats)
    print(f"races: train={len(train)} valid={len(valid)} oos={len(oos)} | model_features={len(feat_used)}")

    # standardize features using train stats
    allX = np.concatenate([r[0] for r in train], axis=0)
    mu = allX.mean(0)
    sd = allX.std(0) + 1e-6
    def norm(races):
        return [((X - mu) / sd, o, w) for X, o, w in races]
    train, valid, oos = norm(train), norm(valid), norm(oos)
    tr_t = [(torch.from_numpy(X).float().to(device),
             torch.from_numpy(o).float().to(device),
             torch.from_numpy(w).float().to(device)) for X, o, w in train]

    model = MLP(len(feat_used)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    rng = np.random.default_rng(0)
    print("\nepoch |  train_profit  train_ROI  valid_ROI   oos_ROI")
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(tr_t))
        opt.zero_grad()
        total = torch.zeros((), device=device)
        for j, k in enumerate(order):
            X, o, w = tr_t[k]
            total = total - race_profit(model(X), o, w)
            if (j + 1) % 256 == 0 or j == len(order) - 1:
                (total / 256).backward()
                opt.step()
                opt.zero_grad()
                total = torch.zeros((), device=device)
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            tr_roi, _, _ = hard_roi(model, train, device)
            va_roi, _, _ = hard_roi(model, valid, device)
            oo_roi, _, oos_pr = hard_roi(model, oos, device)
            # train_profit proxy = (train_roi-1)*stake; just show ROIs
            print(f"{ep+1:5d} |        --      {tr_roi:8.3f}  {va_roi:8.3f}  {oo_roi:8.3f}")

    tr_roi, tr_stake, _ = hard_roi(model, train, device)
    oo_roi, oo_stake, oos_pr = hard_roi(model, oos, device)
    lo, hi = boot_ci(oos_pr)
    print("\n=== ROI of the EV>1 policy from the ROI-trained model ===")
    print(f"  TRAIN ROI = {tr_roi:.3f}  (bets={int(tr_stake)})   <- objective maximises this")
    print(f"  OOS   ROI = {oo_roi:.3f}  (bets={int(oo_stake)})   95% CI [{lo:.3f}, {hi:.3f}]")
    print("  hypothesis check: train>1 & OOS<1  =>  overfit to a zero-signal target (efficient market)")


if __name__ == "__main__":
    main()
