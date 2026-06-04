"""Two-stage betting architecture: ability network (no odds) -> value network (odds).

The user's ideal design, built and tested honestly:

  Stage A (ability, NO odds): race/horse features -> per-horse ability score a_i.
      Grounded by an auxiliary outcome loss so a_i is a real win-ability estimate
      (the model can't just copy the market — it never sees odds).
  Stage B (value, odds enter HERE): (ability prob p_i, odds O_i, 1/O_i, EV=p_i*O_i)
      -> bet logit -> soft bet weight w_i. This is where "is it a good BUY?" lives.

End-to-end objective = realised betting ROI (differentiable win-pool profit using
odds_win + outcome) + λ * ability cross-entropy. So the network is trained to emit
buy-worthy bets, with Stage A kept a meaningful ability score.

Honest expectation: Stage A is built from a SUBSET of the market's information, so
its disagreements with the odds are net-negative -> OOS ROI < 1 (same information
ceiling as the single-stage ROI-objective, 0.83). Value here = correct architecture,
interpretable over/under-valued signal, and a definitive test — not edge.

Controlled 2-year split (matches the reconfirmed verdict):
  train: date < 2023-11-01   valid: 2023-11..2024-05   OOS: 2024-05-01..2026-04-30
Win pool only (odds_win in keiba.db, no odds.db needed, no look-ahead).
Run with KEIBA_EXCLUDE_ODDS_FEATURES=1.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
from roi_objective_experiment import build_races  # same dir (scripts/)

from core.paths import db_path
from db.session import make_engine, session_scope
from features.builder import build_training_frame, get_active_features

TRAIN_END, VALID_END = "2023-11-01", "2024-05-01"
OOS_START, OOS_END = "2024-05-01", "2026-04-30"


class TwoStage(nn.Module):
    def __init__(self, f: int, ha: int = 128, hb: int = 64):
        super().__init__()
        # Stage A: ability from no-odds features only.
        self.ability = nn.Sequential(
            nn.Linear(f, ha), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(ha, ha), nn.ReLU(), nn.Dropout(0.1), nn.Linear(ha, 1),
        )
        # Stage B: value/buy decision from (ability prob, odds, 1/odds, EV).
        self.value = nn.Sequential(
            nn.Linear(4, hb), nn.ReLU(), nn.Linear(hb, 1),
        )

    def race(self, X: torch.Tensor, odds: torch.Tensor):
        """One race. Returns (p ability-win-prob [n], w soft-bet [n], a [n])."""
        a = self.ability(X).squeeze(-1)               # [n] ability score
        p = torch.softmax(a, dim=0)                   # [n] ability win-prob (no odds)
        ev = p * odds
        b_in = torch.stack([p, torch.log(odds), 1.0 / odds, ev - 1.0], dim=1)  # [n,4]
        w = torch.sigmoid(self.value(b_in).squeeze(-1))  # [n] buy weight in (0,1)
        return p, w, a


def load_split(device):
    feats = get_active_features()
    eng = make_engine(db_path())
    with session_scope(eng) as s:
        frame = build_training_frame(s, train_start="2015-01-01", train_end=OOS_END)
    splits = {}
    for name, lo, hi in [("train", "0000", TRAIN_END), ("valid", TRAIN_END, VALID_END),
                         ("oos", OOS_START, OOS_END)]:
        sub = frame[(frame["date"] >= lo) & (frame["date"] < hi)] if name != "oos" \
            else frame[(frame["date"] >= lo) & (frame["date"] <= hi)]
        races, feat_used = build_races(sub, feats)
        splits[name] = races
        splits["_feat"] = feat_used
    return splits


def _norm_stats(races):
    allX = np.concatenate([r[0] for r in races], axis=0)
    return allX.mean(0), allX.std(0) + 1e-6


def _to_tensors(races, mu, sd, device):
    out = []
    for X, o, w in races:
        out.append((
            torch.from_numpy((X - mu) / sd).float().to(device),
            torch.from_numpy(o).float().to(device),
            torch.from_numpy(w).float().to(device),
        ))
    return out


@torch.no_grad()
def hard_roi(model, races, device, mu, sd):
    """ROI of the policy 'buy horse i when Stage-B weight w_i > 0.5'.

    Unit stake per bet; win-pool payout = odds_i if horse i won else 0.
    """
    stake = 0.0
    payout = 0.0
    per_race = []
    for X, o, won in races:
        Xt = torch.from_numpy((X - mu) / sd).float().to(device)
        ot = torch.from_numpy(o).float().to(device)
        _p, w, _a = model.race(Xt, ot)
        bets = w.cpu().numpy() > 0.5
        st = float(bets.sum())
        if st == 0:
            continue
        pay = float((o * won)[bets].sum())
        stake += st
        payout += pay
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
    ap.add_argument("--lam", type=float, default=1.0, help="ability-loss weight")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    sp = load_split(device)
    feat_used = sp["_feat"]
    mu, sd = _norm_stats(sp["train"])
    tr = _to_tensors(sp["train"], mu, sd, device)
    print(f"races: train={len(sp['train'])} valid={len(sp['valid'])} oos={len(sp['oos'])} "
          f"| features={len(feat_used)} (no-odds) device={device}")

    model = TwoStage(len(feat_used)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    rng = np.random.default_rng(0)
    print("\nepoch | bet_loss abil_loss | train_ROI valid_ROI  oos_ROI")
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(tr))
        opt.zero_grad()
        bet_acc = torch.zeros((), device=device)
        abil_acc = torch.zeros((), device=device)
        bl = al = 0.0
        for j, k in enumerate(order):
            X, o, won = tr[k]
            p, w, _a = model.race(X, o)
            profit = (w * (o * won - 1.0)).sum()        # realised profit (maximise)
            ce = -torch.log(p[won.argmax()] + 1e-9)     # ability grounding (winner)
            bet_acc = bet_acc - profit
            abil_acc = abil_acc + ce
            if (j + 1) % 256 == 0 or j == len(order) - 1:
                loss = (bet_acc + args.lam * abil_acc) / 256
                loss.backward()
                opt.step()
                opt.zero_grad()
                bl += float(bet_acc)
                al += float(abil_acc)
                bet_acc = torch.zeros((), device=device)
                abil_acc = torch.zeros((), device=device)
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            tr_roi, _, _ = hard_roi(model, sp["train"], device, mu, sd)
            va_roi, _, _ = hard_roi(model, sp["valid"], device, mu, sd)
            oo_roi, _, _ = hard_roi(model, sp["oos"], device, mu, sd)
            print(f"{ep+1:5d} | {bl/len(tr):8.3f} {al/len(tr):8.3f} | "
                  f"{tr_roi:8.3f} {va_roi:8.3f} {oo_roi:8.3f}")

    _, oo_st, _ = hard_roi(model, sp["oos"], device, mu, sd)
    print("\n=== two-stage (Stage A no-odds ability -> Stage B value) win-pool ===")
    print(f"  learned policy (buy when Stage-B w>0.5): OOS bets={int(oo_st)} "
          f"{'(= abstains: no +EV bet exists)' if oo_st == 0 else ''}")
    print("\n  ROI of the model's MOST buy-worthy bets (top-w% over OOS):")
    roi_sweep(model, sp["oos"], mu, sd, device)
    print("  reference: single-stage ROI-objective OOS 0.832; favorite-win ~0.78; break-even 0.80")


@torch.no_grad()
def roi_sweep(model, races, mu, sd, device):
    """ROI of betting the highest Stage-B-weight horses (top-w% across OOS)."""
    from collections import defaultdict
    rows = []  # (w, odds, won, race_idx)
    for ri, (X, o, won) in enumerate(races):
        Xt = torch.from_numpy((X - mu) / sd).float().to(device)
        ot = torch.from_numpy(o).float().to(device)
        _p, w, _a = model.race(Xt, ot)
        for k, wv in enumerate(w.cpu().numpy()):
            rows.append((float(wv), float(o[k]), float(won[k]), ri))
    rows.sort(key=lambda r: -r[0])
    n = len(rows)
    print(f"    {'top-w%':>7} {'n_bets':>7} {'ROI':>7} {'95% CI':>16} {'avgOdds':>8} {'meanW':>7}")
    for pct in (0.5, 1, 2, 5, 10):
        sel = rows[: int(n * pct / 100)]
        per_race: dict = defaultdict(lambda: [0.0, 0.0])
        for _w, o, won, ri in sel:
            per_race[ri][0] += 1.0
            per_race[ri][1] += o * won
        roi = sum(p for _, p in per_race.values()) / sum(s for s, _ in per_race.values())
        lo, hi = boot_ci(list(per_race.values()))
        avgo = float(np.mean([o for _, o, _, _ in sel]))
        mw = float(np.mean([w for w, _, _, _ in sel]))
        print(f"    {pct:>6.1f}% {len(sel):>7} {roi:>7.3f} [{lo:>5.2f},{hi:>5.2f}] {avgo:>8.1f} {mw:>7.3f}")


if __name__ == "__main__":
    main()
