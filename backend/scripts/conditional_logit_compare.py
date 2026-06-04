"""Is linear OK? — linear vs non-linear conditional-logit on identical inputs.

The value of the conditional-logit (Plackett-Luce) approach is the per-race
SOFTMAX STRUCTURE — proper win probabilities that sum to 1 within a race and
match the betting structure — NOT the linearity of the score function. The score
function is swappable. This isolates exactly the user's question by holding the
structure + features fixed and varying only the score function:

  - linear : score_i = w·x_i            (classic conditional logit, interpretable)
  - mlp    : score_i = MLP(x_i)          (non-linear conditional logit)

Both trained with per-race softmax cross-entropy (rank-1 Plackett-Luce) on the
SAME no-odds features and the SAME races. We also report the favorite (lowest
odds) top1 on the same races as the market reference. Metric = top1_hit (the
model's #1 pick won; == binary ndcg@1), directly comparable across all rows.

Controlled 2-year split; OOS held out. Run with KEIBA_EXCLUDE_ODDS_FEATURES=1.
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


def linear_net(f):
    return nn.Linear(f, 1)


def mlp_net(f, h=128):
    return nn.Sequential(
        nn.Linear(f, h), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 1),
    )


def split_races(frame, feats):
    out = {}
    for name, lo, hi, inc in [("train", "0000", TRAIN_END, False),
                              ("valid", TRAIN_END, VALID_END, False),
                              ("oos", OOS_START, OOS_END, True)]:
        m = (frame["date"] >= lo) & ((frame["date"] <= hi) if inc else (frame["date"] < hi))
        races, feat_used = build_races(frame[m], feats)
        out[name] = races
        out["_feat"] = feat_used
    return out


@torch.no_grad()
def top1(net, races, mu, sd, device):
    hits = []
    for X, _o, won in races:
        s = net(torch.from_numpy((X - mu) / sd).float().to(device)).squeeze(-1)
        hits.append(1.0 if int(s.argmax()) == int(won.argmax()) else 0.0)
    return float(np.mean(hits)), len(races)


def favorite_top1(races):
    hits = [1.0 if int(np.argmin(o)) == int(won.argmax()) else 0.0 for _X, o, won in races]
    return float(np.mean(hits)), len(races)


def train_eval(make_net, sp, mu, sd, device, epochs, lr, tag):
    tr = [(torch.from_numpy((X - mu) / sd).float().to(device),
           torch.from_numpy(won).float().to(device)) for X, _o, won in sp["train"]]
    net = make_net().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    rng = np.random.default_rng(0)
    best_v, best_oos = -1.0, 0.0
    for _ep in range(epochs):
        net.train()
        order = rng.permutation(len(tr))
        opt.zero_grad()
        acc = torch.zeros((), device=device)
        for j, k in enumerate(order):
            X, won = tr[k]
            s = net(X).squeeze(-1)
            acc = acc - torch.log_softmax(s, dim=0)[int(won.argmax())]
            if (j + 1) % 256 == 0 or j == len(order) - 1:
                (acc / 256).backward()
                opt.step()
                opt.zero_grad()
                acc = torch.zeros((), device=device)
        v, _ = top1(net, sp["valid"], mu, sd, device)
        if v > best_v:
            best_v = v
            best_oos, _ = top1(net, sp["oos"], mu, sd, device)
    print(f"  {tag:<10} valid_top1={best_v:.4f}  OOS_top1={best_oos:.4f}")
    return best_oos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    feats = get_active_features()
    engine = make_engine(db_path())
    with session_scope(engine) as s:
        frame = build_training_frame(s, train_start="2015-01-01", train_end=OOS_END)
    sp = split_races(frame, feats)
    mu = np.concatenate([r[0] for r in sp["train"]]).mean(0)
    sd = np.concatenate([r[0] for r in sp["train"]]).std(0) + 1e-6
    print(f"races: train={len(sp['train'])} valid={len(sp['valid'])} oos={len(sp['oos'])} "
          f"| no-odds features={len(sp['_feat'])} device={device}")

    print("\n=== conditional-logit (per-race softmax-CE), no-odds, 2yr OOS — top1_hit ===")
    f = len(sp["_feat"])
    train_eval(lambda: linear_net(f), sp, mu, sd, device, args.epochs, args.lr, "linear")
    train_eval(lambda: mlp_net(f), sp, mu, sd, device, args.epochs, args.lr, "MLP(非線形)")
    fav, n = favorite_top1(sp["oos"])
    print(f"  {'favorite':<10} OOS_top1={fav:.4f}  (市場, 同一{n}レース)")
    print("  ref: no-odds GBDT ~0.27, no-odds GRU(履歴系列) ~0.29 (別 split/subset)")


if __name__ == "__main__":
    main()
