"""Per-feature influence on the MLP (and GBDT) via permutation importance.

NNs have no TreeSHAP, so we use model-agnostic permutation importance: shuffle one
feature's values across all OOS horses and measure how much the per-race winner
log-loss (CE = -log p[winner], the conditional-logit objective) gets WORSE. Bigger
increase = the model relied on that feature more. Reported as % of total importance,
averaged over a few shuffles, for the MLP and a GBDT trained on the SAME 38 no-odds
numeric features — so we can see whether NN and tree lean on the same inputs.

Controlled 2-year split (train<2023-11). Run with KEIBA_EXCLUDE_ODDS_FEATURES=1.
"""

from __future__ import annotations

import argparse

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


def race_ce(probs_per_race, wins_per_race):
    """Mean over races of -log p[winner]."""
    ce = [-np.log(p[int(w.argmax())] + 1e-9) for p, w in zip(probs_per_race, wins_per_race, strict=True)]
    return float(np.mean(ce))


def mlp_probs(net, Xs, device):
    out = []
    with torch.no_grad():
        for X in Xs:
            s = net(torch.from_numpy(X).float().to(device)).squeeze(-1)
            out.append(torch.softmax(s, dim=0).cpu().numpy())
    return out


def gbdt_probs(booster, Xs):
    out = []
    for X in Xs:
        raw = booster.predict(X)
        out.append(raw / raw.sum() if raw.sum() > 0 else np.full(len(raw), 1 / len(raw)))
    return out


def perm_importance(prob_fn, Xs_norm, wins, feats, n_rep=3, seed=0):
    """ΔCE when each feature column is shuffled across all OOS horses."""
    base = race_ce(prob_fn(Xs_norm), wins)
    # flatten index map: which (race, row) each global row is
    sizes = [X.shape[0] for X in Xs_norm]
    flat = np.concatenate(Xs_norm, axis=0)
    rng = np.random.default_rng(seed)
    imp = np.zeros(len(feats))
    for j in range(len(feats)):
        deltas = []
        for _ in range(n_rep):
            perm = flat.copy()
            perm[:, j] = flat[rng.permutation(flat.shape[0]), j]
            # re-split into races
            Xs_p, off = [], 0
            for n in sizes:
                Xs_p.append(perm[off:off + n])
                off += n
            deltas.append(race_ce(prob_fn(Xs_p), wins) - base)
        imp[j] = max(0.0, float(np.mean(deltas)))
    return base, imp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    feats_all = get_active_features()
    engine = make_engine(db_path())
    with session_scope(engine) as s:
        frame = build_training_frame(s, train_start="2015-01-01", train_end=OOS_END)

    def races_of(lo, hi, inc):
        m = (frame["date"] >= lo) & ((frame["date"] <= hi) if inc else (frame["date"] < hi))
        return build_races(frame[m], feats_all)

    train, feats = races_of("0000", TRAIN_END, False)
    valid, _ = races_of(TRAIN_END, VALID_END, False)
    oos, _ = races_of(OOS_START, OOS_END, True)
    mu = np.concatenate([r[0] for r in train]).mean(0)
    sd = np.concatenate([r[0] for r in train]).std(0) + 1e-6
    Xtr_n = [(r[0] - mu) / sd for r in train]
    oos_X = [(r[0] - mu) / sd for r in oos]
    oos_w = [r[2] for r in oos]
    print(f"OOS races={len(oos)} | features={len(feats)}")

    # GBDT (binary)
    booster = lgb.train(
        {"objective": "binary", "num_leaves": 127, "learning_rate": 0.03,
         "min_data_in_leaf": 50, "feature_fraction": 0.85, "bagging_fraction": 0.85,
         "bagging_freq": 5, "verbose": -1},
        lgb.Dataset(np.concatenate(Xtr_n), label=np.concatenate([r[2] for r in train])),
        num_boost_round=600,
    )

    # MLP conditional logit
    net = nn.Sequential(
        nn.Linear(len(feats), 128), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    tr_t = [(torch.from_numpy(X).float().to(device), int(r[2].argmax()))
            for X, r in zip(Xtr_n, train, strict=True)]
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

    base_m, imp_m = perm_importance(lambda Xs: mlp_probs(net, Xs, device), oos_X, oos_w, feats)
    base_g, imp_g = perm_importance(lambda Xs: gbdt_probs(booster, Xs), oos_X, oos_w, feats)
    pm = 100 * imp_m / (imp_m.sum() or 1)
    pg = 100 * imp_g / (imp_g.sum() or 1)

    order = np.argsort(-pm)
    print(f"\n=== permutation importance (% of total ΔCE); base CE: MLP {base_m:.3f} GBDT {base_g:.3f} ===")
    print(f"{'feature':<32}{'MLP%':>8}{'GBDT%':>8}")
    for i in order:
        print(f"{feats[i]:<32}{pm[i]:>8.2f}{pg[i]:>8.2f}")


if __name__ == "__main__":
    main()
