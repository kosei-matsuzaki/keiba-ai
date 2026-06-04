"""Does the AGGREGATION of existing history lose signal? — A/B feature variants.

Same raw per-past-race data (finish, agari_3f, margin, passing, class, won), but
built into the MLP in DIFFERENT ways, to test the user's hypothesis that simple
averages throw information away. Each variant -> identical conditional-logit MLP
-> OOS top1_hit / CE on the same 2-year split. Holding model + data fixed and
varying only the feature representation isolates the effect of the aggregation.

Variants (history part; a shared no-odds context vector is always appended):
  V0_mean        : mean of last-K (the current style — baseline)
  V1_recencyW    : exponential-recency-weighted mean (recent races weighted more)
  V2_rawlags     : last-K races kept as SEPARATE features (no aggregation)
  V3_mean_std_tr : mean + std + linear trend (consistency / improving-declining)
  V4_mean_peak   : mean + career-best (peak agari / best finish) + win count
  V5_kitchensink : V1 + V2 + V3 + V4 combined

Leak-safe: only races strictly before the target date. 2-year split
(train<2023-11, OOS 2024-05..2026-04). GPU. No odds anywhere.
"""

from __future__ import annotations

import argparse
import sqlite3
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from core.paths import db_path

TRAIN_END, VALID_END = "2023-11-01", "2024-05-01"
OOS_START, OOS_END = "2024-05-01", "2026-04-30"
K = 10  # most recent K past races
_CLASS = {"新馬": 1, "未勝利": 1, "1勝クラス": 2, "2勝クラス": 3, "3勝クラス": 4,
          "OP": 5, "Listed": 5, "重賞": 6, "G3": 6, "G2": 7, "G1": 8}


def _passing_first(p):
    if not p:
        return np.nan
    try:
        return float(p.split("-")[0])
    except (ValueError, IndexError):
        return np.nan


def _margin(m):
    if not m:
        return np.nan
    s = {"同着": 0.0, "ハナ": 0.05, "アタマ": 0.1, "クビ": 0.2, "大差": 10.0}.get(m.strip())
    if s is not None:
        return s
    try:
        return float(m.split()[0].split("/")[0].replace("+", ""))
    except (ValueError, IndexError):
        return np.nan


def load_samples(db: str):
    con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    rows = con.execute(
        "SELECT e.horse_id,e.race_id,r.date,e.finish_position,e.agari_3f,e.margin,"
        "e.passing,e.weight_carried,e.age,e.post_position,r.distance,r.race_class,r.n_runners "
        "FROM entries e JOIN races r ON r.race_id=e.race_id "
        "WHERE e.finish_position IS NOT NULL AND r.date IS NOT NULL "
        "ORDER BY e.horse_id,r.date,e.race_id"
    ).fetchall()
    con.close()
    field = defaultdict(int)
    for r in rows:
        field[r[1]] += 1
    by_horse = defaultdict(list)
    for r in rows:
        by_horse[r[0]].append(r)

    samples = {"train": [], "valid": [], "oos": []}
    for _hid, hist in by_horse.items():
        past = []  # list of dicts (most-recent-LAST)
        for i, r in enumerate(hist):
            (_h, rid, dte, fin, ag, mg, pa, wc, age, pp, dist, rcls, nr) = r
            fs = field[rid]
            if i >= 1:
                window = past[-K:][::-1]  # most-recent-first
                ctx = [float(dist or 0), float(_CLASS.get(rcls, 0)), float(age or 0),
                       float(fs), float(pp or 0), float(wc or 0), float(i),
                       float((np.datetime64(dte) - np.datetime64(past[-1]["d"]))
                             .astype("timedelta64[D]").astype(float))]
                smp = {"rid": rid, "win": 1.0 if fin == 1 else 0.0,
                       "ctx": np.array(ctx, np.float32), "hist": window}
                if dte < TRAIN_END:
                    samples["train"].append(smp)
                elif dte < VALID_END:
                    samples["valid"].append(smp)
                elif OOS_START <= dte <= OOS_END:
                    samples["oos"].append(smp)
            past.append({"d": dte, "fn": (fin / fs) if fs else np.nan, "ag": ag,
                         "mg": _margin(mg), "pf": (_passing_first(pa) / fs) if fs else np.nan,
                         "cw": _CLASS.get(rcls, 0), "wn": 1.0 if fin == 1 else 0.0})
    return samples


# ── feature variant builders: (window list, ctx) -> history feature vector ──
_FIELDS = ["fn", "ag", "mg", "pf", "cw", "wn"]


def _col(window, f):
    return np.array([w[f] for w in window], dtype=np.float64)


def _nanmean(a):
    return np.nan if len(a) == 0 or np.all(np.isnan(a)) else float(np.nanmean(a))


def v0_mean(window):
    return [_nanmean(_col(window, f)) for f in _FIELDS]


def v1_recency(window):
    n = len(window)
    if n == 0:
        return [np.nan] * len(_FIELDS)
    w = np.exp(-0.35 * np.arange(n))  # most-recent-first -> decay
    out = []
    for f in _FIELDS:
        a = _col(window, f)
        ok = ~np.isnan(a)
        out.append(float(np.sum(a[ok] * w[ok]) / np.sum(w[ok])) if ok.any() else np.nan)
    return out


def v2_rawlags(window):
    out = []
    for f in ("fn", "ag", "pf"):
        a = _col(window, f)
        for k in range(K):
            out.append(float(a[k]) if k < len(a) else np.nan)
    return out


def _trend(a):
    ok = ~np.isnan(a)
    if ok.sum() < 2:
        return 0.0
    x = np.arange(len(a))[ok]
    return float(np.polyfit(x, a[ok], 1)[0])


def v3_mean_std_trend(window):
    out = []
    for f in ("fn", "ag", "pf"):
        a = _col(window, f)
        out += [_nanmean(a), float(np.nanstd(a)) if (~np.isnan(a)).any() else np.nan, _trend(a)]
    return out


def v4_mean_peak(window):
    fn, ag = _col(window, "fn"), _col(window, "ag")
    peak = [np.nanmin(ag) if (~np.isnan(ag)).any() else np.nan,
            np.nanmin(fn) if (~np.isnan(fn)).any() else np.nan,
            float(np.nansum(_col(window, "wn")))]
    return v0_mean(window) + peak


def v5_kitchensink(window):
    return v1_recency(window) + v2_rawlags(window) + v3_mean_std_trend(window) + v4_mean_peak(window)


VARIANTS = {
    "V0_mean": v0_mean, "V1_recencyW": v1_recency, "V2_rawlags": v2_rawlags,
    "V3_mean_std_tr": v3_mean_std_trend, "V4_mean_peak": v4_mean_peak,
    "V5_kitchensink": v5_kitchensink,
}


def build_matrix(samples, builder):
    """Return per-split list of races; race = (X[n,F], win_idx)."""
    out = {}
    for split in ("train", "valid", "oos"):
        by_race = defaultdict(list)
        for s in samples[split]:
            feat = np.array(builder(s["hist"]), np.float32)
            x = np.concatenate([feat, s["ctx"]])
            by_race[s["rid"]].append((x, s["win"]))
        races = []
        for items in by_race.values():
            if len(items) < 2 or not any(w == 1.0 for _x, w in items):
                continue
            X = np.nan_to_num(np.stack([x for x, _w in items]), nan=0.0)
            wi = next(k for k, (_x, w) in enumerate(items) if w == 1.0)
            races.append((X, wi))
        out[split] = races
    return out


def train_eval(races, device, epochs, seed=0):
    allX = np.concatenate([r[0] for r in races["train"]])
    mu, sd = allX.mean(0), allX.std(0) + 1e-6
    f = allX.shape[1]
    net = nn.Sequential(nn.Linear(f, 128), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(128, 1)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    tr = [(torch.from_numpy((X - mu) / sd).float().to(device), wi) for X, wi in races["train"]]

    def top1(split):
        net.eval()
        hits = []
        with torch.no_grad():
            for X, wi in races[split]:
                s = net(torch.from_numpy((X - mu) / sd).float().to(device)).squeeze(-1)
                hits.append(1.0 if int(s.argmax()) == wi else 0.0)
        return float(np.mean(hits))

    rng = np.random.default_rng(seed)
    best_v, best_o = -1.0, 0.0
    for _ep in range(epochs):
        net.train()
        opt.zero_grad()
        acc = torch.zeros((), device=device)
        for j, k in enumerate(rng.permutation(len(tr))):
            X, wi = tr[k]
            acc = acc - torch.log_softmax(net(X).squeeze(-1), dim=0)[wi]
            if (j + 1) % 256 == 0 or j == len(tr) - 1:
                (acc / 256).backward()
                opt.step()
                opt.zero_grad()
                acc = torch.zeros((), device=device)
        v = top1("valid")
        if v > best_v:
            best_v, best_o = v, top1("oos")
    return f, best_o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    samples = load_samples(str(db_path()))
    print(f"samples: train={len(samples['train'])} valid={len(samples['valid'])} "
          f"oos={len(samples['oos'])} (load {time.time()-t0:.0f}s)")
    print("\n=== feature-aggregation A/B (no-odds MLP, 2yr OOS top1_hit) ===")
    print(f"{'variant':<16}{'dims':>6}{'OOS_top1':>10}")
    for name, builder in VARIANTS.items():
        races = build_matrix(samples, builder)
        f, oos = train_eval(races, device, args.epochs)
        print(f"{name:<16}{f:>6}{oos:>10.4f}")
    print("  (ctx 8 dims appended to every variant; favorite~0.33, GBDT/MLP aggregates~0.27-0.28)")


if __name__ == "__main__":
    main()
