"""Option C experiment: per-horse RACE-HISTORY sequence model (no-odds).

Hypothesis: a model that reads a horse's raw past-race *sequence* (finish, agari,
margin, pace, class, distance, days-since...) can extract signal that the
hand-aggregated features (recent_avg_finish etc.) throw away. Trained WITHOUT
odds features so it is a market-independent "ability" model.

Self-contained (does NOT touch the production bundle/registry/inference path):
builds its own leak-safe sequence dataset from the DB, trains a GRU history
encoder + context MLP with a per-race rank-1 (Plackett-Luce) softmax loss on
GPU, and reports OOS ndcg@1 / top1-hit on the SAME controlled split used for the
GBDT anchor — so the comparison is apples-to-apples.

Controlled split (identical to the GBDT experiment):
    train : date <  2025-01-05
    valid : 2025-01-05 .. 2025-07-05   (early stopping)
    oos   : 2025-10-01 .. 2026-04-30   (held out from everything)

Fairness: no odds anywhere; same races/labels as the GBDT eval; ndcg@1 computed
identically; OOS never seen in training or model selection.

Usage:
    uv run python scripts/seq_experiment.py --epochs 12 --device cuda
    uv run python scripts/seq_experiment.py --limit-train-races 3000 --epochs 2   # smoke
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ndcg_score

from core.paths import db_path

# KEIBA_SEQ_WITH_ODDS=1 appends [log(odds_win), 1/odds_win] to the race context,
# turning the no-odds GRU into a with-odds GRU (to test if the history sequence
# adds anything ON TOP of the market odds vs the with-odds GBDT ~0.33).
_WITH_ODDS = os.environ.get("KEIBA_SEQ_WITH_ODDS", "0") == "1"

TRAIN_END = "2023-11-01"   # 2-year OOS split (NN direction, matches GBDT verifications)
VALID_END = "2024-05-01"
OOS_START = "2024-05-01"
OOS_END = "2026-04-30"

MAX_HIST = 15  # most recent N past races per horse (career median 6, p90 21)

_SURFACES = ["芝", "ダ", "障"]
# coarse class ordering (higher = better company); unknown -> 0
_CLASS_RANK = {
    "新馬": 1, "未勝利": 1, "1勝クラス": 2, "2勝クラス": 3, "3勝クラス": 4,
    "OP": 5, "Listed": 5, "重賞": 6, "G3": 6, "G2": 7, "G1": 8,
}


def _passing_first(passing: str | None) -> float:
    """First corner position from a '3-3-2-1' style string. NaN if absent."""
    if not passing:
        return np.nan
    head = passing.split("-")[0].strip()
    try:
        return float(head)
    except ValueError:
        return np.nan


def _margin_num(margin: str | None) -> float:
    """Rough numeric margin. 0 for win/'クビ'/'ハナ' etc.; NaN if unknown."""
    if margin is None or margin == "":
        return np.nan
    m = margin.strip()
    small = {"同着": 0.0, "ハナ": 0.05, "アタマ": 0.1, "クビ": 0.2, "大差": 10.0}
    if m in small:
        return small[m]
    # leading numeric like "1.1/2" -> just take leading float-ish
    try:
        return float(m.split()[0].split("/")[0].replace("+", ""))
    except (ValueError, IndexError):
        return np.nan


# Per-past-race token features (NO odds). Order matters; keep in sync with H below.
HIST_NUM = [
    "finish_norm",     # finish_position / field_size  (∈(0,1], lower=better)
    "field_size",
    "agari_3f",
    "margin",
    "passing_first",
    "weight_carried",
    "horse_weight",
    "distance",
    "class_rank",
    "days_since_prev",
    "won",
]
SURF_DIM = len(_SURFACES)
H = len(HIST_NUM) + SURF_DIM  # token feature dim

# Current-race context (NO odds).
CTX_NUM = [
    "distance", "class_rank", "field_size", "age", "weight_carried",
    "days_since_last", "career_len", "post_position",
]
CTX_DIM = len(CTX_NUM) + SURF_DIM + (2 if _WITH_ODDS else 0)


def _surf_onehot(surface: str | None) -> list[float]:
    return [1.0 if surface == s else 0.0 for s in _SURFACES]


def load_history_and_samples(db: str, limit_train_races: int | None):
    """Load all entries, build per-horse chronological history, and emit samples.

    Returns dict with train/valid/oos sample lists. Each sample:
        {race_id, hist:[L,H], ctx:[CTX_DIM], is_winner:0/1, finish:int}
    Leak-safe: a sample's history uses only the horse's races strictly earlier
    than the target race date (ties broken so same-day later races are excluded).
    """
    con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
    rows = con.execute(
        """
        SELECT e.horse_id, e.race_id, r.date, e.finish_position, e.finish_time,
               e.margin, e.agari_3f, e.passing, e.weight_carried, e.horse_weight,
               e.age, e.post_position, r.distance, r.surface, r.race_class, r.n_runners,
               e.odds_win
        FROM entries e JOIN races r ON r.race_id = e.race_id
        WHERE e.finish_position IS NOT NULL AND r.date IS NOT NULL
        ORDER BY e.horse_id, r.date, e.race_id
        """
    ).fetchall()
    con.close()

    # field size per race (count finishers)
    field: dict[str, int] = {}
    for row in rows:
        field[row[1]] = field.get(row[1], 0) + 1

    # group by horse, in chronological order (query already sorted)
    from collections import defaultdict
    by_horse: dict[str, list] = defaultdict(list)
    for row in rows:
        by_horse[row[0]].append(row)

    def token(row, prev_date) -> list[float]:
        (_hid, rid, date, fin, ftime, margin, agari, passing, wcar, hw, _age,
         _pp, dist, surf, rclass, _nr, _ow) = row
        fs = field.get(rid, 1)
        days = np.nan
        if prev_date is not None:
            days = (np.datetime64(date) - np.datetime64(prev_date)).astype("timedelta64[D]").astype(float)
        num = [
            (fin / fs) if fs else np.nan,
            float(fs),
            float(agari) if agari is not None else np.nan,
            _margin_num(margin),
            _passing_first(passing),
            float(wcar) if wcar is not None else np.nan,
            float(hw) if hw is not None else np.nan,
            float(dist) if dist is not None else np.nan,
            float(_CLASS_RANK.get(rclass, 0)),
            days,
            1.0 if fin == 1 else 0.0,
        ]
        return num + _surf_onehot(surf)

    # precompute token per (horse, index) and the date
    train, valid, oos = [], [], []
    for _hid, hist_rows in by_horse.items():
        toks: list[list[float]] = []
        dates: list[str] = []
        prev_date = None
        for row in hist_rows:
            toks.append(token(row, prev_date))
            dates.append(row[2])
            prev_date = row[2]
        # emit a sample for each race that has >=1 prior race
        for i, row in enumerate(hist_rows):
            if i == 0:
                continue  # no history -> skip (cannot encode a sequence)
            date = row[2]
            hist = toks[max(0, i - MAX_HIST):i]  # strictly past (indices < i)
            (_h, rid, _d, fin, _ft, _m, _a, _ps, wcar, _hw, age, pp,
             dist, surf, rclass, _nr, ow) = row
            fs = field.get(rid, 1)
            days_last = (np.datetime64(date) - np.datetime64(dates[i - 1])).astype("timedelta64[D]").astype(float)
            ctx = [
                float(dist) if dist is not None else np.nan,
                float(_CLASS_RANK.get(rclass, 0)),
                float(fs),
                float(age) if age is not None else np.nan,
                float(wcar) if wcar is not None else np.nan,
                float(days_last),
                float(i),  # career_len (# prior races)
                float(pp) if pp is not None else np.nan,
            ] + _surf_onehot(surf)
            if _WITH_ODDS:
                o = float(ow) if ow is not None and ow > 0 else np.nan
                ctx += [np.log(o) if o == o else np.nan, (1.0 / o) if o == o else np.nan]
            sample = {
                "race_id": rid, "hist": np.array(hist, dtype=np.float32),
                "ctx": np.array(ctx, dtype=np.float32),
                "is_winner": 1.0 if fin == 1 else 0.0, "finish": int(fin),
            }
            if date < TRAIN_END:
                train.append(sample)
            elif date < VALID_END:
                valid.append(sample)
            elif OOS_START <= date <= OOS_END:
                oos.append(sample)
    if limit_train_races is not None:
        keep = set(list({s["race_id"] for s in train})[:limit_train_races])
        train = [s for s in train if s["race_id"] in keep]
    return {"train": train, "valid": valid, "oos": oos}


def fit_normalizer(samples) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean/std for hist tokens and ctx, from TRAIN only. NaN-robust."""
    hs = np.concatenate([s["hist"] for s in samples], axis=0)  # [sumL, H]
    cs = np.stack([s["ctx"] for s in samples], axis=0)         # [N, CTX]
    h_mean = np.nanmean(hs, axis=0)
    h_std = np.nanstd(hs, axis=0) + 1e-6
    c_mean = np.nanmean(cs, axis=0)
    c_std = np.nanstd(cs, axis=0) + 1e-6
    return h_mean, h_std, c_mean, c_std


def group_by_race(samples) -> list[list[dict]]:
    from collections import defaultdict
    g: dict[str, list] = defaultdict(list)
    for s in samples:
        g[s["race_id"]].append(s)
    # only races where the winner is present as a sample (needed for the loss)
    return [v for v in g.values() if any(x["is_winner"] == 1.0 for x in v) and len(v) >= 2]


class SeqRanker(nn.Module):
    def __init__(self, h_dim: int, ctx_dim: int, hidden: int = 64, gru: int = 64):
        super().__init__()
        self.gru = nn.GRU(h_dim, gru, batch_first=True)
        self.ctx = nn.Sequential(nn.Linear(ctx_dim, hidden), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(gru + hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, hist, lengths, ctx):
        # hist [M, L, H], lengths [M], ctx [M, ctx_dim]
        packed = nn.utils.rnn.pack_padded_sequence(
            hist, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hn = self.gru(packed)        # hn [1, M, gru]
        h = hn.squeeze(0)               # [M, gru]
        c = self.ctx(ctx)               # [M, hidden]
        return self.head(torch.cat([h, c], dim=1)).squeeze(1)  # [M]


def _normalize(samples, h_mean, h_std, c_mean, c_std):
    for s in samples:
        s["hist"] = np.nan_to_num((s["hist"] - h_mean) / h_std, nan=0.0).astype(np.float32)
        s["ctx"] = np.nan_to_num((s["ctx"] - c_mean) / c_std, nan=0.0).astype(np.float32)


def _collate(races, device):
    """races: list[list[sample]] -> flat padded tensors + race spans + winner idx."""
    horses = [h for race in races for h in race]
    M = len(horses)
    hist = torch.zeros(M, MAX_HIST, H)
    lengths = torch.ones(M, dtype=torch.long)
    ctx = torch.zeros(M, CTX_DIM)
    for j, hsmp in enumerate(horses):
        L = hsmp["hist"].shape[0]
        hist[j, :L] = torch.from_numpy(hsmp["hist"])
        lengths[j] = max(1, L)
        ctx[j] = torch.from_numpy(hsmp["ctx"])
    spans, winners, off = [], [], 0
    for race in races:
        n = len(race)
        spans.append((off, off + n))
        wi = next(k for k, x in enumerate(race) if x["is_winner"] == 1.0)
        winners.append(off + wi)
        off += n
    return (hist.to(device), lengths.to(device), ctx.to(device),
            spans, torch.tensor(winners, device=device))


def _loss(scores, spans, winners):
    ls = []
    for (a, b), w in zip(spans, winners, strict=True):
        ls.append(torch.log_softmax(scores[a:b], dim=0)[w - a])
    return -torch.stack(ls).mean()


@torch.no_grad()
def _evaluate(model, races, device, batch=256):
    model.eval()
    ndcgs, hits = [], []
    for i in range(0, len(races), batch):
        chunk = races[i:i + batch]
        hist, lengths, ctx, spans, _w = _collate(chunk, device)
        scores = model(hist, lengths, ctx).cpu().numpy()
        for race, (a, b) in zip(chunk, spans, strict=True):
            true_rel = np.array([[x["is_winner"] for x in race]])
            pred = scores[a:b].reshape(1, -1)
            ndcgs.append(ndcg_score(true_rel, pred, k=1))
            hits.append(1.0 if int(np.argmax(pred)) == next(
                k for k, x in enumerate(race) if x["is_winner"] == 1.0) else 0.0)
    return float(np.mean(ndcgs)), float(np.mean(hits)), len(races)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-races", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit-train-races", type=int, default=None)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    t0 = time.time()
    data = load_history_and_samples(str(db_path()), args.limit_train_races)
    print(f"samples: train={len(data['train'])} valid={len(data['valid'])} oos={len(data['oos'])} "
          f"(load {time.time()-t0:.0f}s)")
    norm = fit_normalizer(data["train"])
    for split in ("train", "valid", "oos"):
        _normalize(data[split], *norm)
    train_r = group_by_race(data["train"])
    valid_r = group_by_race(data["valid"])
    oos_r = group_by_race(data["oos"])
    print(f"races: train={len(train_r)} valid={len(valid_r)} oos={len(oos_r)} "
          f"| H={H} CTX={CTX_DIM} device={device}")

    model = SeqRanker(H, CTX_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(0)
    best_valid, best_oos = -1.0, None
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(train_r))
        tot = 0.0
        for i in range(0, len(order), args.batch_races):
            idx = order[i:i + args.batch_races]
            hist, lengths, ctx, spans, winners = _collate([train_r[k] for k in idx], device)
            opt.zero_grad()
            loss = _loss(model(hist, lengths, ctx), spans, winners)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        v_ndcg, v_hit, _ = _evaluate(model, valid_r, device)
        marker = ""
        if v_ndcg > best_valid:
            best_valid = v_ndcg
            best_oos = _evaluate(model, oos_r, device)
            marker = " *"
        print(f"epoch {ep+1:2d} | train_loss {tot/len(train_r):.4f} | "
              f"valid ndcg1 {v_ndcg:.4f} hit {v_hit:.4f}{marker}")

    o_ndcg, o_hit, o_n = best_oos
    print("\n=== OOS (2025-10..2026-04, best-valid checkpoint) ===")
    print(f"  seq model: ndcg1={o_ndcg:.4f}  top1_hit={o_hit:.4f}  races={o_n}")
    print("  anchor    : no-odds GBDT test_ndcg1≈0.511 | with-odds 0.558 | favorite 0.583")


if __name__ == "__main__":
    main()
