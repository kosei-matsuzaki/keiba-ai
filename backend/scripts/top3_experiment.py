"""Does training on the *right target* (top-3 finish) raise 複勝 ROI?

The production model only ever optimises a win/ranking objective; place & all
combo probabilities are DERIVED from those win-centric scores via Plackett-Luce.
Nothing is trained directly on "finishes in the money (top-3)" — which is the
quantity 複勝 / ワイド / 三連系 actually pay on. A horse that reliably places but
rarely wins is structurally under-valued by a win objective.

This experiment trains a **no-odds GBDT directly on is_top3 = (finish<=3)**,
isotonic-calibrated, and value-bets 複勝 against the REAL place odds (odds.db,
all horses) on the held-out OOS window, settling with the REAL place payouts.
ROI is reported at several EV thresholds with race-level bootstrap 95% CI, and
compared against the favorite-place baseline.

no-odds (market-independent ability) is the point: edge can only come from the
model DISAGREEING with the market where it is right. Controlled split identical
to the other experiments; OOS held out from train+valid+threshold choice.

Run with KEIBA_EXCLUDE_ODDS_FEATURES=1.
"""

from __future__ import annotations

import contextlib
import gzip
import json
import sqlite3

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression

from core.paths import db_path, odds_db_path
from db.session import make_engine, session_scope
from features.builder import CATEGORICAL_FEATURES, build_training_frame, get_active_features

TRAIN_END, VALID_END = "2025-01-05", "2025-07-05"
OOS_START, OOS_END = "2025-10-01", "2026-04-30"


def _prep(df, feats):
    X = df[feats].copy()
    for c in CATEGORICAL_FEATURES:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


def train_top3(frame, feats):
    tr = frame[frame["date"] < TRAIN_END]
    va = frame[(frame["date"] >= TRAIN_END) & (frame["date"] < VALID_END)]
    cats = [c for c in CATEGORICAL_FEATURES if c in feats]
    y_tr = (tr["finish_position"] <= 3).astype(int).values
    y_va = (va["finish_position"] <= 3).astype(int).values
    dtr = lgb.Dataset(_prep(tr, feats), label=y_tr, categorical_feature=cats)
    dva = lgb.Dataset(_prep(va, feats), label=y_va, reference=dtr, categorical_feature=cats)
    params = {
        "objective": "binary", "metric": "binary_logloss", "num_leaves": 127,
        "learning_rate": 0.03, "min_data_in_leaf": 50, "feature_fraction": 0.85,
        "bagging_fraction": 0.85, "bagging_freq": 5, "verbose": -1,
    }
    model = lgb.train(
        params, dtr, num_boost_round=2000, valid_sets=[dva],
        callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)],
    )
    # isotonic calibration on valid (P(top3) -> empirical top3 rate)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(model.predict(_prep(va, feats)), y_va)
    return model, iso, cats


def load_place_odds() -> dict[str, dict[str, float]]:
    """odds.db 複勝: {race_id: {pp_str: min_odds}} (lower bound = conservative)."""
    # NOTE: plain mode=ro (NOT immutable) — the odds backfill may still be
    # writing odds.db; immutable=1 bypasses locking and yields torn reads.
    con = sqlite3.connect(f"file:{odds_db_path()}?mode=ro", uri=True, timeout=60)
    out: dict[str, dict[str, float]] = {}
    for rid, blob in con.execute("SELECT race_id, data FROM race_odds WHERE bet_type='複勝'"):
        d = json.loads(gzip.decompress(blob))
        out[rid] = {k: float(v[0]) for k, v in d.items()}
    con.close()
    return out


def load_place_payouts() -> dict[str, dict[int, float]]:
    """races.payout_place: {race_id: {finish_pos:int -> odds(=amount/100)}}."""
    con = sqlite3.connect(f"file:{db_path()}?mode=ro&immutable=1", uri=True)
    out: dict[str, dict[int, float]] = {}
    rows = con.execute(
        "SELECT race_id, payout_place FROM races "
        "WHERE date >= ? AND date <= ? AND payout_place IS NOT NULL", (OOS_START, OOS_END),
    ).fetchall()
    con.close()
    for rid, pj in rows:
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            out[rid] = {int(k): v / 100.0 for k, v in json.loads(pj).items() if k.isdigit()}
    return out


def _ci(per_race: dict, n_boot=2000, seed=0):
    arr = np.array(list(per_race.values()), dtype=np.float64)  # [R,2] stake,payout
    if len(arr) == 0:
        return 0.0, 0.0, 0.0, 0
    tot = arr[:, 0].sum()
    point = arr[:, 1].sum() / tot if tot > 0 else 0.0
    rng = np.random.default_rng(seed)
    b = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(arr), len(arr))
        s = arr[idx, 0].sum()
        b[i] = arr[idx, 1].sum() / s if s > 0 else 0.0
    lo, hi = np.percentile(b, [2.5, 97.5])
    return point, float(lo), float(hi), len(arr)


def main() -> None:
    feats = get_active_features()
    print(f"features: {len(feats)} (no-odds={'odds_win' not in feats})")
    engine = make_engine(db_path())
    with session_scope(engine) as s:
        frame = build_training_frame(s, train_start="2015-01-01", train_end=OOS_END)
    model, iso, _cats = train_top3(frame, feats)

    oos = frame[(frame["date"] >= OOS_START) & (frame["date"] <= OOS_END)].copy()
    oos["p_top3"] = iso.predict(model.predict(_prep(oos, feats)))

    place_odds = load_place_odds()
    place_pay = load_place_payouts()

    # value-bet 複勝 at several EV thresholds; settle with real payouts.
    from collections import defaultdict
    print(f"\n=== 複勝 value-bet (no-odds direct P(top3) vs real place odds), OOS {OOS_START}..{OOS_END} ===")
    print(f"{'EV>=':>5} {'n_bets':>7} {'n_race':>6} {'payback':>8} {'95% CI':>16} {'hit%':>6} {'avgOdds':>8}")
    for thr in (1.0, 1.1, 1.2, 1.5):
        per_race: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        nbets = 0
        hits = 0
        odds_sum = 0.0
        for rid, grp in oos.groupby("race_id"):
            po = place_odds.get(rid)
            pay = place_pay.get(rid)
            if not po or not pay:
                continue
            for _, row in grp.iterrows():
                pp = row["post_position"]
                if pp is None or np.isnan(pp):
                    continue
                o = po.get(str(int(pp)))
                if o is None:
                    continue
                ev = float(row["p_top3"]) * o
                if ev < thr:
                    continue
                nbets += 1
                odds_sum += o
                fin = row["finish_position"]
                payout = pay.get(int(fin), 0.0) if fin is not None and not np.isnan(fin) else 0.0
                if payout > 0:
                    hits += 1
                per_race[rid][0] += 1.0
                per_race[rid][1] += payout
        pb, lo, hi, nr = _ci(per_race)
        hitp = 100.0 * hits / nbets if nbets else 0.0
        avgo = odds_sum / nbets if nbets else 0.0
        print(f"{thr:>5.1f} {nbets:>7} {nr:>6} {pb:>8.3f} [{lo:>5.2f},{hi:>5.2f}] {hitp:>6.1f} {avgo:>8.1f}")

    # reference: favorite-place (lowest place odds) blind bet ROI on same races
    per_race = defaultdict(lambda: [0.0, 0.0])
    for rid, grp in oos.groupby("race_id"):
        po = place_odds.get(rid)
        pay = place_pay.get(rid)
        if not po or not pay:
            continue
        fav_pp = min(po, key=po.get)
        per_race[rid][0] += 1.0
        # find finisher with that post_position
        fr = grp[grp["post_position"] == int(fav_pp)]
        if not fr.empty:
            fin = fr.iloc[0]["finish_position"]
            per_race[rid][1] += pay.get(int(fin), 0.0) if fin is not None and not np.isnan(fin) else 0.0
    pb, lo, hi, nr = _ci(per_race)
    print(f"\n  reference favorite-place (blind): payback={pb:.3f} CI[{lo:.2f},{hi:.2f}] races={nr}")


if __name__ == "__main__":
    main()
