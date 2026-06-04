"""Controlled A/B: do the 8 richer history features help, with vs without odds?

Trains lambdarank GBDT on the SAME cached frame / 2-year split with the new 8
features included vs excluded, for both the with-odds and no-odds feature sets,
and reports OOS ndcg@1. Isolates the effect of the new aggregations alone.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
from sklearn.metrics import ndcg_score

from ai.labels import assign_relevance
from core.paths import db_path
from db.session import make_engine, session_scope
from features.builder import CATEGORICAL_FEATURES, build_training_frame, get_active_features

NEW8 = ["recent_wavg_finish", "recent_finish_std", "recent_finish_trend",
        "recent_best_finish", "recent_wavg_agari_3f", "recent_agari_std",
        "recent_agari_trend", "recent_wavg_margin"]
TRAIN_END, OOS_START, OOS_END = "2023-11-01", "2024-05-01", "2026-04-30"
PARAMS = {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [1],
          "num_leaves": 127, "learning_rate": 0.03, "min_data_in_leaf": 50,
          "feature_fraction": 0.85, "bagging_fraction": 0.85, "bagging_freq": 5,
          "lambdarank_truncation_level": 5, "verbose": -1}


def _ds(df, feats):
    cats = [c for c in CATEGORICAL_FEATURES if c in feats]
    X = df[feats].copy()
    for c in cats:
        X[c] = X[c].astype("category")
    y = df["finish_position"].map(assign_relevance).values
    grp = df.groupby("race_id", sort=False).size().values
    return lgb.Dataset(X, label=y, group=grp, categorical_feature=cats), X


def ndcg1(model, df, feats):
    Xc = df[feats].copy()
    for c in [c for c in CATEGORICAL_FEATURES if c in feats]:
        Xc[c] = Xc[c].astype("category")
    pred = model.predict(Xc)
    df = df.assign(_p=pred)
    vals = []
    for _rid, g in df.groupby("race_id", sort=False):
        rel = g["finish_position"].map(assign_relevance).to_numpy().reshape(1, -1)
        if rel.shape[1] >= 2:
            vals.append(ndcg_score(rel, g["_p"].to_numpy().reshape(1, -1), k=1))
    return float(np.mean(vals))


def run(df_tr, df_oos, feats, label):
    dtr, _ = _ds(df_tr, feats)
    model = lgb.train(PARAMS, dtr, num_boost_round=800)
    print(f"  {label:<28} feats={len(feats):>3}  OOS ndcg1={ndcg1(model, df_oos, feats):.4f}")


def main() -> None:
    engine = make_engine(db_path())
    with session_scope(engine) as s:
        frame = build_training_frame(s, train_start="2015-01-01", train_end=OOS_END)
    tr = frame[frame["date"] < TRAIN_END]
    oos = frame[(frame["date"] >= OOS_START) & (frame["date"] <= OOS_END)]
    print(f"train rows={len(tr)} oos rows={len(oos)}")

    import os
    os.environ["KEIBA_EXCLUDE_ODDS_FEATURES"] = "0"
    wo = get_active_features()
    os.environ["KEIBA_EXCLUDE_ODDS_FEATURES"] = "1"
    no = get_active_features()

    print("\n=== with-odds ===")
    run(tr, oos, [f for f in wo if f not in NEW8], "without new8")
    run(tr, oos, wo, "with new8")
    print("\n=== no-odds ===")
    run(tr, oos, [f for f in no if f not in NEW8], "without new8")
    run(tr, oos, no, "with new8")


if __name__ == "__main__":
    main()
