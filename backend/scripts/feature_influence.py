"""SHAP-based feature influence analysis to guide training improvements.

Gain importance (feature_analysis.py) is biased toward high-cardinality features.
This uses LightGBM's exact TreeSHAP (Booster.predict(pred_contrib=True), no extra
deps) on held-out rows to measure, per feature:

  - mean|SHAP|  : true average influence on the model's output score (unbiased).
  - direction   : sign of corr(feature_value, SHAP) — does higher value push the
                  score up or down? (e.g. lower odds_win SHOULD push score up.)
  - monotonicity: |corr(value, SHAP)| — ~1 = the model uses the feature
                  monotonically; low = wiggly/non-monotonic usage = overfit-prone
                  and a candidate for a LightGBM monotone constraint.

Output guides concrete training changes: prune SHAP-dead features, add monotone
constraints to clearly-monotonic ones (better OOS generalisation), and spot
non-monotonic-but-important features that may be fitting noise.

Read-only; uses cached frame + existing booster(s). OOS window 2024-05..2026-04.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np

from core.paths import db_path
from db.session import make_engine, session_scope
from features.builder import CATEGORICAL_FEATURES, build_training_frame

OOS_START, OOS_END = "2024-05-01", "2026-04-30"
MODELS = {
    "with-odds (46f, 20260603-215349)":
        "/mnt/c/Users/PC_User/private_production/keiba-ai/data/models/20260603-215349/model.txt",
    "no-odds (20260601-170042)":
        "/mnt/c/Users/PC_User/private_production/keiba-ai/data/models/20260601-170042/model.txt",
}
SAMPLE = 25_000


def analyse(name: str, model_file: str, frame) -> None:
    b = lgb.Booster(model_file=model_file)
    feats = b.feature_name()
    X = frame[feats].copy()
    for c in CATEGORICAL_FEATURES:
        if c in X.columns:
            X[c] = X[c].astype("category")
    if len(X) > SAMPLE:
        X = X.sample(SAMPLE, random_state=0)

    contrib = b.predict(X, pred_contrib=True)  # [n, F+1] (last col = base)
    shap = contrib[:, :-1]
    mean_abs = np.abs(shap).mean(axis=0)
    tot = mean_abs.sum() or 1.0
    gains = b.feature_importance(importance_type="gain")
    gtot = gains.sum() or 1.0

    rows = []
    for i, f in enumerate(feats):
        sh = shap[:, i]
        if f in CATEGORICAL_FEATURES:
            direction, mono = "cat", float("nan")
        else:
            xv = X[f].to_numpy(dtype=np.float64)
            ok = np.isfinite(xv) & np.isfinite(sh)
            if ok.sum() > 100 and np.std(xv[ok]) > 1e-9 and np.std(sh[ok]) > 1e-9:
                r = float(np.corrcoef(xv[ok], sh[ok])[0, 1])
            else:
                r = float("nan")
            direction = "↑+" if r > 0 else ("↓−" if r < 0 else "·")
            mono = abs(r)
        rows.append((f, 100 * mean_abs[i] / tot, 100 * gains[i] / gtot, direction, mono))

    rows.sort(key=lambda x: -x[1])
    print(f"\n===== {name}  (n={len(X)}) =====")
    print(f"{'feature':<32}{'SHAP%':>7}{'gain%':>7}{'dir':>5}{'mono|r|':>8}")
    for f, shp, gn, d, mono in rows:
        flag = ""
        if shp < 0.3:
            flag = " ←SHAP-dead"
        elif not np.isnan(mono) and mono < 0.2 and shp > 2.0:
            flag = " ←非単調&重要(過学習/制約候補)"
        elif not np.isnan(mono) and mono >= 0.6 and shp > 1.0:
            flag = " ←単調(制約OK)"
        print(f"{f:<32}{shp:>7.2f}{gn:>7.2f}{d:>5}{mono:>8.2f}{flag}")

    dead = [f for f, shp, *_ in rows if shp < 0.3]
    print(f"  SHAP-dead ({len(dead)}): {', '.join(dead) if dead else 'none'}")


def main() -> None:
    engine = make_engine(db_path())
    with session_scope(engine) as s:
        frame = build_training_frame(s, train_start=OOS_START, train_end=OOS_END)
    print(f"OOS frame rows={len(frame)} ({OOS_START}..{OOS_END})")
    for name, mf in MODELS.items():
        analyse(name, mf, frame)


if __name__ == "__main__":
    main()
