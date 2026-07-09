"""Model-side A/B harness — feature / preprocessing / loss knobs vs baseline.

2026-06〜07 に「事前データ処理・特徴量・損失関数で本番 ROI を改善できるか」を
検証した一連の A/B を 1 本に集約したもの。各ノブは env-gated (default-off) で
実装済み ([[builder.py]] / [[preprocess.py]] / [[speed_figure.py]] /
[[relative_features.py]] / [[loss.py]])、本スクリプトは baseline↔treatment を
**同一 seed で paired 比較**し、with-odds (本番) / no-odds (ability) の両 regime で
ROI・的中率・ndcg3 の差分を出す。`--no-persist` 相当 (persist=False) なので
models/ と keiba.db は一切触らない。

ノブ (`--knob`):
  missing_log : A1+A2  KEIBA_MISSING_INDICATORS + KEIBA_LOG_FEATURES (欠損flag+log)
  speed       : B1     KEIBA_SPEED_FIGURE (par-time/track-variant タイム指数, 履歴17次元)
  pace        : B2     KEIBA_PACE_FEATURES (projected_pace + 脚質×ペース交互作用)
  loss_kelly  : L1     --loss kelly_deploy (デプロイ整合 Kelly) vs log_growth

結論 (全ノブ・multi-seed): **本番 (with-odds) の tansho ROI はどれも改善せず** —
missing_log/pace は再表現で悪化、speed(新情報)は single-seed の見かけ改善が
multi-seed で霧散 (-0.03)、loss_kelly は ROI -0.06 だが ndcg3 +0.085 (精度↑=本命
追従=ROI↓)。odds を入力する本番モデルは市場効率の壁を越えられない、が実証された。
詳細は docs/ai-model.md の「実験ノブと A/B 知見」。

Usage:
    PYTHONPATH=src uv run python -m scripts.model_side_ab --knob loss_kelly \
        --seeds 42,1,7 --train-end 2025-06-30 --valid-months 6 --test-months 6 \
        --max-epochs 60 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from core.paths import db_path
from db.session import make_engine, session_scope
from features.builder import build_training_frame

# knob -> treatment env flags / loss overrides / build strategy
_TREATMENT_FLAGS = {
    "missing_log": {"KEIBA_MISSING_INDICATORS": "1", "KEIBA_LOG_FEATURES": "1"},
    "speed": {"KEIBA_SPEED_FIGURE": "1"},
    "pace": {"KEIBA_PACE_FEATURES": "1"},
    "loss_kelly": {},  # differs by --loss, not env
}
_KNOB_ALL_FLAGS = [
    "KEIBA_MISSING_INDICATORS", "KEIBA_LOG_FEATURES", "KEIBA_LOG_FEATURE_COLS",
    "KEIBA_SPEED_FIGURE", "KEIBA_PACE_FEATURES",
]
# frame changes with the flag (need a separate treatment frame build)
_KNOB_CHANGES_FRAME = {"missing_log", "pace"}
# history token dim changes with the flag (can't share a prebuilt history)
_KNOB_CHANGES_HISTORY = {"speed"}
_METRICS = [
    "test_tansho_roi", "test_fukusho_roi", "test_tansho_hit",
    "test_fukusho_hit", "test_ndcg3",
]


def _clear_flags() -> None:
    for k in _KNOB_ALL_FLAGS:
        os.environ.pop(k, None)


def _apply(flags: dict[str, str]) -> None:
    _clear_flags()
    for k, v in flags.items():
        os.environ[k] = v


def _seed(seed: int) -> None:
    import lightning as L  # noqa: N812
    L.seed_everything(seed, workers=True)


def _build_frame(flags: dict[str, str]) -> pd.DataFrame:
    _apply(flags)
    eng = make_engine(db_path())
    with session_scope(eng) as s:
        df = build_training_frame(s)
    eng.dispose()
    print(f"[frame] flags={flags or 'base'} rows={len(df)} cols={len(df.columns)}", flush=True)
    return df


def _build_history():
    _clear_flags()
    from features.history_sequence import build_history_sequences
    eng = make_engine(db_path())
    with session_scope(eng) as s:
        h = build_history_sequences(s, max_len=15)
    eng.dispose()
    print(f"[history] seqs={len(h.seqs)} feat_dim={h.n_features}", flush=True)
    return h


def _run(args, label, frame, history, *, seed, flags, loss, no_odds) -> dict:
    _apply(flags)
    os.environ["KEIBA_EXCLUDE_ODDS_FEATURES"] = "1" if no_odds else "0"
    _seed(seed)
    from ai.training.train_nn import train_nn

    m = train_nn(
        train_end=args.train_end, valid_months=args.valid_months,
        test_months=args.test_months, loss=loss, monitor="valid_tansho_roi",
        device=args.device, max_epochs=args.max_epochs,
        prebuilt_frame=frame, prebuilt_history=history, persist=False,
    )
    keep = {k: m.get(k) for k in _METRICS}
    print(f"[result] s{seed} {label}: {json.dumps(keep)}", flush=True)
    return {"label": label, "seed": seed, **keep}


def _print_deltas(rows, seeds, base_lbl, treat_lbl, title) -> None:
    by = {(r["seed"], r["label"]): r for r in rows}
    print(f"\n===== paired delta (treatment - base), {title} =====", flush=True)
    print(f"{'seed':>6} {'d_tan_roi':>10} {'d_fuk_roi':>10} {'d_t_hit':>9} {'d_f_hit':>9} {'d_ndcg3':>9}")
    acc = {k: [] for k in _METRICS}
    for seed in seeds:
        b, p = by[(seed, base_lbl)], by[(seed, treat_lbl)]
        for k in _METRICS:
            acc[k].append(p[k] - b[k])
        print(f"{seed:>6} {p[_METRICS[0]]-b[_METRICS[0]]:>10.4f} {p[_METRICS[1]]-b[_METRICS[1]]:>10.4f} "
              f"{p[_METRICS[2]]-b[_METRICS[2]]:>9.4f} {p[_METRICS[3]]-b[_METRICS[3]]:>9.4f} "
              f"{p[_METRICS[4]]-b[_METRICS[4]]:>9.4f}")
    n = len(seeds)
    print(f"{'mean':>6} " + " ".join(
        f"{sum(acc[k])/n:>{w}.4f}" for k, w in zip(_METRICS, (10, 10, 9, 9, 9), strict=True)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--knob", required=True, choices=list(_TREATMENT_FLAGS))
    ap.add_argument("--seeds", default="42,1,7")
    ap.add_argument("--train-end", default="2025-06-30")
    ap.add_argument("--valid-months", type=int, default=6)
    ap.add_argument("--test-months", type=int, default=6)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    knob = args.knob
    treat_flags = _TREATMENT_FLAGS[knob]
    base_loss = "log_growth"
    treat_loss = "kelly_deploy" if knob == "loss_kelly" else "log_growth"

    print(f"knob={knob} seeds={seeds} train_end={args.train_end} "
          f"valid/test {args.valid_months}/{args.test_months}m epochs={args.max_epochs}", flush=True)

    frame_base = _build_frame({})
    frame_treat = _build_frame(treat_flags) if knob in _KNOB_CHANGES_FRAME else frame_base
    # speed changes history dim → rebuild per run (prebuilt_history=None); else share.
    history = None if knob in _KNOB_CHANGES_HISTORY else _build_history()

    rows = []
    for seed in seeds:
        for no_odds in (False, True):
            tag = "no-odds" if no_odds else "odds"
            rows.append(_run(args, f"base/{tag}", frame_base, history,
                             seed=seed, flags={}, loss=base_loss, no_odds=no_odds))
            rows.append(_run(args, f"treat/{tag}", frame_treat, history,
                             seed=seed, flags=treat_flags, loss=treat_loss, no_odds=no_odds))

    _print_deltas(rows, seeds, "base/odds", "treat/odds", "WITH-ODDS (production)")
    _print_deltas(rows, seeds, "base/no-odds", "treat/no-odds", "NO-ODDS (ability)")

    out = args.out or f"/tmp/model_side_ab_{knob}.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
