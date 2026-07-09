"""A/B: does batch size (with sqrt-scaled lr) change OOS ROI / hit-rate?

本番構成 (multi loss, 履歴GRU, odds-head, max_epochs 50, monitor=valid_tansho_roi) を
固定し、batch_size のみ振る統制比較。フレーム/履歴は 1 回だけ構築し全 config で再利用。
評価は train_nn の held-out test 窓 (train_end 後 valid→test) の実オッズ top-1 単勝 bets
({odds, won, tansho_ret}) から ROI / 的中率 / 購入オッズ分布 / race-level bootstrap CI。

Usage:
    PYTHONPATH=src uv run python -m scripts.batch_size_experiment \
        --batches 32,256 --seeds 2 --train-end 2024-04-28 --test-months 14 --max-epochs 50
"""
from __future__ import annotations

import argparse
import gc
import json
import math

import numpy as np
import lightning.pytorch as pl

from core.paths import db_path
from db.session import make_engine, session_scope
from features.builder import build_training_frame
from features.history_sequence import build_history_sequences
from ai.training.train_nn import train_nn


def _summ(bets, n_boot=2000, seed=0):
    """test_bets -> ROI / hit / pick-odds 分布 + race-level bootstrap CI。"""
    if not bets:
        return None
    rets = np.array([b["tansho_ret"] for b in bets], dtype=float)  # =odds if won else 0
    odds = np.array([b["odds"] for b in bets], dtype=float)
    won = np.array([1.0 if b["won"] else 0.0 for b in bets])
    # オッズ欠損 (NaN) のベットは価格付け不能 → ROI 計算から除外 (nan-safe)。
    valid = ~np.isnan(rets)
    rets = rets[valid]
    roi = float(rets.mean()) if len(rets) else float("nan")
    rng = np.random.default_rng(seed)
    n = len(rets)
    boot = np.array([rets[rng.integers(0, n, n)].mean() for _ in range(n_boot)]) if n else np.array([np.nan])
    return {
        "n_bets": n,
        "roi": roi,
        "roi_ci": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "hit": float(won.mean()),
        "pick_odds_mean": float(np.nanmean(odds)),
        "pick_odds_gt20_pct": float(100.0 * np.mean(odds > 20)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="32,256")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--train-end", default="2024-04-28")
    ap.add_argument("--valid-months", type=int, default=6)
    ap.add_argument("--test-months", type=int, default=14)
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--base-lr", type=float, default=1e-3, help="lr at batch=32; sqrt-scaled")
    args = ap.parse_args()

    batches = [int(b) for b in args.batches.split(",")]

    # ---- build frame + history ONCE (reused across all configs) ----
    engine = make_engine(db_path())
    print("# building full training frame (cached)...", flush=True)
    with session_scope(engine) as session:
        frame = build_training_frame(session)
        print(f"# frame: {len(frame)} rows / {frame['race_id'].nunique()} races", flush=True)
        print("# building history sequences...", flush=True)
        history = build_history_sequences(session, max_len=15)
    print("# prebuilt ready.\n", flush=True)

    results = []
    for bs in batches:
        lr = args.base_lr * math.sqrt(bs / 32.0)
        for seed in range(args.seeds):
            pl.seed_everything(seed, workers=True)
            tag = f"bs{bs}_lr{lr:.2e}_s{seed}"
            print(f"=== training {tag} ===", flush=True)
            out = train_nn(
                train_end=args.train_end,
                valid_months=args.valid_months,
                test_months=args.test_months,
                loss="multi",
                batch_size=bs,
                learning_rate=lr,
                max_epochs=args.max_epochs,
                device="cuda",
                monitor="valid_tansho_roi",
                combo_bet_type="馬連",
                combo_weight=0.01,
                history_seq_len=15,
                prebuilt_frame=frame,
                prebuilt_history=history,
                persist=False,
                return_test_bets=True,
            )
            s = _summ(out.get("test_bets", []), seed=seed)
            m = out  # metrics dict also at top level
            rec = {
                "batch_size": bs, "lr": lr, "seed": seed,
                "test_tansho_roi": m.get("test_tansho_roi"),
                "test_fukusho_roi": m.get("test_fukusho_roi"),
                "test_tansho_hit": m.get("test_tansho_hit"),
                "summary": s,
            }
            results.append(rec)
            if s:
                print(f"  -> ROI={s['roi']:.4f} CI{ [round(x,3) for x in s['roi_ci']] } "
                      f"hit={s['hit']:.4f} pick_odds={s['pick_odds_mean']:.1f} "
                      f">20={s['pick_odds_gt20_pct']:.1f}% n={s['n_bets']}", flush=True)
            del out
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    # ---- aggregate by batch size ----
    print("\n# === SUMMARY (mean over seeds) ===", flush=True)
    print(f"{'batch':>6} {'lr':>9} {'ROI':>8} {'hit':>7} {'pick_odds':>10} {'>20%':>6} {'n':>6}")
    agg = {}
    for bs in batches:
        rs = [r for r in results if r["batch_size"] == bs and r["summary"]]
        if not rs:
            continue
        roi = np.mean([r["summary"]["roi"] for r in rs])
        hit = np.mean([r["summary"]["hit"] for r in rs])
        po = np.mean([r["summary"]["pick_odds_mean"] for r in rs])
        gt = np.mean([r["summary"]["pick_odds_gt20_pct"] for r in rs])
        n = int(np.mean([r["summary"]["n_bets"] for r in rs]))
        lr = rs[0]["lr"]
        agg[bs] = {"roi": roi, "hit": hit, "pick_odds": po, "gt20": gt, "n": n, "lr": lr}
        print(f"{bs:>6} {lr:>9.2e} {roi:>8.4f} {hit:>7.4f} {po:>10.1f} {gt:>6.1f} {n:>6}")

    with open("/tmp/batch_size_experiment.json", "w") as fh:
        json.dump({"results": results, "agg": agg}, fh, ensure_ascii=False, indent=2)
    print("\n# wrote /tmp/batch_size_experiment.json", flush=True)


if __name__ == "__main__":
    main()
