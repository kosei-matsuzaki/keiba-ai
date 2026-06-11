"""統制 A/B: 履歴系列エンコーダ (use_history) は本番 NN の OOS 精度/ROI を改善するか?

本番 RaceTransformerModel/train_nn に統合した履歴エンコーダを、同一フレーム・
同一 split・同一 seed で use_history ON vs OFF だけ変えて比較する。フレームと
履歴キャッシュは 1 度だけ構築して全アーム/seed で再利用 (prebuilt)。

C 実験 (scripts/seq_experiment.py) は「履歴系列 × ランキング損失 × ndcg」でしか
回しておらず +0.018 top1 だった。本実験は「履歴系列 × 本番 ROI 目的(log_growth)
× 本番 Set Transformer × ROI+ndcg 評価」という未検証の組み合わせを潰す。

まず no-odds (KEIBA_EXCLUDE_ODDS_FEATURES=1 = 市場非依存の実力モデル) で信号を見る。

使い方:
  UV_PROJECT_ENVIRONMENT=/tmp/keiba-linux-venv PYTHONPATH=src \
      KEIBA_EXCLUDE_ODDS_FEATURES=1 uv run python scripts/seq_integrated_experiment.py
"""

from __future__ import annotations

import json
import os
import statistics

from lightning.pytorch import seed_everything

from ai.training.train_nn import train_nn
from core.paths import data_dir, db_path
from db.session import make_engine, session_scope
from features.builder import build_training_frame
from features.history_sequence import build_history_sequences

TRAIN_END = "2025-04-30"   # train<2024-04-30 / test 2024-10..最新 (長期 OOS)
VALID_MONTHS = 6
TEST_MONTHS = 6
LOSS = "log_growth"        # 本番 ROI 目的 (単勝 fractional-Kelly)
MONITOR = "valid_tansho_roi"
DEVICE = "cuda"
MAX_EPOCHS = 50
PATIENCE = 8
HISTORY_SEQ_LEN = 15
SEEDS = [0, 1]
METRIC_KEYS = ["test_ndcg1", "test_ndcg3", "test_tansho_roi", "test_fukusho_roi"]

OUT = data_dir() / "cache" / "seq_integrated_results.json"


def run_one(frame, history_cache, use_history: bool, seed: int) -> dict:
    seed_everything(seed, workers=True)
    res = train_nn(
        prebuilt_frame=frame,
        prebuilt_history=history_cache if use_history else None,
        use_history=use_history,
        history_seq_len=HISTORY_SEQ_LEN,
        train_end=TRAIN_END,
        valid_months=VALID_MONTHS,
        test_months=TEST_MONTHS,
        loss=LOSS,
        monitor=MONITOR,
        device=DEVICE,
        max_epochs=MAX_EPOCHS,
        early_stopping_patience=PATIENCE,
        persist=False,          # keiba.db read-only, models/ を汚さない
        fit_temperature=False,  # metrics は温度前に計算済み → スキップで高速化
    )
    return {k: res.get(k) for k in METRIC_KEYS}


def main() -> None:
    no_odds = os.environ.get("KEIBA_EXCLUDE_ODDS_FEATURES", "").strip().lower() in {"1", "true", "yes"}
    eng = make_engine(db_path())
    with session_scope(eng) as session:
        frame = build_training_frame(session)
    print(f"frame built: {len(frame):,} rows, {frame['race_id'].nunique():,} races (no_odds={no_odds})", flush=True)
    with session_scope(eng) as session:
        history_cache = build_history_sequences(session, max_len=HISTORY_SEQ_LEN)
    print(f"history built: {len(history_cache.seqs):,} (race,horse) seqs, {history_cache.n_features} token feats", flush=True)

    results: dict[str, list[dict]] = {"base": [], "treat": []}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        for arm, use_hist in (("base", False), ("treat", True)):
            print(f"\n=== {arm} seed={seed} (use_history={use_hist}) ===", flush=True)
            r = run_one(frame, history_cache, use_hist, seed)
            r["seed"] = seed
            results[arm].append(r)
            OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
            print(f"  {arm} seed={seed}: " + ", ".join(
                f"{k}={r[k]:.4f}" if isinstance(r[k], (int, float)) else f"{k}={r[k]}"
                for k in METRIC_KEYS
            ), flush=True)

    print("\n" + "=" * 66)
    print(f"履歴系列 統制 A/B  (loss={LOSS}, monitor={MONITOR}, no_odds={no_odds}, seeds={SEEDS})")
    print("  base = 集約のみ (use_history=False) / treat = 集約 + 履歴系列エンコーダ")
    print("=" * 66)
    print(f"{'metric':18} {'base mean±std':>20} {'treat mean±std':>20} {'Δ(treat-base)':>14}")
    for k in METRIC_KEYS:
        b = [r[k] for r in results["base"] if isinstance(r[k], (int, float))]
        t = [r[k] for r in results["treat"] if isinstance(r[k], (int, float))]
        if not b or not t:
            continue
        bm, bs = statistics.mean(b), (statistics.stdev(b) if len(b) > 1 else 0.0)
        tm, ts = statistics.mean(t), (statistics.stdev(t) if len(t) > 1 else 0.0)
        print(f"{k:18} {bm:>11.4f}±{bs:<7.4f} {tm:>11.4f}±{ts:<7.4f} {tm - bm:>+14.4f}")
    print(f"\n結果 JSON: {OUT}")
    print("判定: treat の test_ndcg1 が base を seed をまたいで上回れば、履歴系列は "
          "本番アーキ×ROI目的でも accuracy を改善 (C の +0.018 を再現)。ROI は副次。")


if __name__ == "__main__":
    main()
