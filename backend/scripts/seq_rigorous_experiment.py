"""厳密検証: 履歴系列エンコーダの ROI lift は本物か (大穴ノイズ vs 本物のエッジ)?

no-odds 2seed の暫定で treat(履歴) が test_tansho_roi を +0.10 一貫上昇させた。
過去の ROI シグナルは例外なく race-level bootstrap CI で消えたため、ここで:
  - {no_odds, with_odds} × {base(集約), treat(履歴)} × 3 seed を学習
  - 各 test 賭けの per-race 記録 (top-1 のオッズ/的中/払戻) を集め
  - race-level bootstrap 95% CI と 購入馬のオッズ分布 (大穴率) を出す
ことで、+0.10 が「妙味=大穴を引いただけ」か統計的に有意なエッジかを判定する。

フレームと履歴は 1 度だけ構築して全 config で再利用 (prebuilt)。persist=False。
ビルド段階のみ keiba.db を read する (以降は GPU のみ)。

使い方:
  UV_PROJECT_ENVIRONMENT=/tmp/keiba-linux-venv PYTHONPATH=src \
      uv run python scripts/seq_rigorous_experiment.py
"""

from __future__ import annotations

import json
import os

import numpy as np
from lightning.pytorch import seed_everything

from ai.training.train_nn import train_nn
from core.paths import data_dir, db_path
from db.session import make_engine, session_scope
from features.builder import build_training_frame
from features.history_sequence import build_history_sequences

TRAIN_END = "2025-04-30"
VALID_MONTHS = 6
TEST_MONTHS = 6
LOSS = "log_growth"
MONITOR = "valid_tansho_roi"
DEVICE = "cuda"
MAX_EPOCHS = 50
PATIENCE = 8
HISTORY_SEQ_LEN = 15
SEEDS = [0, 1, 2]
ODDS_MODES = [("no_odds", "1"), ("with_odds", "")]
ARMS = [("base", False), ("treat", True)]
N_BOOT = 3000

OUT = data_dir() / "cache" / "seq_rigorous_results.json"


def _roi_and_ci(records: list[dict], key: str, rng: np.random.Generator) -> tuple[float, float, float, int]:
    """ROI (= mean return, stake 1) と race-level bootstrap 95% CI。"""
    rets = np.array([r[key] for r in records if r.get(key) is not None], dtype=float)
    if len(rets) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    roi = float(rets.mean())
    boot = np.empty(N_BOOT)
    n = len(rets)
    for i in range(N_BOOT):
        boot[i] = rets[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return roi, float(lo), float(hi), n


def _odds_profile(records: list[dict]) -> dict:
    odds = np.array([r["odds"] for r in records if r.get("odds") is not None], dtype=float)
    if len(odds) == 0:
        return {"mean": float("nan"), "median": float("nan"), "pct_gt20": float("nan")}
    return {
        "mean": float(np.mean(odds)),
        "median": float(np.median(odds)),
        "pct_gt20": float(np.mean(odds > 20.0) * 100.0),
    }


def main() -> None:
    eng = make_engine(db_path())
    with session_scope(eng) as session:
        frame = build_training_frame(session)
    print(f"frame built: {len(frame):,} rows, {frame['race_id'].nunique():,} races", flush=True)
    with session_scope(eng) as session:
        history_cache = build_history_sequences(session, max_len=HISTORY_SEQ_LEN)
    print(f"history built: {len(history_cache.seqs):,} seqs, {history_cache.n_features} feats", flush=True)

    # (odds_mode, arm) -> {"per_seed_roi": [...], "bets": [...pooled...]}
    agg: dict[tuple[str, str], dict] = {}
    metrics_log: dict[str, list[dict]] = {}
    OUT.parent.mkdir(parents=True, exist_ok=True)

    for odds_mode, env_val in ODDS_MODES:
        if env_val:
            os.environ["KEIBA_EXCLUDE_ODDS_FEATURES"] = env_val
        else:
            os.environ.pop("KEIBA_EXCLUDE_ODDS_FEATURES", None)
        for arm, use_hist in ARMS:
            key = (odds_mode, arm)
            agg[key] = {"per_seed_roi": [], "bets": []}
            for seed in SEEDS:
                print(f"\n=== {odds_mode} / {arm} / seed={seed} ===", flush=True)
                seed_everything(seed, workers=True)
                res = train_nn(
                    prebuilt_frame=frame,
                    prebuilt_history=history_cache if use_hist else None,
                    use_history=use_hist,
                    history_seq_len=HISTORY_SEQ_LEN,
                    train_end=TRAIN_END, valid_months=VALID_MONTHS, test_months=TEST_MONTHS,
                    loss=LOSS, monitor=MONITOR, device=DEVICE,
                    max_epochs=MAX_EPOCHS, early_stopping_patience=PATIENCE,
                    persist=False, fit_temperature=False, return_test_bets=True,
                )
                agg[key]["per_seed_roi"].append(res.get("test_tansho_roi"))
                agg[key]["bets"].extend(res.get("test_bets", []))
                metrics_log.setdefault(f"{odds_mode}/{arm}", []).append({
                    "seed": seed,
                    "test_ndcg1": res.get("test_ndcg1"),
                    "test_tansho_roi": res.get("test_tansho_roi"),
                    "test_fukusho_roi": res.get("test_fukusho_roi"),
                })
                OUT.write_text(json.dumps(metrics_log, ensure_ascii=False, indent=2))
                print(f"  ndcg1={res.get('test_ndcg1'):.4f} tansho_roi={res.get('test_tansho_roi'):.4f} "
                      f"(bets pooled={len(agg[key]['bets'])})", flush=True)

    # ---- 分析: race-level bootstrap CI + オッズ分布 ----
    rng = np.random.default_rng(0)
    print("\n" + "=" * 78)
    print(f"厳密検証  (loss={LOSS}, monitor={MONITOR}, seeds={SEEDS}, pooled bootstrap n={N_BOOT})")
    print("=" * 78)
    for odds_mode, _ in ODDS_MODES:
        print(f"\n--- {odds_mode} ---")
        print(f"{'arm':6} {'tansho ROI':>11} {'95% CI':>20} {'fukusho ROI':>12} "
              f"{'pick odds mean/med':>20} {'%odds>20':>9} {'nbets':>7}")
        for arm, _ in ARMS:
            rec = agg[(odds_mode, arm)]["bets"]
            t_roi, t_lo, t_hi, n_t = _roi_and_ci(rec, "tansho_ret", rng)
            f_roi, _flo, _fhi, _nf = _roi_and_ci(rec, "place_ret", rng)
            od = _odds_profile(rec)
            print(f"{arm:6} {t_roi:>11.4f} [{t_lo:>6.3f},{t_hi:>6.3f}]    {f_roi:>12.4f} "
                  f"{od['mean']:>9.1f}/{od['median']:<8.1f} {od['pct_gt20']:>8.1f}% {n_t:>7}")
        # treat-base 差の有意性メモ
        base_roi = float(np.nanmean(agg[(odds_mode, 'base')]["per_seed_roi"]))
        treat_roi = float(np.nanmean(agg[(odds_mode, 'treat')]["per_seed_roi"]))
        print(f"  per-seed mean tansho ROI: base={base_roi:.4f} treat={treat_roi:.4f} "
              f"Δ={treat_roi - base_roi:+.4f}  (per-seed: "
              f"base={[round(x,3) for x in agg[(odds_mode,'base')]['per_seed_roi']]} "
              f"treat={[round(x,3) for x in agg[(odds_mode,'treat')]['per_seed_roi']]})")

    print(f"\n結果 JSON: {OUT}")
    print("判定: treat の tansho ROI CI 下限が base の ROI を上回り、かつ pick odds 分布が "
          "base と大きく変わらなければ『本物のエッジ』寄り。treat の %odds>20 が高い・CI が広いなら "
          "『妙味=大穴ノイズ』寄り。いずれも ROI<1.0 なら黒字化せず (市場効率の壁)。")


if __name__ == "__main__":
    main()
