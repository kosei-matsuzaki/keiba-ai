"""統制 A/B: odds をスコアリング head へ移す再設計 + per-race 履歴の効果検証。

4 構成を同一フレーム/履歴/split/seed で比較し、ROI + 的中率 + race-level
bootstrap CI + 購入馬オッズ分布を出す:
  baseline_v2 : odds は encoder (現行), 履歴なし
  odds_head   : odds は head (ability/value 分離), 履歴なし
  history     : odds は encoder, per-race 履歴あり
  target      : odds は head, per-race 履歴あり  ← ユーザー方針の本命

with-odds (odds は常に使う。encoder か head かが違う)。loss=multi(本番)。
フレーム+履歴は 1 回構築して再利用。persist=False (実験)。

使い方:
  UV_PROJECT_ENVIRONMENT=/tmp/keiba-linux-venv PYTHONPATH=src \
      uv run python scripts/odds_head_experiment.py
"""

from __future__ import annotations

import json

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
LOSS = "multi"             # 本番 all-markets 目的
MONITOR = "valid_tansho_roi"
DEVICE = "cuda"
MAX_EPOCHS = 50
PATIENCE = 8
HISTORY_SEQ_LEN = 15
SEEDS = [0, 1]
N_BOOT = 3000
# (name, use_history, use_odds_head)
CONFIGS = [
    ("baseline_v2", False, False),
    ("odds_head", False, True),
    ("history", True, False),
    ("target", True, True),
]
OUT = data_dir() / "cache" / "odds_head_results.json"


def _roi_ci(records: list[dict], key: str, rng) -> tuple[float, float, float, int]:
    rets = np.array([r[key] for r in records if r.get(key) is not None], dtype=float)
    if len(rets) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    roi = float(rets.mean())
    n = len(rets)
    boot = np.array([rets[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return roi, float(lo), float(hi), n


def _hit(records: list[dict], gate: str, hit) -> float:
    rs = [r for r in records if r.get(gate) is not None]
    return float(sum(1 for r in rs if hit(r)) / len(rs)) if rs else float("nan")


def _odds_profile(records: list[dict]) -> tuple[float, float, float]:
    odds = np.array([r["odds"] for r in records if r.get("odds") is not None], dtype=float)
    if len(odds) == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(odds)), float(np.median(odds)), float(np.mean(odds > 20) * 100)


def main() -> None:
    eng = make_engine(db_path())
    with session_scope(eng) as s:
        frame = build_training_frame(s)
    print(f"frame built: {len(frame):,} rows, {frame['race_id'].nunique():,} races", flush=True)
    with session_scope(eng) as s:
        history_cache = build_history_sequences(s, max_len=HISTORY_SEQ_LEN)
    print(f"history built: {len(history_cache.seqs):,} seqs", flush=True)

    agg: dict[str, dict] = {}
    log: dict[str, list[dict]] = {}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for name, use_hist, use_odds in CONFIGS:
        agg[name] = {"per_seed_roi": [], "bets": []}
        for seed in SEEDS:
            print(f"\n=== {name} (history={use_hist}, odds_head={use_odds}) seed={seed} ===", flush=True)
            seed_everything(seed, workers=True)
            res = train_nn(
                prebuilt_frame=frame,
                prebuilt_history=history_cache if use_hist else None,
                use_history=use_hist, history_seq_len=HISTORY_SEQ_LEN,
                use_odds_head=use_odds,
                train_end=TRAIN_END, valid_months=VALID_MONTHS, test_months=TEST_MONTHS,
                loss=LOSS, monitor=MONITOR, device=DEVICE,
                max_epochs=MAX_EPOCHS, early_stopping_patience=PATIENCE,
                persist=False, fit_temperature=False, return_test_bets=True,
            )
            agg[name]["per_seed_roi"].append(res.get("test_tansho_roi"))
            agg[name]["bets"].extend(res.get("test_bets", []))
            log.setdefault(name, []).append({
                "seed": seed,
                "test_tansho_roi": res.get("test_tansho_roi"),
                "test_tansho_hit": res.get("test_tansho_hit"),
                "test_fukusho_hit": res.get("test_fukusho_hit"),
            })
            OUT.write_text(json.dumps(log, ensure_ascii=False, indent=2))
            print(f"  tansho_roi={res.get('test_tansho_roi'):.4f} "
                  f"tansho_hit={res.get('test_tansho_hit'):.4f} "
                  f"fukusho_hit={res.get('test_fukusho_hit'):.4f}", flush=True)

    rng = np.random.default_rng(0)
    print("\n" + "=" * 88)
    print(f"odds-at-scoring 統制 A/B (loss={LOSS}, seeds={SEEDS}, bootstrap n={N_BOOT})")
    print("=" * 88)
    print(f"{'config':13} {'tansho ROI':>11} {'95% CI':>15} {'tansho_hit':>10} "
          f"{'fuku_hit':>9} {'odds m/med':>13} {'%>20':>6} {'nbets':>7}")
    for name, _, _ in CONFIGS:
        rec = agg[name]["bets"]
        t_roi, lo, hi, n = _roi_ci(rec, "tansho_ret", rng)
        t_hit = _hit(rec, "tansho_ret", lambda r: bool(r["won"]))
        f_hit = _hit(rec, "place_ret", lambda r: r["place_ret"] > 0)
        m, med, gt20 = _odds_profile(rec)
        print(f"{name:13} {t_roi:>11.4f} [{lo:>5.2f},{hi:>5.2f}] {t_hit:>10.3f} "
              f"{f_hit:>9.3f} {m:>6.1f}/{med:<6.1f} {gt20:>5.1f}% {n:>7}")
    print(f"\n結果 JSON: {OUT}")
    print("判定: target/odds_head の tansho ROI CI 下限が baseline_v2 を上回り、"
          "的中率も維持/向上すれば odds 分離が有効。ROI<1.0 なら黒字化せず (市場効率)。")


if __name__ == "__main__":
    main()
