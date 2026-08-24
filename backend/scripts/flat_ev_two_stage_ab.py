"""flat_ev vs log_growth の二段階ペア比較 (案B の検証)。

`log_growth` は「資金の何%を賭けるか」= Kelly を前提にした目的関数だが、賭け金の決定
からは Kelly を廃止し 1 点定額 (`assign_flat_stakes`) に変更済みで、目的関数だけが
取り残されている。`flat_ev` はそのズレを埋める損失 (`ai/model/loss.py`)。

**単体では回せない**: flat_ev は「買わない = 損失 0」が床なので、ゼロから学習すると
順位を学ばずに崩壊する (2 エポックの試走で ndcg3 0.051 / log_growth は 0.577)。
docs の推奨レシピどおり **PL 事前学習 → fine-tune** の二段階でのみ意味を持つ。

そこで seed ごとに:

    stage1  plackett_luce で事前学習           (1 回だけ・両腕で共有)
    stage2a 同じ初期値から log_growth で fine-tune   ← baseline
    stage2b 同じ初期値から flat_ev  で fine-tune     ← treatment

同一初期値・同一データ・同一 seed の **paired 比較**にすることで、単独学習どうしを
比べるより seed ノイズに強くなる。このリポジトリは過去に「単一 seed で有望に見えて
multi-seed で霧散」を踏んでいる (docs/ai-model.md B1) ので seed は複数必須。

**副作用を出さないための細工**: stage1 は `--init-from` に渡すため model.pt を保存する
必要があり persist=True になる。そのままだと実 DB の model_runs に行が増え `data/models/`
も汚れるので、`KEIBA_DATA_DIR` をスクラッチに向け、`db=` にスクラッチ SQLite を渡す。
`prebuilt_frame` / `prebuilt_history` を渡すと train_nn は DB を読まないので、実データは
事前にこちらで読んでおけばよい (フレームはキャッシュヒットで 0.1 秒)。

Usage:
  PYTHONIOENCODING=utf-8 PYTHONPATH=src python -m scripts.flat_ev_two_stage_ab \\
      --seeds 42,1,7 --out ../data/reports/flat_ev_ab.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

# 学習に使う窓。既にフレームキャッシュがある範囲を使う (新規ビルドは ~240 races/min
# かかり、全期間だと 2.7 時間。ここが実験時間の支配項になる)。
FRAME_START, FRAME_END = "2024-11-02", "2026-05-31"
TRAIN_END = "2025-08-31"   # → valid 2025-09..11 / test 2025-12..2026-05
VALID_MONTHS, TEST_MONTHS = 3, 6

METRICS = [
    "test_tansho_roi", "test_fukusho_roi", "test_tansho_hit",
    "test_fukusho_hit", "test_ndcg3",
]


def _seed_everything(seed: int) -> None:
    import lightning as L  # noqa: N812
    L.seed_everything(seed, workers=True)


def _scratch_db(scratch: Path) -> Path:
    """model_runs だけ作った空 SQLite を用意する (実 DB を汚さないため)。"""
    import db.models  # noqa: F401 — Base.metadata を全モデルで埋めるため
    from db.base import Base
    from db.session import make_engine

    p = scratch / "scratch.db"
    eng = make_engine(p)
    Base.metadata.create_all(eng)
    eng.dispose()
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="42,1,7")
    ap.add_argument("--pretrain-epochs", type=int, default=10)
    ap.add_argument("--finetune-epochs", type=int, default=6)
    ap.add_argument("--finetune-lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    # --- 実データはここで読む (この時点ではまだ本物の KEIBA_DATA_DIR) ---
    from core.paths import db_path
    from db.session import make_engine, session_scope
    from features.builder import build_training_frame
    from features.history_sequence import build_history_sequences

    t0 = time.time()
    real_engine = make_engine(db_path())
    with session_scope(real_engine) as s:
        frame = build_training_frame(s, train_start=FRAME_START, train_end=FRAME_END)
    print(f"[frame] {len(frame)} rows / {frame['race_id'].nunique()} races "
          f"({time.time()-t0:.1f}s)", flush=True)
    t1 = time.time()
    with session_scope(real_engine) as s:
        history = build_history_sequences(s, max_len=15)
    real_engine.dispose()
    print(f"[history] {time.time()-t1:.1f}s", flush=True)

    # --- 以降の書き込みはすべてスクラッチへ ---
    scratch = Path(tempfile.mkdtemp(prefix="keiba_flat_ev_ab_"))
    os.environ["KEIBA_DATA_DIR"] = str(scratch)
    sdb = _scratch_db(scratch)
    print(f"[scratch] {scratch}", flush=True)

    from ai.training.train_nn import train_nn

    def run(loss: str, *, seed: int, epochs: int, persist: bool,
            init_from: Path | None, lr: float, monitor: str) -> dict:
        _seed_everything(seed)
        t = time.time()
        m = train_nn(
            db=sdb, train_end=TRAIN_END,
            valid_months=VALID_MONTHS, test_months=TEST_MONTHS,
            loss=loss, monitor=monitor, device=args.device,
            max_epochs=epochs, learning_rate=lr,
            prebuilt_frame=frame, prebuilt_history=history,
            init_from=init_from, persist=persist, fit_temperature=False,
        )
        m["_elapsed_s"] = round(time.time() - t, 1)
        return m

    rows: list[dict] = []
    try:
        for seed in seeds:
            print(f"\n===== seed {seed} =====", flush=True)

            pre = run("plackett_luce", seed=seed, epochs=args.pretrain_epochs,
                      persist=True, init_from=None, lr=1e-3, monitor="valid_ndcg3")
            init = Path(pre["model_dir"])
            print(f"[s{seed}] pretrain(PL) {pre['_elapsed_s']}s ndcg3={pre.get('test_ndcg3')}",
                  flush=True)

            for arm, loss in (("base", "log_growth"), ("treat", "flat_ev")):
                m = run(loss, seed=seed, epochs=args.finetune_epochs, persist=False,
                        init_from=init, lr=args.finetune_lr, monitor="valid_tansho_roi")
                keep = {k: m.get(k) for k in METRICS}
                rows.append({"seed": seed, "arm": arm, "loss": loss,
                             "elapsed_s": m["_elapsed_s"], **keep})
                print(f"[s{seed}] {arm:5s} {loss:11s} {m['_elapsed_s']:6.1f}s "
                      f"{json.dumps(keep)}", flush=True)

            # ここまでの paired delta を毎 seed 出す (途中で止めても読めるように)
            _print_deltas(rows)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        print(f"[scratch] removed {scratch}", flush=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved: {args.out}", flush=True)


def _print_deltas(rows: list[dict]) -> None:
    by = {(r["seed"], r["arm"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    done = [s for s in seeds if (s, "base") in by and (s, "treat") in by]
    if not done:
        return
    print("\n----- paired delta (flat_ev - log_growth) -----", flush=True)
    print(f"{'seed':>6} {'d_tan_roi':>10} {'d_fuk_roi':>10} {'d_t_hit':>9} "
          f"{'d_f_hit':>9} {'d_ndcg3':>9}")
    acc = {k: [] for k in METRICS}
    for s in done:
        b, t = by[(s, "base")], by[(s, "treat")]
        vals = []
        for k in METRICS:
            d = (t[k] - b[k]) if (t[k] is not None and b[k] is not None) else float("nan")
            acc[k].append(d)
            vals.append(d)
        print(f"{s:>6} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>9.4f} "
              f"{vals[3]:>9.4f} {vals[4]:>9.4f}")
    n = len(done)
    print(f"{'mean':>6} " + " ".join(
        f"{sum(acc[k])/n:>{w}.4f}"
        for k, w in zip(METRICS, (10, 10, 9, 9, 9), strict=True)
    ), flush=True)


if __name__ == "__main__":
    main()
