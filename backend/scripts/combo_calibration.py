"""連系の確率が「実際に出た組み合わせ」にどれだけ確率を与えられているかを測る。

**なぜこれを見るか**: 連系の確率は NN スコアから Plackett-Luce サンプリングで導出する
(`compute_all_combination_probs(frame_scores, ...)`)。この導出は `exp(score)` が PL の
強度パラメータであることを前提にしている。ところが本番の active は `log_growth` 系
(単勝の回収率) で学習されており、スコアの**大きさ**が PL パラメータである保証が無い。
単勝で測った「確率と勝敗の相関 0.073」がその症状で、連系は複数頭のスコア差の積で
決まるぶん誤差が増幅されるはず。

`plackett_luce` 損失は「スコアを PL の最尤パラメータに合わせる」学習なので、
**導出の前提と学習の目的が初めて一致する**。ここではその差を、賭け方を一切挟まずに
**proper scoring rule (実際に出た組み合わせの対数確率)** で直接比べる。

回収率ではなく確率の質を見るのは、賭け方 (EV 閾値・点数上限・予算) を挟むと
「確率が直ったのか、買い方がたまたま噛み合ったのか」が分離できないため。

Usage:
  PYTHONIOENCODING=utf-8 PYTHONPATH=src python -m scripts.combo_calibration \
      --models ../data/models/20260613T114817-nn,../data/models/20260825T084014-nn \
      --start 2024-11-02 --end 2026-05-31 --out ../data/analysis/combo_calibration.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ai.core.probabilities import compute_all_combination_probs
from ai.inference.predict import _predict_race_nn
from ai.model.registry import load_model_full
from core.paths import db_path
from db.session import make_engine, session_scope
from features.builder import build_training_frame

#: 確率が 0 のときに log が発散しないための下限。10,000 サンプルの MC なので
#: 1/n_samples より小さい確率は「観測できていない」= この値で頭打ちにする。
EPS = 1e-4


def _actual_top3(rf: pd.DataFrame) -> list[str] | None:
    """1〜3 着の horse_id。着順が欠けていれば None。"""
    fin = rf[["horse_id", "finish_position"]].dropna()
    if fin.empty:
        return None
    top = fin[fin["finish_position"] <= 3].sort_values("finish_position")
    if len(top) < 3:
        return None
    return list(top["horse_id"].values)


def _row_for_race(scores: np.ndarray, horse_ids: list[str], top3: list[str],
                  n_samples: int, rng) -> dict:
    """実際に出た組み合わせに、そのモデルが与えた確率。"""
    idx = {h: i for i, h in enumerate(horse_ids)}
    i1, i2, i3 = idx[top3[0]], idx[top3[1]], idx[top3[2]]
    cp = compute_all_combination_probs(scores, k=3, n_samples=n_samples, rng=rng)

    triple = cp["triple"].get(frozenset((i1, i2, i3)), 0.0)
    return {
        "単勝": float(cp["position"][i1, 0]),
        "馬連": float(cp["pair"][i1, i2]),
        "馬単": float(cp["ordered_pair"][i1, i2]),
        "三連複": float(triple),
        "三連単": float(cp["ordered_triple"][i1, i2, i3]),
        # ワイドは 3 通りのうち 1〜2 着の組を代表に採る
        "ワイド": float(cp["pair"][i1, i2]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", required=True, help="カンマ区切りのモデルディレクトリ")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--n-samples", type=int, default=20_000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    model_dirs = [Path(m) for m in args.models.split(",")]
    bundles = [(d.name, load_model_full(d)) for d in model_dirs]
    for name, _ in bundles:
        print(f"model: {name}", flush=True)

    engine = make_engine(db_path())
    rows: list[dict] = []
    # session はループ外で保持する (履歴 GRU が zero に degrade するのを防ぐ)
    with session_scope(engine) as session:
        frame = build_training_frame(session, train_start=args.start, train_end=args.end)
        race_ids = frame["race_id"].unique()
        print(f"{len(race_ids)} races", flush=True)
        for i, rid in enumerate(race_ids):
            rf = frame[frame["race_id"] == rid]
            if len(rf) < 4:
                continue
            top3 = _actual_top3(rf)
            if top3 is None:
                continue
            horse_ids = list(rf["horse_id"].values)
            if any(h not in horse_ids for h in top3):
                continue
            # 同じ乱数列を全モデルで使う (MC のばらつきで差がつかないように)
            row = {"race_id": rid, "date": rf["date"].iloc[0], "n_runners": len(rf)}
            for name, bundle in bundles:
                try:
                    base = _predict_race_nn(bundle, rf, session=session)
                except Exception as exc:  # noqa: BLE001
                    print(f"  predict failed {rid} ({name}): {exc}", flush=True)
                    row = None
                    break
                score_map = dict(zip(base["horse_id"].values, base["score"].values, strict=True))
                scores = np.array([score_map[h] for h in horse_ids], dtype=float)
                probs = _row_for_race(scores, horse_ids, top3,
                                      args.n_samples, np.random.default_rng(0))
                for bt, p in probs.items():
                    row[f"{name}::{bt}"] = p
            if row is not None:
                rows.append(row)
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(race_ids)}", flush=True)

    d = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.out, index=False, encoding="utf-8")
    print(f"wrote {len(d)} races -> {args.out}\n", flush=True)

    names = [n for n, _ in bundles]
    print("実際に出た組み合わせの平均対数確率 (0 に近いほど良い / proper scoring rule)")
    header = f"{'券種':<8}" + "".join(f"{n[:18]:>22}" for n in names)
    print(header)
    for bt in ("単勝", "馬連", "馬単", "三連複", "三連単"):
        cells = ""
        for n in names:
            col = d[f"{n}::{bt}"].to_numpy(float)
            cells += f"{float(np.log(np.clip(col, EPS, 1.0)).mean()):>22.4f}"
        print(f"{bt:<8}{cells}")
    print()
    print(f"参考: 実際に出た組み合わせに与えた平均確率 (下限 {EPS})")
    for bt in ("単勝", "馬連", "馬単", "三連複", "三連単"):
        cells = "".join(f"{d[f'{n}::{bt}'].mean():>22.5f}" for n in names)
        print(f"{bt:<8}{cells}")


if __name__ == "__main__":
    main()
