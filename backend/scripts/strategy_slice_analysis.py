"""strategy_signal_dump.py の CSV から「どのレースを買うと回収率が上がるか」を探す。

探索と検証を **時系列で分ける**。前半 (dev) で候補ルールを探し、後半 (holdout) で
そのまま測り直す。同じ期間で探して同じ期間で報告すると、5,000 レースに対して
数十通りの切り口を試した時点で「たまたま良かった帯」が必ず見つかるため。

Usage:
  PYTHONIOENCODING=utf-8 PYTHONPATH=src python -m scripts.strategy_slice_analysis \
      --csv ../data/analysis/strategy_signals.csv
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

BOOTSTRAP_N = 2000
RNG_SEED = 0


def _roi_ci(returns: np.ndarray) -> tuple[float, float, float]:
    """平均リターンと 95% ブートストラップ信頼区間 (レース単位の置換抽出)。"""
    if returns.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, returns.size, size=(BOOTSTRAP_N, returns.size))
    means = returns[idx].mean(axis=1)
    return float(returns.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _summarise(d: pd.DataFrame, label: str) -> dict:
    win = d["win_return"].dropna().to_numpy(dtype=float)
    place = d["place_return"].dropna().to_numpy(dtype=float)
    w, wlo, whi = _roi_ci(win)
    p, plo, phi = _roi_ci(place)
    return {
        "label": label,
        "races": int(len(d)),
        "win_roi": round(w, 4), "win_ci": (round(wlo, 3), round(whi, 3)),
        "place_roi": round(p, 4), "place_ci": (round(plo, 3), round(phi, 3)),
        "win_hit": round(float((d["top1_finish"] == 1).mean()), 4),
    }


def _fmt(s: dict) -> str:
    return (
        f"{s['label']:<34} n={s['races']:>5}  "
        f"単勝 {s['win_roi']:.3f} [{s['win_ci'][0]:.2f},{s['win_ci'][1]:.2f}]  "
        f"複勝 {s['place_roi']:.3f} [{s['place_ci'][0]:.2f},{s['place_ci'][1]:.2f}]"
    )


def _candidate_rules(dev: pd.DataFrame) -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    """賭ける前に判定できる単変数ルール。閾値は dev の分位点から作る。"""
    rules: dict[str, Callable[[pd.DataFrame], pd.Series]] = {}

    for lo, hi in [(1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, 1e9)]:
        rules[f"本命オッズ {lo}-{hi if hi < 1e9 else '∞'}"] = (
            lambda d, lo=lo, hi=hi: (d["top1_odds"] >= lo) & (d["top1_odds"] < hi)
        )
    rules["本命 = 1番人気"] = lambda d: d["top1_is_favorite"]
    rules["本命 ≠ 1番人気"] = lambda d: ~d["top1_is_favorite"]
    for k in (2, 3, 5):
        rules[f"本命が人気{k}番以内"] = lambda d, k=k: d["top1_popularity"] <= k

    for col, name in [
        ("prob_margin", "1位-2位の確率差"),
        ("score_margin", "1位-2位のスコア差"),
        ("top1_win_prob", "本命の単勝確率"),
        ("win_prob_entropy", "確率のばらつき"),
        ("n_runners", "頭数"),
        ("mean_starts", "平均出走数"),
    ]:
        qs = dev[col].quantile([0.25, 0.5, 0.75]).to_dict()
        for q, v in qs.items():
            rules[f"{name} ≥ {v:.3g} (上位{int((1 - q) * 100)}%)"] = (
                lambda d, col=col, v=v: d[col] >= v
            )
            rules[f"{name} < {v:.3g} (下位{int(q * 100)}%)"] = (
                lambda d, col=col, v=v: d[col] < v
            )

    for cls in dev["race_class"].dropna().unique():
        if (dev["race_class"] == cls).sum() >= 200:
            rules[f"クラス = {cls}"] = lambda d, cls=cls: d["race_class"] == cls
    rules["履歴のあるレースのみ"] = lambda d: d["debut_ratio"] < 0.5
    for t in (0.9, 1.0, 1.1):
        rules[f"本命の EV ≥ {t}"] = lambda d, t=t: d["top1_ev"] >= t
        rules[f"本命の EV < {t}"] = lambda d, t=t: d["top1_ev"] < t
    return rules


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--min-races", type=int, default=400,
                    help="この件数を下回るルールは候補にしない (推定が定まらないため)")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    d = pd.read_csv(args.csv)
    d = d.sort_values("date").reset_index(drop=True)
    cut = str(d["date"].iloc[len(d) // 2])
    dev = d[d["date"] < cut]
    hold = d[d["date"] >= cut]
    print(f"全 {len(d)} レース  dev: {len(dev)} ({dev['date'].min()}〜{dev['date'].max()})  "
          f"holdout: {len(hold)} ({hold['date'].min()}〜{hold['date'].max()})\n", flush=True)

    print("── 何もしない (全レースで本命 1 点) ──")
    print(_fmt(_summarise(dev, "dev 全レース")))
    print(_fmt(_summarise(hold, "holdout 全レース")))
    print(_fmt(_summarise(d, "全期間")))
    print()

    rules = _candidate_rules(dev)
    scored = []
    for name, fn in rules.items():
        sub = dev[fn(dev)]
        if len(sub) < args.min_races:
            continue
        s = _summarise(sub, name)
        scored.append(s)

    print(f"── dev で単勝回収率が高い順 (n ≥ {args.min_races}) ──")
    for s in sorted(scored, key=lambda x: -x["win_roi"])[: args.top]:
        print(_fmt(s))
        h = hold[rules[s["label"]](hold)]
        print("   → holdout: " + _fmt(_summarise(h, "")).strip())
    print()
    print(f"── dev で複勝回収率が高い順 (n ≥ {args.min_races}) ──")
    for s in sorted(scored, key=lambda x: -x["place_roi"])[: args.top]:
        print(_fmt(s))
        h = hold[rules[s["label"]](hold)]
        print("   → holdout: " + _fmt(_summarise(h, "")).strip())


if __name__ == "__main__":
    main()
