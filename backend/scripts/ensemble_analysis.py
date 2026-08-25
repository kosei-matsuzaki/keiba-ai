"""役割分担 (active が買う馬を決め、確率モデルが確からしさを答える) の効き目を測る。

判定は 3 段階。**前半 dev で見て、後半 holdout で確かめる**。

  1. 確率モデルの確率は情報を持つか (相関・log-loss を active と市場に対して比較)
  2. その確率でレースを選ぶと回収率が上がるか (買う割合 vs 回収率)
  3. **オッズ帯を超えるか** — これが本番。オッズ 10-25 の帯だけで、確率モデルの
     スコアが上下を分けられるかを見る。分けられなければ、役割分担しても
     「オッズを見ているだけ」ということになる。

Usage:
  PYTHONIOENCODING=utf-8 PYTHONPATH=src python -m scripts.ensemble_analysis \
      --csv ../data/analysis/ensemble_signals.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)


def _ci(x) -> tuple[float, float, float]:
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    if x.size == 0:
        return (float("nan"),) * 3
    b = x[RNG.integers(0, x.size, size=(4000, x.size))].mean(axis=1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _log_loss(y, p) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _row(label: str, d: pd.DataFrame) -> str:
    w, wl, wh = _ci(d.win_return)
    p, pl, ph = _ci(d.place_return)
    return (f"{label:<30}n={len(d):>5}  単勝 {w:.3f} [{wl:.2f},{wh:.2f}]  複勝 {p:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    args = ap.parse_args()

    d = pd.read_csv(args.csv).sort_values("date").reset_index(drop=True)
    cut = str(d["date"].iloc[len(d) // 2])
    dev, hold = d[d["date"] < cut], d[d["date"] >= cut]
    y = (hold.finish == 1).astype(int)
    mkt = 1.0 / hold.bet_odds

    print(f"dev {len(dev)} / holdout {len(hold)}  (境界 {cut})\n")

    print("=== 1. 確率は情報を持つか (holdout) ===")
    print(f"{'確率の出どころ':<24}{'相関':>8}{'log-loss':>10}{'平均':>8}")
    for name, q in (("active (multi)", hold.bet_win_prob),
                    ("確率モデル (PL)", hold.prob_model_p),
                    ("市場 (1/オッズ)", mkt)):
        print(f"{name:<24}{np.corrcoef(q, y)[0, 1]:>8.4f}{_log_loss(y, q):>10.4f}{q.mean():>8.3f}")
    print(f"{'実際の勝率':<24}{'':>8}{'':>10}{y.mean():>8.3f}")

    print("\n=== 2. 確からしさでレースを選ぶ (holdout) ===")
    print(_row("何もしない (全レース)", hold))
    print(_row("2 モデルが一致したレースのみ", hold[hold.models_agree]))
    print(_row("一致しなかったレースのみ", hold[~hold.models_agree]))
    for col, name in (("prob_model_p", "確率モデルの確率"), ("prob_model_ev", "確率モデルの EV")):
        for q in (0.5, 0.75, 0.9):
            thr = dev[col].quantile(q)          # 閾値は dev で決める
            print(_row(f"{name} 上位{int((1 - q) * 100)}% (≥{thr:.3g})", hold[hold[col] >= thr]))

    print("\n=== 3. オッズ帯を超えるか (holdout・帯の中だけ) ===")
    band = hold[(hold.bet_odds >= 10) & (hold.bet_odds < 25)]
    print(_row("オッズ 10-25 (基準)", band))
    for col, name in (("prob_model_p", "確率モデルの確率"), ("prob_model_ev", "確率モデルの EV")):
        med = dev[(dev.bet_odds >= 10) & (dev.bet_odds < 25)][col].median()
        hi, lo = band[band[col] >= med], band[band[col] < med]
        print(_row(f"  └ {name} 上位半分", hi))
        print(_row(f"  └ {name} 下位半分", lo))
    ag = band[band.models_agree]
    print(_row("  └ 2 モデルが一致", ag))
    print(_row("  └ 一致せず", band[~band.models_agree]))

    print("\n=== 参考: 確率モデル単体で本命を買った場合 (全期間) ===")
    print(f"  active の本命:     単勝 {d.win_return.mean():.4f}")
    print(f"  確率モデルの本命:  単勝 {d.prob_pick_win_return.mean():.4f}")
    print(f"  2 モデルの一致率:  {d.models_agree.mean():.3f}")


if __name__ == "__main__":
    main()
