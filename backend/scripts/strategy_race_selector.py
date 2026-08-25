"""レースの「確証度」を複数ファクターから学習し、買うレースを選べるかを測る。

オッズ帯 (本命が 10〜25 倍) は「レース選別が効く」ことの証明にはなったが、単勝にしか
効かず、帯の外を一律に捨てる粗いルールでもある。ここでは代わりに

    p_true = f(モデルの確率, オッズ, 市場との乖離, 確信度, レース条件, ...)

を学習する。**これは要するに「市場の文脈を条件にした再較正」** で、今回オッズ水準が
回収率を予測したこと自体が「モデルの確率が市場に対して系統的に歪んでいる」証拠なので、
その歪みを直接学習させる。較正が直れば EV = p_true × odds が意味を取り戻し、
帯のように 10 倍以下を全部捨てる必要もなくなる (連続なスコアで順位付けできる)。

2 通りを比べる:
  A. 分類 → 本命の勝率 p_true を当て、EV_cal = p_true × オッズ でレースを並べる
  B. 回帰 → 実現リターンの条件付き期待値 E[return | x] を直接当てて並べる

**学習は dev (前半) のみ、報告は holdout (後半) のみ**。参照として
「全レース買う」「オッズ帯 10〜25」「素の EV (p̂ × オッズ)」も同じ holdout で測る。

Usage:
  PYTHONIOENCODING=utf-8 PYTHONPATH=src python -m scripts.strategy_race_selector \
      --csv ../data/analysis/strategy_signals_v2.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

# 結果側の列 (特徴量に混ぜてはいけない)
OUTCOME_COLS = [
    "top1_finish", "win_return", "place_return", "fav_win_return", "fav_place_return",
]
ID_COLS = ["race_id", "date"]
CAT_COLS = ["race_class", "course", "surface", "track_condition"]

# 5,390 レースしかないので木は小さく保つ。深い木は「たまたま回収率の高い
# レース群」を丸暗記できてしまい、holdout で消える。
MODEL_KW = dict(
    max_depth=3,
    max_iter=200,
    learning_rate=0.05,
    min_samples_leaf=60,
    l2_regularization=1.0,
    random_state=0,
)


def _features(d: pd.DataFrame) -> pd.DataFrame:
    x = d.drop(columns=[c for c in OUTCOME_COLS + ID_COLS if c in d.columns])
    for c in CAT_COLS:
        if c in x.columns:
            x[c] = x[c].astype("category")
    return x


def _coverage_curve(d: pd.DataFrame, score: np.ndarray, ret_col: str,
                    fractions=(0.05, 0.10, 0.15, 0.25, 0.50, 1.00)) -> list[dict]:
    """スコアの高い順に上位 x% を買ったときの回収率。"""
    order = np.argsort(-score)
    ret = d[ret_col].to_numpy(dtype=float)[order]
    out = []
    for f in fractions:
        k = max(1, int(round(len(ret) * f)))
        sel = ret[:k]
        sel = sel[np.isfinite(sel)]
        out.append({"買う割合": f, "レース数": len(sel), "回収率": float(sel.mean()) if len(sel) else float("nan")})
    return out


def _print_curve(title: str, rows: list[dict]) -> None:
    print(f"  {title}")
    for r in rows:
        print(f"    上位{r['買う割合'] * 100:>5.0f}%  n={r['レース数']:>5}  回収率 {r['回収率']:.3f}")


def _calibration_table(d: pd.DataFrame, p: np.ndarray, label: str) -> None:
    """予測確率の十分位ごとに、実際の勝率と比べる (較正が直ったかの確認)。"""
    q = pd.qcut(pd.Series(p), 10, labels=False, duplicates="drop")
    print(f"  {label}")
    print("    帯   予測勝率  実際の勝率   n")
    for b in sorted(pd.Series(q).dropna().unique()):
        m = q == b
        print(f"    {int(b):>2}   {p[m].mean():>8.3f}  {(d['top1_finish'].to_numpy()[m] == 1).mean():>9.3f}  {int(m.sum()):>4}")



def _decisive_tests(dev, hold, xd, xh, cat_mask) -> None:
    """決め手: **オッズ水準を超える情報があるか**。

    A/B の見かけの効果はほぼオッズ項だった (シャッフル対照が同等以上)。
    そこでオッズを固定した条件で、多ファクターのスコアが勝ち負けを分けられるかを見る。
    """
    print("\n=== 決定的な検証: オッズを超える情報はあるか ===")

    # (1) 素朴な「オッズ順」ベースライン
    print("\n  [1] 単にオッズの高い順に買う (学習なし)")
    _print_curve("", _coverage_curve(hold, hold.top1_odds.to_numpy(float), "win_return"))

    # (2) オッズ系を全部落とした特徴量で学習し、オッズと独立な signal があるかを見る
    odds_like = [c for c in xd.columns if any(
        k in c for k in ("odds", "implied", "popularity", "ev", "vs_market", "overround")
    )]
    print(f"\n  [2] オッズ系 {len(odds_like)} 列を除いて学習 → その並びで買う")
    print(f"      除いた列: {', '.join(odds_like)}")
    reg = HistGradientBoostingRegressor(
        categorical_features=[m for c, m in zip(xd.columns, cat_mask, strict=True) if c not in odds_like],
        **MODEL_KW,
    )
    m = dev.win_return.notna()
    reg.fit(xd.drop(columns=odds_like)[m.to_numpy()], dev.loc[m, "win_return"])
    s_free = reg.predict(xh.drop(columns=odds_like))
    _print_curve("", _coverage_curve(hold, s_free, "win_return"))

    # (3) オッズ帯の中だけで、学習スコアが上下を分けられるか
    #     ここで差が出なければ「オッズが全部」で、多ファクター化の余地は無い。
    print("\n  [3] 本命オッズ 10-25 の中だけで、学習スコアの上位半分 vs 下位半分")
    band = hold[(hold.top1_odds >= 10) & (hold.top1_odds < 25)]
    xb = xh.loc[band.index]
    reg_all = HistGradientBoostingRegressor(categorical_features=cat_mask, **MODEL_KW)
    reg_all.fit(xd[m.to_numpy()], dev.loc[m, "win_return"])
    for name, score in (("全ファクター", reg_all.predict(xb)),
                        ("オッズ系を除く", reg.predict(xb.drop(columns=odds_like)))):
        med = np.median(score)
        hi = band.win_return.to_numpy(float)[score >= med]
        lo = band.win_return.to_numpy(float)[score < med]
        print(f"    {name:<14} 上位 n={len(hi):>4} 回収率 {hi.mean():.3f}  /  "
              f"下位 n={len(lo):>4} 回収率 {lo.mean():.3f}  (差 {hi.mean() - lo.mean():+.3f})")

    # (4) 同じ手続きを目的変数シャッフルで踏み、(3) の差がどれだけ出うるかを見る
    rng = np.random.default_rng(0)
    diffs = []
    for seed in range(20):
        r = HistGradientBoostingRegressor(categorical_features=cat_mask, random_state=seed,
                                          **{k: v for k, v in MODEL_KW.items() if k != "random_state"})
        r.fit(xd[m.to_numpy()], rng.permutation(dev.loc[m, "win_return"].to_numpy()))
        sc = r.predict(xb)
        med = np.median(sc)
        ret = band.win_return.to_numpy(float)
        diffs.append(ret[sc >= med].mean() - ret[sc < med].mean())
    diffs = np.array(diffs)
    print(f"    対照 (目的変数シャッフル 20 回): 差の平均 {diffs.mean():+.3f}, "
          f"標準偏差 {diffs.std():.3f}, 範囲 [{diffs.min():+.3f}, {diffs.max():+.3f}]")

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    args = ap.parse_args()

    d = pd.read_csv(args.csv).sort_values("date").reset_index(drop=True)
    cut = str(d["date"].iloc[len(d) // 2])
    dev, hold = d[d["date"] < cut].copy(), d[d["date"] >= cut].copy()
    print(f"dev {len(dev)} ({dev['date'].min()}〜{dev['date'].max()}) / "
          f"holdout {len(hold)} ({hold['date'].min()}〜{hold['date'].max()})\n")

    xd, xh = _features(dev), _features(hold)
    cat_mask = [c in CAT_COLS for c in xd.columns]

    print("=== 参照: 学習なしのルール (holdout) ===")
    print(f"  全レース買う            n={len(hold):>5}  単勝 {hold.win_return.mean():.3f}  "
          f"複勝 {hold.place_return.dropna().mean():.3f}")
    band = hold[(hold.top1_odds >= 10) & (hold.top1_odds < 25)]
    print(f"  本命オッズ 10-25        n={len(band):>5}  単勝 {band.win_return.mean():.3f}  "
          f"複勝 {band.place_return.dropna().mean():.3f}")
    print("  素の EV (p̂ × オッズ) で並べる:")
    _print_curve("", _coverage_curve(hold, hold.top1_ev.to_numpy(float), "win_return"))

    print("\n=== A. 分類 → EV_cal = p_true × オッズ で並べる ===")
    clf = HistGradientBoostingClassifier(categorical_features=cat_mask, **MODEL_KW)
    clf.fit(xd, (dev.top1_finish == 1).astype(int))
    p_hold = clf.predict_proba(xh)[:, 1]
    ev_cal = p_hold * hold.top1_odds.to_numpy(float)
    _print_curve("単勝リターン:", _coverage_curve(hold, ev_cal, "win_return"))
    _print_curve("同じ並びでの複勝リターン:", _coverage_curve(hold, ev_cal, "place_return"))
    print()
    _calibration_table(hold, hold.top1_win_prob.to_numpy(float), "較正前 (モデルの素の win_prob)")
    _calibration_table(hold, p_hold, "較正後 (市場の文脈を条件にした p_true)")

    print("\n=== B. 回帰 → E[リターン | x] を直接当てて並べる ===")
    for target, ret_col in (("win_return", "win_return"), ("place_return", "place_return")):
        m = dev[target].notna()
        reg = HistGradientBoostingRegressor(categorical_features=cat_mask, **MODEL_KW)
        reg.fit(xd[m.to_numpy()], dev.loc[m, target])
        _print_curve(f"{target} を予測して並べる:",
                     _coverage_curve(hold, reg.predict(xh), ret_col))

    print("\n=== 対照: 目的変数をシャッフルして同じ手順を踏む (見かけの効果の量) ===")
    rng = np.random.default_rng(0)
    y = (dev.top1_finish == 1).astype(int).to_numpy()
    clf2 = HistGradientBoostingClassifier(categorical_features=cat_mask, **MODEL_KW)
    clf2.fit(xd, rng.permutation(y))
    _print_curve("", _coverage_curve(hold, clf2.predict_proba(xh)[:, 1] * hold.top1_odds.to_numpy(float),
                                     "win_return"))
    print("  (目的変数を壊すと p はほぼ定数になり、EV_cal = p × オッズ は「オッズ順」に退化する。")
    print("   この列が学習ありと同等以上なら、効いていたのは学習ではなくオッズ項である。)")

    _decisive_tests(dev, hold, xd, xh, cat_mask)


if __name__ == "__main__":
    main()
