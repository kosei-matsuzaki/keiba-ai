"""二段目 (active への信用度) を OOF で学習し、単純ルールを超えるかを測る。

一段目の out-of-sample 出力 (`scripts/walk_forward_oof.py` が生成) を入力に、
**「active の本命がどれくらい信用できるか」**を学習する。狙いは、手で決めた
しきい値 (オッズ帯・PL 確率の上位 x%) を、連続な信用度に置き換えること。

**判定基準**: OOF の前半で学習し後半で測る。単純ルール
(PL の確率をそのまま信用度に使う) を **holdout で上回らなければ採用しない**。
複雑にするだけの価値が無いため。

Usage:
  PYTHONIOENCODING=utf-8 PYTHONPATH=src python -m scripts.second_stage_trust
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sqlalchemy import text

from ai.evaluation.backtest import _parse_payout_place
from core.paths import data_dir, db_path
from db.session import make_engine, session_scope

CAT = ["race_class"]
MODEL_KW = dict(max_depth=3, max_iter=300, learning_rate=0.05,
                min_samples_leaf=80, l2_regularization=1.0, random_state=0)


def _load() -> pd.DataFrame:
    base = data_dir() / "analysis" / "oof"
    rd = lambda f: pd.read_csv(f, dtype={"race_id": str, "horse_id": str})  # noqa: E731
    m = pd.concat([rd(f) for f in sorted(base.glob("oof_*_multi.csv"))], ignore_index=True)
    p = pd.concat([rd(f) for f in sorted(base.glob("oof_*_plackett_luce.csv"))], ignore_index=True)

    # active 相当 (multi) の本命 1 頭が 1 行
    top = m[m.model_rank == 0].copy()
    # 同じレースの 2・3 番手との開き (確信度の材料)
    m2 = m[m.model_rank == 1][["race_id", "score", "win_prob"]].rename(
        columns={"score": "score2", "win_prob": "win_prob2"})
    top = top.merge(m2, on="race_id", how="left")
    # レース単位の市場情報
    mk = m.groupby("race_id").agg(
        overround=("odds_win", lambda s: float((1.0 / s.dropna()).sum())),
        log_odds_std=("odds_win", lambda s: float(np.log(s.dropna()).std())),
        min_odds=("odds_win", "min"),
    ).reset_index()
    top = top.merge(mk, on="race_id", how="left")
    # PL 側: 同じ馬に対する確率と順位、および PL 自身の本命との一致
    pl = p[["race_id", "horse_id", "score", "win_prob", "place_prob", "model_rank"]].rename(
        columns={"score": "pl_score", "win_prob": "pl_p",
                 "place_prob": "pl_place", "model_rank": "pl_rank"})
    d = top.merge(pl, on=["race_id", "horse_id"], how="inner")
    d["models_agree"] = (d.pl_rank == 0).astype(int)
    d["prob_margin"] = d.win_prob - d.win_prob2
    d["score_margin"] = d.score - d.score2
    d["implied"] = 1.0 / d.odds_win
    d["prob_vs_market"] = d.pl_p / d.implied

    # 複勝の払戻
    eng = make_engine(db_path())
    with session_scope(eng) as s:
        pay = dict(s.execute(
            text("SELECT race_id, payout_place FROM races WHERE payout_place IS NOT NULL")
        ).all())
    eng.dispose()

    def _pr(rid, fin):
        raw = pay.get(str(rid))
        if raw is None or pd.isna(fin):
            return np.nan
        mp = _parse_payout_place(raw)
        f = int(fin)
        if f in mp:
            return mp[f] / 100.0
        return 0.0 if f > 3 else np.nan

    d["place_return"] = [_pr(r, f) for r, f in zip(d.race_id, d.finish_position, strict=True)]
    d["win_return"] = np.where(d.finish_position == 1, d.odds_win, 0.0)
    d["won"] = (d.finish_position == 1).astype(int)
    d["placed"] = (d.finish_position <= 3).astype(int)
    return d.sort_values("date").reset_index(drop=True)


FEATS = [
    "odds_win", "popularity", "n_runners", "race_class", "train_races",
    "score", "win_prob", "place_prob", "score2", "win_prob2",
    "prob_margin", "score_margin", "implied", "prob_vs_market",
    "overround", "log_odds_std", "min_odds",
    "pl_score", "pl_p", "pl_place", "pl_rank", "models_agree",
]


def _curve(d: pd.DataFrame, score: np.ndarray, col: str, label: str) -> None:
    order = np.argsort(-score)
    ret = d[col].to_numpy(float)[order]
    out = []
    for f in (0.10, 0.25, 0.50, 1.00):
        k = max(1, int(round(len(ret) * f)))
        sel = ret[:k][np.isfinite(ret[:k])]
        out.append(f"上位{int(f * 100):>3}% {sel.mean():.3f}")
    print(f"  {label:<28}" + "  ".join(out))


def main() -> None:
    d = _load()
    cut = str(d["date"].iloc[len(d) // 2])
    tr, te = d[d["date"] < cut], d[d["date"] >= cut]
    print(f"OOF {len(d):,} レース  学習 {len(tr):,} ({tr.date.min()}〜{tr.date.max()}) / "
          f"検証 {len(te):,} ({te.date.min()}〜{te.date.max()})\n")

    xtr, xte = tr[FEATS].copy(), te[FEATS].copy()
    for c in CAT:
        xtr[c] = xtr[c].astype("category")
        xte[c] = xte[c].astype("category")
    cat_mask = [c in CAT for c in FEATS]

    for target, col, name in (("won", "win_return", "単勝"), ("placed", "place_return", "複勝")):
        print(f"=== {name} ===")
        _curve(te, te.pl_p.to_numpy(float), col, "単純ルール (PL の確率)")
        _curve(te, -te.odds_win.to_numpy(float), col, "参考: 人気順 (オッズ低い順)")
        clf = HistGradientBoostingClassifier(categorical_features=cat_mask, **MODEL_KW)
        clf.fit(xtr, tr[target])
        _curve(te, clf.predict_proba(xte)[:, 1], col, "二段目 (全ファクター)")
        # 目的変数を壊した対照
        rng = np.random.default_rng(0)
        clf2 = HistGradientBoostingClassifier(categorical_features=cat_mask, **MODEL_KW)
        clf2.fit(xtr, rng.permutation(tr[target].to_numpy()))
        _curve(te, clf2.predict_proba(xte)[:, 1], col, "対照 (目的変数シャッフル)")
        print()


if __name__ == "__main__":
    main()
