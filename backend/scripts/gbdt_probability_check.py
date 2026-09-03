"""LightGBM の勝率が、市場 (1/オッズ) より正しいかを測る。

**回収率は見ない。確率だけを見る。** 理由:

1 を超えるには「市場が間違っているところだけ買う」しかない。全レース買えば控除率
(単勝 20%) に算術的に負ける。そして選別するには「自分の確率が市場より正しい」と
言えなければならない。いまの active はそれが言えない — 本命の win_prob と勝敗の
相関が 0.073、市場 (1/オッズ) は 0.354。ROI 志向の損失は順序しか最適化しないため。

だからここで測るのは log-loss だけ。ROI は seed 間で ±0.11 も動いて、この規模では
0.02〜0.05 の効果を解像できない (2026-09-02 の A/B 4 本で確認)。log-loss は安定して
いるので、少ない試行で判定できる。

出す数字:
  field NLL   レース内で正規化した勝率の -log p(勝ち馬)。真の proper scoring rule
  top NLL     モデルの本命 1 頭についての二値 NLL。backtest の log_loss と同じ量
              (CLAUDE.md 記載の active 0.574 / 市場 0.483 と直接比べられる)
  corr        本命の予測確率と実際の勝敗の相関 (active 0.073 / 市場 0.354)

odds あり / なしの 2 本を出す。**なし** の方が本題で、市場を見ずに市場を上回れるか
= 独立な情報を持っているか、を測る。あり は上限の目安。

Usage:
    PYTHONPATH=src uv run python -m scripts.gbdt_probability_check
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from core.logging import configure_logging, get_logger
from core.paths import db_path
from db.session import make_engine, session_scope
from features.builder import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    ODDS_FEATURE_COLUMNS,
    build_training_frame,
)

log = get_logger(__name__)

_EPS = 1e-9


def _split(frame: pd.DataFrame, train_end: str, valid_end: str):
    d = frame["date"]
    return (
        frame[d <= train_end],
        frame[(d > train_end) & (d <= valid_end)],
        frame[d > valid_end],
    )


def _prepare(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """LightGBM に渡す X。カテゴリは category dtype にして native 扱いさせる。"""
    x = frame[cols].copy()
    for c in cols:
        if c in CATEGORICAL_FEATURES:
            x[c] = x[c].astype("category")
    return x


def _race_normalised(frame: pd.DataFrame, raw: np.ndarray) -> pd.Series:
    """per-horse スコアをレース内で合計 1 に正規化する。"""
    s = pd.Series(np.clip(raw, _EPS, None), index=frame.index)
    return s / s.groupby(frame["race_id"]).transform("sum")


def _market_probs(frame: pd.DataFrame) -> pd.Series:
    """1/オッズ をレース内で正規化した市場の勝率 (控除率を割り戻した形)。"""
    inv = 1.0 / frame["odds_win"].clip(lower=1.0)
    return _race_normalised(frame, inv.to_numpy())


def _scores(frame: pd.DataFrame, probs: pd.Series, label: str, market: pd.Series) -> dict:
    """field NLL / top NLL / corr と、レースごとの市場との差を返す。

    top NLL は **同じ馬 (モデルの本命) について model と市場の両方を測る**。
    各モデルが自分の本命を採点すると別々の馬を比べることになり、意味を持たない。
    backtest の log_loss / market_log_loss も同じ組み方をしている。
    """
    won = (frame["finish_position"] == 1).to_numpy()
    p = probs.to_numpy()
    m = market.to_numpy()

    field, field_mkt, top, top_mkt, corr_p, corr_y, keys = [], [], [], [], [], [], []
    for race_id, idx in frame.groupby("race_id", sort=False).indices.items():
        w = won[idx]
        if w.sum() != 1:  # 同着・着順欠損のレースは確率評価から外す
            continue
        pi, mi = p[idx], m[idx]
        field.append(-math.log(max(float(pi[w][0]), _EPS)))
        field_mkt.append(-math.log(max(float(mi[w][0]), _EPS)))
        j = int(np.argmax(pi))  # モデルの本命。市場も**この馬**で測る
        q = min(max(float(pi[j]), _EPS), 1 - _EPS)
        qm = min(max(float(mi[j]), _EPS), 1 - _EPS)
        top.append(-math.log(q if w[j] else 1 - q))
        top_mkt.append(-math.log(qm if w[j] else 1 - qm))
        corr_p.append(q)
        corr_y.append(1.0 if w[j] else 0.0)
        keys.append(race_id)

    corr = float(np.corrcoef(corr_p, corr_y)[0, 1]) if len(set(corr_y)) > 1 else float("nan")
    return {
        "label": label,
        "races": len(field),
        "field_nll": float(np.mean(field)),
        "field_nll_market": float(np.mean(field_mkt)),
        "top_nll": float(np.mean(top)),
        "top_nll_market": float(np.mean(top_mkt)),
        "corr": corr,
        "per_race": pd.Series(np.array(field) - np.array(field_mkt), index=keys),
    }


def _slices(frame: pd.DataFrame, diff: pd.Series) -> None:
    """市場に勝てる部分集合があるか。diff = model NLL - market NLL (負なら勝ち)。"""
    fav = frame.loc[frame.groupby("race_id")["odds_win"].idxmin()].set_index("race_id")
    fav = fav.loc[fav.index.intersection(diff.index)]
    d = diff.loc[fav.index]

    print("\n市場との差 (負 = モデルが上) を切り口ごとに:")
    bands = pd.cut(fav["odds_win"], [0, 2, 3, 5, 10, 1e9],
                   labels=["1番人気 〜2.0", "〜3.0", "〜5.0", "〜10.0", "10.0〜"])
    for name, g in d.groupby(bands, observed=True):
        print(f"  1番人気のオッズ {str(name):14s} {len(g):>6,} レース  {g.mean():+.4f}")
    sizes = pd.cut(fav["n_runners"], [0, 10, 14, 100], labels=["〜10 頭", "11〜14 頭", "15 頭〜"])
    for name, g in d.groupby(sizes, observed=True):
        print(f"  頭数 {str(name):22s} {len(g):>6,} レース  {g.mean():+.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-end", default="2024-04-30")
    ap.add_argument("--valid-end", default="2024-10-31")
    ap.add_argument("--frame-start", default="2015-01-01")
    ap.add_argument("--rounds", type=int, default=2000)
    args = ap.parse_args()

    configure_logging()
    import lightgbm as lgb

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        frame = build_training_frame(session)
    engine.dispose()

    frame = frame[frame["date"] >= args.frame_start]
    frame = frame[frame["finish_position"].notna() & frame["odds_win"].notna()]
    frame = frame.reset_index(drop=True)
    train, valid, test = _split(frame, args.train_end, args.valid_end)
    log.info(
        "rows train=%d valid=%d test=%d (test %s..%s)",
        len(train), len(valid), len(test),
        test["date"].min(), test["date"].max(),
    )

    odds_set = set(ODDS_FEATURE_COLUMNS)
    variants = {
        "gbdt (odds あり)": [c for c in FEATURE_COLUMNS if c in frame.columns],
        "gbdt (odds なし)": [c for c in FEATURE_COLUMNS if c in frame.columns and c not in odds_set],
    }

    market = _market_probs(test)
    rows = []
    for label, cols in variants.items():
        y_tr = (train["finish_position"] == 1).astype(int)
        y_va = (valid["finish_position"] == 1).astype(int)
        booster = lgb.train(
            {
                "objective": "binary",      # proper scoring rule。順序だけでなく確率を合わせる
                "metric": "binary_logloss",
                "learning_rate": 0.03,
                "num_leaves": 63,
                "min_data_in_leaf": 200,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "verbosity": -1,
                "seed": 42,
            },
            lgb.Dataset(_prepare(train, cols), label=y_tr),
            num_boost_round=args.rounds,
            valid_sets=[lgb.Dataset(_prepare(valid, cols), label=y_va)],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        log.info("%s: best_iteration=%d", label, booster.best_iteration)
        raw = booster.predict(_prepare(test, cols), num_iteration=booster.best_iteration)
        rows.append(_scores(test, _race_normalised(test, np.asarray(raw)), label, market))

    print(f"\n評価: {rows[0]['races']:,} レース ({test['date'].min()}..{test['date'].max()})\n")
    print(f"{'':22s} {'field NLL':>10} {'(市場)':>9} {'top NLL':>9} {'(市場)':>9} {'corr':>7}")
    for r in rows:
        print(f"{r['label']:22s} {r['field_nll']:>10.4f} {r['field_nll_market']:>9.4f} "
              f"{r['top_nll']:>9.4f} {r['top_nll_market']:>9.4f} {r['corr']:>7.3f}")
    print(
        "\nfield NLL = レース内で正規化した勝率の -log p(勝ち馬)。小さいほど良い。"
        "\ntop NLL   = **モデルの本命 1 頭**についての二値 NLL。(市場) 列は同じ馬を"
        " 1/オッズ で採点したもの。backtest の log_loss / market_log_loss と同じ組み方で、"
        " 別々の馬を比べないようにしている。"
        "\ncorr      = 本命の予測確率と勝敗の相関 (active 0.073 / 市場 0.354)。"
    )
    for r in rows:
        print(f"\n── {r['label']}")
        _slices(test, r["per_race"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
