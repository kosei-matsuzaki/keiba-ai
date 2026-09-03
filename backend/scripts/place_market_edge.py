"""複勝市場に本当に勝てているか。確率の優位が回収率になるかまで見る。

`place_probability_compare.py` で GBDT の 3 着内率が市場を NLL で 0.015 上回った。
単勝では ±0.0014 の互角だったので、複勝市場のほうが粗い可能性がある。ただし
**複勝オッズは [下限, 上限] の幅を持つ**（当たった 3 頭の組合せで払戻が変わる。
元返し保証の穴埋めが他の 2 頭の取り分から出るため）。前回は中点を暗黙確率に使った
が、それは近似で、市場側を不当に低く見せているかもしれない。

ここでは 2 段で確かめる:

1. **下限・上限の両方**で市場の NLL を出す。どちらでも負けているなら、幅の扱いで
   結論はひっくり返らない
2. **実払戻で回収率を測る**。確率が正しくても控除率 20% を超えるとは限らない。
   payouts テーブルの実額を使い、EV > 閾値 の馬だけ買ったときの回収率を出す

買うときに見えているのはオッズの**下限**（保証される最低額）なので、EV も下限で
計算する。楽観側で判定して悲観側で決済する、という都合のよい混ぜ方をしない。

Usage:
    PYTHONPATH=src uv run python -m scripts.place_market_edge
"""

from __future__ import annotations

import argparse
import gzip
import json
import sqlite3

import numpy as np
import pandas as pd

from core.logging import configure_logging, get_logger
from core.paths import db_path, odds_db_path
from db.session import make_engine, session_scope
from features.builder import CATEGORICAL_FEATURES, FEATURE_COLUMNS, build_training_frame

log = get_logger(__name__)
_EPS = 1e-9


def _prepare(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = frame[cols].copy()
    for c in cols:
        if c in CATEGORICAL_FEATURES:
            x[c] = x[c].astype("category")
    return x


def _to_three(frame: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    """レース内で合計 3 にそろえる。ちょうど 3 頭が 3 着内に入るため。"""
    s = pd.Series(np.clip(raw, _EPS, None), index=frame.index)
    return np.clip((3.0 * s / s.groupby(frame["race_id"]).transform("sum")).to_numpy(),
                   _EPS, 1 - _EPS)


def _place_odds(race_ids: list[str]) -> dict[str, dict[str, tuple[float, float]]]:
    """{race_id: {馬番: (下限, 上限)}}。上限が 0 の行は下限で埋める。"""
    con = sqlite3.connect(f"file:{odds_db_path()}?mode=ro", uri=True)
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for i in range(0, len(race_ids), 500):
        chunk = race_ids[i : i + 500]
        q = ",".join("?" * len(chunk))
        for race_id, blob in con.execute(
            f"SELECT race_id, data FROM race_odds WHERE bet_type='複勝' AND race_id IN ({q})",
            chunk,
        ):
            d = json.loads(gzip.decompress(blob))
            out[race_id] = {
                k: (float(v[0]), float(v[1]) if float(v[1]) > 0 else float(v[0]))
                for k, v in d.items()
                if v and float(v[0]) > 0
            }
    con.close()
    return out


def _place_payouts(race_ids: list[str]) -> dict[tuple[str, str], float]:
    """{(race_id, 馬番): 100 円あたりの実払戻}。当たった馬だけキーを持つ。"""
    con = sqlite3.connect(f"file:{db_path()}?mode=ro", uri=True)
    out: dict[tuple[str, str], float] = {}
    for i in range(0, len(race_ids), 500):
        chunk = race_ids[i : i + 500]
        q = ",".join("?" * len(chunk))
        for race_id, combo, amount in con.execute(
            f"SELECT race_id, combo, amount FROM payouts "
            f"WHERE bet_type='複勝' AND race_id IN ({q})",
            chunk,
        ):
            out[(race_id, "".join(str(combo).split()))] = float(amount) / 100.0
    con.close()
    return out


def _nll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-end", default="2024-04-30")
    ap.add_argument("--valid-end", default="2024-10-31")
    args = ap.parse_args()

    configure_logging()
    import lightgbm as lgb

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        frame = build_training_frame(session)
    engine.dispose()

    frame = frame[frame["finish_position"].notna() & frame["post_position"].notna()]
    frame = frame[frame["n_runners"] >= 8].reset_index(drop=True)
    d = frame["date"]
    train, valid = frame[d <= args.train_end], frame[(d > args.train_end) & (d <= args.valid_end)]
    test = frame[d > args.valid_end].reset_index(drop=True)

    cols = [c for c in FEATURE_COLUMNS if c in frame.columns]
    booster = lgb.train(
        {
            "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.03,
            "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1, "seed": 42,
        },
        lgb.Dataset(_prepare(train, cols), label=(train["finish_position"] <= 3).astype(int)),
        num_boost_round=2000,
        valid_sets=[
            lgb.Dataset(_prepare(valid, cols), label=(valid["finish_position"] <= 3).astype(int))
        ],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    p_model = _to_three(test, np.asarray(
        booster.predict(_prepare(test, cols), num_iteration=booster.best_iteration)))
    log.info("best_iteration=%d / test %d 行", booster.best_iteration, len(test))

    race_ids = list(dict.fromkeys(test["race_id"]))
    odds = _place_odds(race_ids)
    payout = _place_payouts(race_ids)
    umaban = [str(int(v)) for v in test["post_position"]]
    lo = np.array([odds.get(r, {}).get(u, (np.nan, np.nan))[0]
                   for r, u in zip(test["race_id"], umaban, strict=True)])
    hi = np.array([odds.get(r, {}).get(u, (np.nan, np.nan))[1]
                   for r, u in zip(test["race_id"], umaban, strict=True)])
    y = (test["finish_position"] <= 3).to_numpy().astype(float)
    have = np.isfinite(lo) & np.isfinite(hi)
    log.info("複勝オッズのある行: %d / %d", int(have.sum()), len(test))

    # ── 1. 幅の両端で市場の NLL ────────────────────────────────────────────
    sub = test[have]
    print(f"\n3 着内率の NLL — {have.sum():,} 頭 / {sub['race_id'].nunique():,} レース\n")
    print(f"  {'gbdt':26s} {_nll(p_model[have], y[have]):.4f}")
    for name, o in (("市場 (下限 = 保証される額)", lo), ("市場 (上限)", hi),
                    ("市場 (中点)", (lo + hi) / 2)):
        print(f"  {name:26s} {_nll(_to_three(sub, 1.0 / o[have]), y[have]):.4f}")

    # ── 2. 実払戻で回収率 ─────────────────────────────────────────────────
    # 買う時点で見えているのは下限なので、EV も下限で出す。
    ev = p_model * lo
    ret = np.array([payout.get((r, u), 0.0) for r, u in zip(test["race_id"], umaban, strict=True)])
    print("\n実払戻での複勝回収率 (EV は下限オッズで計算、控除率 20%)\n")
    print(f"  {'買う条件':22s} {'点数':>9} {'的中率':>8} {'回収率':>8} {'95% 区間':>18}")
    rng = np.random.default_rng(0)
    race_of = test["race_id"].to_numpy()
    for label, sel in [
        ("全部買う", have),
        ("本命 1 頭 (確率最大)", have & (p_model >= pd.Series(p_model, index=test.index)
                                        .groupby(test["race_id"]).transform("max").to_numpy())),
        *[(f"EV > {t:.2f}", have & (ev > t)) for t in (0.90, 1.00, 1.05, 1.10, 1.20)],
    ]:
        n = int(sel.sum())
        if n == 0:
            print(f"  {label:22s} {0:>9,}")
            continue
        roi = float(ret[sel].sum()) / n
        # **レース単位**でリサンプルする。同じレースの複数点は独立ではないので、
        # 1 点ずつ引くと区間が実際より狭く出る。
        by_race: dict[str, list[float]] = {}
        for r, v in zip(race_of[sel], ret[sel], strict=True):
            by_race.setdefault(r, []).append(v)
        groups = list(by_race.values())
        boot = []
        for _ in range(2000):
            pick = rng.integers(0, len(groups), len(groups))
            vals = [v for i in pick for v in groups[i]]
            boot.append(float(np.mean(vals)))
        lo_ci, hi_ci = np.percentile(boot, [2.5, 97.5])
        print(f"  {label:22s} {n:>9,} {float(y[sel].mean()):>8.3f} {roi:>8.3f} "
              f"{lo_ci:>8.3f}–{hi_ci:<8.3f}")
    print(
        "\n回収率 1.0 を超えて初めてエッジ。控除率 20% なので無選別なら 0.80 前後。"
        "\n**区間が 1.0 をまたいでいるうちは「勝っている」と言えない。**"
        "\nまた閾値を 5 つ試して良いものを見ているので、多重比較のぶんだけ甘い。"
        "\n決着には前進検証 (walk-forward) が要る — 同じ形の発見が 2026-08 に一度撤回されている。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
