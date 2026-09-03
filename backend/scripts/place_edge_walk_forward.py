"""複勝のバリューベットを前進検証する。単一 holdout の結果が本物かを決める。

`place_market_edge.py` は単一 holdout で EV 選別の回収率 1.03〜1.20 を出したが、
95% 区間が全部 1.0 をまたいでいた。しかも閾値を 5 つ試して良いものを見ていた。
**同じ形の発見が 2026-08 に一度撤回されている**（オッズ帯の選別。dev/holdout で
再現したのに 9 fold で消えた）ので、ここで決着させる。

検証する規則は**実際に採用するもの**そのもの:

    GBDT が 3 着内率を出す → EV = 確率 × 複勝オッズ(下限) → EV > 閾値 の馬を 1 点買う

active NN は使わない。ほとんどのレースは買わない。本命も買わない（市場が過小評価
している馬を買う規則なので）。

**多重比較を避けるため、閾値は各 fold の valid 期間で決めて test に当てる。**
test を見て選ばない。参考として固定閾値 1.00 も出す。

fold は 6 ヶ月ごと。train は test の 6 ヶ月前まで（間の 6 ヶ月が valid）。
GBDT は fold ごとに学習し直す（1 fold 約 45 秒）。

Usage:
    PYTHONPATH=src uv run python -m scripts.place_edge_walk_forward
"""

from __future__ import annotations

import argparse
import gzip
import json
import sqlite3

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from core.logging import configure_logging, get_logger
from core.paths import db_path, odds_db_path
from db.session import make_engine, session_scope
from features.builder import CATEGORICAL_FEATURES, FEATURE_COLUMNS, build_training_frame

log = get_logger(__name__)
_EPS = 1e-9
_THRESHOLDS = np.arange(0.80, 1.31, 0.05)


def _prepare(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = frame[cols].copy()
    for c in cols:
        if c in CATEGORICAL_FEATURES:
            x[c] = x[c].astype("category")
    return x


def _to_three(frame: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    s = pd.Series(np.clip(raw, _EPS, None), index=frame.index)
    return np.clip((3.0 * s / s.groupby(frame["race_id"]).transform("sum")).to_numpy(),
                   _EPS, 1 - _EPS)


def _place_odds(race_ids: list[str]) -> dict[str, dict[str, float]]:
    """{race_id: {馬番: 下限オッズ}}。買う時点で保証される額を使う。"""
    con = sqlite3.connect(f"file:{odds_db_path()}?mode=ro", uri=True)
    out: dict[str, dict[str, float]] = {}
    for i in range(0, len(race_ids), 500):
        chunk = race_ids[i : i + 500]
        q = ",".join("?" * len(chunk))
        for race_id, blob in con.execute(
            f"SELECT race_id, data FROM race_odds WHERE bet_type='複勝' AND race_id IN ({q})",
            chunk,
        ):
            d = json.loads(gzip.decompress(blob))
            out[race_id] = {k: float(v[0]) for k, v in d.items() if v and float(v[0]) > 0}
    con.close()
    return out


def _place_payouts(race_ids: list[str]) -> dict[tuple[str, str], float]:
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


def _roi(ret: np.ndarray, sel: np.ndarray) -> float:
    n = int(sel.sum())
    return float(ret[sel].sum()) / n if n else float("nan")


def _bootstrap(races: np.ndarray, ret: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    """レース単位でリサンプルした 95% 区間。同レースの複数点は独立でないため。"""
    by_race: dict[str, list[float]] = {}
    for r, v in zip(races, ret, strict=True):
        by_race.setdefault(r, []).append(v)
    groups = list(by_race.values())
    if not groups:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        boot.append(float(np.mean([v for i in pick for v in groups[i]])))
    return tuple(np.percentile(boot, [2.5, 97.5]))  # type: ignore[return-value]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first-test", default="2019-11-01")
    ap.add_argument("--folds", type=int, default=12)
    ap.add_argument("--months", type=int, default=6)
    args = ap.parse_args()

    configure_logging()
    import lightgbm as lgb

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        frame = build_training_frame(session)
    engine.dispose()

    frame = frame[frame["finish_position"].notna() & frame["post_position"].notna()]
    frame = frame[frame["n_runners"] >= 8].reset_index(drop=True)
    cols = [c for c in FEATURE_COLUMNS if c in frame.columns]

    all_ids = list(dict.fromkeys(frame["race_id"]))
    odds_book = _place_odds(all_ids)
    payout_book = _place_payouts(all_ids)
    data_end = str(frame["date"].max())

    picked_ret, picked_race, fixed_ret, fixed_race = [], [], [], []
    print(f"\n{'fold':>4} {'test 期間':23s} {'閾値':>6} {'点数':>7} {'回収率':>8} "
          f"{'固定1.00 点数':>13} {'回収率':>8}")
    for i in range(args.folds):
        test_start = pd.Timestamp(args.first_test) + relativedelta(months=args.months * i)
        test_end = test_start + relativedelta(months=args.months)
        valid_start = test_start - relativedelta(months=args.months)
        if str(test_end.date()) > data_end:
            break

        d = pd.to_datetime(frame["date"])
        train = frame[d < valid_start]
        valid = frame[(d >= valid_start) & (d < test_start)]
        test = frame[(d >= test_start) & (d < test_end)]
        if len(valid) == 0 or len(test) == 0:
            continue

        booster = lgb.train(
            {
                "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.03,
                "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.8,
                "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1, "seed": 42,
            },
            lgb.Dataset(_prepare(train, cols), label=(train["finish_position"] <= 3).astype(int)),
            num_boost_round=2000,
            valid_sets=[lgb.Dataset(_prepare(valid, cols),
                                    label=(valid["finish_position"] <= 3).astype(int))],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        def _ev_and_return(part: pd.DataFrame, bst=booster) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            # bst を既定引数で束縛する。ループ変数を閉包で掴むと fold がずれる。
            part = part.reset_index(drop=True)
            p = _to_three(part, np.asarray(
                bst.predict(_prepare(part, cols), num_iteration=bst.best_iteration)))
            uma = [str(int(v)) for v in part["post_position"]]
            o = np.array([odds_book.get(r, {}).get(u, np.nan)
                          for r, u in zip(part["race_id"], uma, strict=True)])
            ret = np.array([payout_book.get((r, u), 0.0)
                            for r, u in zip(part["race_id"], uma, strict=True)])
            ok = np.isfinite(o)
            return p * o, ret, ok & part["race_id"].notna().to_numpy()

        # 閾値は **valid** で決める。test を見て選ばない。
        ev_v, ret_v, ok_v = _ev_and_return(valid)
        scores = [(_roi(ret_v, ok_v & (ev_v > t)), t) for t in _THRESHOLDS
                  if int((ok_v & (ev_v > t)).sum()) >= 30]
        thr = max(scores)[1] if scores else 1.0

        ev_t, ret_t, ok_t = _ev_and_return(test)
        t_races = test.reset_index(drop=True)["race_id"].to_numpy()
        sel = ok_t & (ev_t > thr)
        fix = ok_t & (ev_t > 1.0)
        picked_ret += list(ret_t[sel])
        picked_race += list(t_races[sel])
        fixed_ret += list(ret_t[fix])
        fixed_race += list(t_races[fix])
        print(f"{i + 1:>4} {str(test_start.date())}..{str(test_end.date())} {thr:>6.2f} "
              f"{int(sel.sum()):>7,} {_roi(ret_t, sel):>8.3f} "
              f"{int(fix.sum()):>13,} {_roi(ret_t, fix):>8.3f}")

    print()
    for label, r, g in (("valid で選んだ閾値", picked_ret, picked_race),
                        ("固定 1.00", fixed_ret, fixed_race)):
        arr, races = np.array(r), np.array(g)
        if not len(arr):
            continue
        lo, hi = _bootstrap(races, arr)
        print(f"  {label:20s} {len(arr):>7,} 点  回収率 {arr.mean():.3f}  "
              f"95% 区間 {lo:.3f}–{hi:.3f}")
    print("\n区間が 1.0 をまたぐなら「勝っている」とは言えない。fold ごとの符号も見る"
          " — 単一 holdout の見かけの勝ちは、ここで消えることがある。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
