"""連系の市場は単勝の市場より粗いか。GBDT の勝率 → Plackett-Luce と突き合わせる。

単勝では GBDT は市場と互角にしかならなかった (field NLL 1.9192 対 1.9206)。
だが群衆が下手なのは組合せが多く流動性の低い市場のはずで、そこなら勝てるかもしれない
—— というのがこの検証。控除率は連系のほうが高い (単勝 20% / 三連単 27.5%) ので、
**勝てる幅がその差を超えないと意味がない**。

測り方は単勝のときと同じで、**回収率ではなく確率**を見る:

    NLL = -log P(実際に当たった組合せ)

モデル側は勝率から Plackett-Luce で解析的に出す。PL は全組合せで合計 1 になる
真の分布なので正規化は要らない。市場側は 1/オッズ を全組合せで正規化する
(控除率を割り戻す)。当たり組合せは payouts ではなく finish_position から作る
—— payouts の combo は表記ゆれがあるうえ、ここでは 1〜3 着の馬番だけで足りる。

ワイドと枠連は外す。ワイドは当たりが 3 通りあって単一の分布にならず、枠連は
馬番ではなく枠番なので同じ土俵に乗らない。

Usage:
    PYTHONPATH=src uv run python -m scripts.gbdt_exotic_check
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sqlite3
from collections import defaultdict

import numpy as np
import pandas as pd

from core.logging import configure_logging, get_logger
from core.paths import db_path, odds_db_path
from db.session import make_engine, session_scope
from features.builder import CATEGORICAL_FEATURES, FEATURE_COLUMNS, build_training_frame

log = get_logger(__name__)

_EPS = 1e-12
# 券種 -> (何頭使うか, 順序を見るか)
_BET_TYPES = {
    "馬連": (2, False),
    "馬単": (2, True),
    "三連複": (3, False),
    "三連単": (3, True),
}


def _pl_ordered(p: list[float]) -> float:
    """Plackett-Luce: この順で 1,2,3 着になる確率。p は勝率 (レース内で合計 1)。"""
    remaining = 1.0
    out = 1.0
    for pi in p:
        if remaining <= _EPS:
            return _EPS
        out *= pi / remaining
        remaining -= pi
    return max(out, _EPS)


def _pl_combo(probs: dict[int, float], horses: tuple[int, ...], ordered: bool) -> float:
    """当たり組合せの確率。順不同なら全順列を足す。"""
    import itertools

    if ordered:
        return _pl_ordered([probs[h] for h in horses])
    return sum(_pl_ordered([probs[h] for h in perm]) for perm in itertools.permutations(horses))


def _combo_key(horses: tuple[int, ...], ordered: bool) -> str:
    """odds.db の combo 表記に合わせる。順不同は昇順で '-'、順序ありは '→'。"""
    return ("→" if ordered else "-").join(
        str(h) for h in (horses if ordered else tuple(sorted(horses)))
    )


def _load_odds(race_ids: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    """{race_id: {bet_type: {combo: odds}}}。無い race_id はキーを持たない。"""
    con = sqlite3.connect(f"file:{odds_db_path()}?mode=ro", uri=True)
    out: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    want = set(_BET_TYPES)
    for i in range(0, len(race_ids), 500):
        chunk = race_ids[i : i + 500]
        q = ",".join("?" * len(chunk))
        rows = con.execute(
            f"SELECT race_id, bet_type, data FROM race_odds WHERE race_id IN ({q})", chunk
        ).fetchall()
        for race_id, bet_type, blob in rows:
            if bet_type not in want:
                continue
            d = json.loads(gzip.decompress(blob))
            out[race_id][bet_type] = {
                k: float(v[0]) for k, v in d.items() if v and float(v[0]) > 0
            }
    con.close()
    return out


def _prepare(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = frame[cols].copy()
    for c in cols:
        if c in CATEGORICAL_FEATURES:
            x[c] = x[c].astype("category")
    return x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-end", default="2024-04-30")
    ap.add_argument("--valid-end", default="2024-10-31")
    ap.add_argument("--rounds", type=int, default=2000)
    args = ap.parse_args()

    configure_logging()
    import lightgbm as lgb

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        frame = build_training_frame(session)
    engine.dispose()

    frame = frame[frame["finish_position"].notna() & frame["odds_win"].notna()]
    frame = frame[frame["post_position"].notna()].reset_index(drop=True)
    d = frame["date"]
    train = frame[d <= args.train_end]
    valid = frame[(d > args.train_end) & (d <= args.valid_end)]
    test = frame[d > args.valid_end].reset_index(drop=True)

    cols = [c for c in FEATURE_COLUMNS if c in frame.columns]
    booster = lgb.train(
        {
            "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.03,
            "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1, "seed": 42,
        },
        lgb.Dataset(_prepare(train, cols), label=(train["finish_position"] == 1).astype(int)),
        num_boost_round=args.rounds,
        valid_sets=[
            lgb.Dataset(_prepare(valid, cols), label=(valid["finish_position"] == 1).astype(int))
        ],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    raw = np.clip(booster.predict(_prepare(test, cols), num_iteration=booster.best_iteration), _EPS, None)
    s = pd.Series(raw, index=test.index)
    test = test.assign(p_win=(s / s.groupby(test["race_id"]).transform("sum")).to_numpy())
    log.info("best_iteration=%d / test %d 行", booster.best_iteration, len(test))

    race_ids = list(dict.fromkeys(test["race_id"]))
    odds_by_race = _load_odds(race_ids)
    log.info("オッズのあるレース: %d / %d", len(odds_by_race), len(race_ids))

    stats: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for race_id, g in test.groupby("race_id", sort=False):
        book = odds_by_race.get(race_id)
        if not book:
            continue
        top = g.nsmallest(3, "finish_position")
        if len(top) < 3 or top["finish_position"].tolist() != [1.0, 2.0, 3.0]:
            continue  # 同着・着順欠損は外す
        order = tuple(int(x) for x in top["post_position"])
        probs = {int(r.post_position): float(r.p_win) for r in g.itertuples()}
        if len(probs) < 3:
            continue

        for bet_type, (k, ordered) in _BET_TYPES.items():
            combos = book.get(bet_type)
            if not combos:
                continue
            key = _combo_key(order[:k], ordered)
            if key not in combos:
                continue
            total = sum(1.0 / o for o in combos.values())
            p_market = (1.0 / combos[key]) / total
            p_model = _pl_combo(probs, order[:k], ordered)
            stats[bet_type].append((-math.log(max(p_model, _EPS)), -math.log(max(p_market, _EPS))))

    print(f"\n当たり組合せの NLL (小さいほど良い) — test {len(race_ids):,} レース中\n")
    print(f"{'券種':8s} {'レース':>7} {'GBDT+PL':>9} {'市場':>9} {'差':>8} {'控除率':>7}")
    takeout = {"馬連": "22.5%", "馬単": "25.0%", "三連複": "25.0%", "三連単": "27.5%"}
    for bet_type in _BET_TYPES:
        v = stats.get(bet_type)
        if not v:
            print(f"{bet_type:8s} {'—':>7}")
            continue
        model = float(np.mean([a for a, _ in v]))
        market = float(np.mean([b for _, b in v]))
        print(f"{bet_type:8s} {len(v):>7,} {model:>9.4f} {market:>9.4f} "
              f"{model - market:>+8.4f} {takeout[bet_type]:>7}")
    print(
        "\n差が負なら GBDT の確率が市場より正しい。**負であることに加えて、"
        "\n単勝で測った差 (-0.0014) より大きくないと「連系のほうが粗い」とは言えない。**"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
