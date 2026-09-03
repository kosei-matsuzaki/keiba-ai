"""回収率の差を検出するのに何レース要るか。券種ごとに実データから出す。

競馬の回収率は当たりが稀で配当が裾の重い分布なので、短い窓の数字は
「モデルの差」ではなく「その期間にたまたま高配当が当たったか」を測ってしまう。
実際 2026-09-02〜03 に 3 回、単一窓で見えた差が別の期間で消えた。

ここでは逆に、**必要な標本数を先に決める**ための数字を出す:

    ROI の標準誤差 = レース単位でブートストラップした分布の標準偏差
    n レースでの標準誤差 ≈ 全体の標準誤差 × sqrt(全体のレース数 / n)
    差 d を 95% で検出するには 標準誤差 ≦ d / (2 × 1.96)  ← 2 群の比較なので √2 込み

JRA は年 3,300 レース前後（2015〜2025 の実測）。必要レース数を年に直して出す。

賭け方は本番に合わせる: 単勝・複勝は 1 位を 1 点、連系は的中確率の下限を超えた
買い目を全部。確率は GBDT（NN を使わないので数分で回る）。**どちらのモデルを
使うかで必要標本数は大きく変わらない** — 効いているのは配当分布の裾の重さで、
これは券種の性質だから。

Usage:
    PYTHONPATH=src uv run python -m scripts.roi_measurement_power
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from itertools import combinations, permutations

import numpy as np
import pandas as pd

from ai.core.probabilities import compute_all_combination_probs
from core.bet_types import normalize_combo
from core.logging import configure_logging, get_logger
from core.paths import db_path
from db.session import make_engine, session_scope
from features.builder import CATEGORICAL_FEATURES, FEATURE_COLUMNS, build_training_frame

log = get_logger(__name__)
_EPS = 1e-12
_RACES_PER_YEAR = 3300  # 2015〜2025 の実測 (3,262〜3,395)
_MULTIPLIER = 1.25
_COMBO_TYPES = ("馬連", "馬単", "三連複", "三連単")


def _prepare(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = frame[cols].copy()
    for c in cols:
        if c in CATEGORICAL_FEATURES:
            x[c] = x[c].astype("category")
    return x


def _combo_probs(scores: np.ndarray, post: np.ndarray,
                 rng: np.random.Generator) -> dict[str, dict[str, float]]:
    n = len(scores)
    cp = compute_all_combination_probs(scores, k=3, n_samples=10_000, rng=rng)
    out: dict[str, dict[str, float]] = {b: {} for b in _COMBO_TYPES}
    for i, j in combinations(range(n), 2):
        lo, hi = sorted((int(post[i]), int(post[j])))
        out["馬連"][f"{lo}-{hi}"] = float(cp["pair"][i, j])
    for i, j in permutations(range(n), 2):
        out["馬単"][f"{int(post[i])}→{int(post[j])}"] = float(cp["ordered_pair"][i, j])
    for fs, p in cp["triple"].items():
        out["三連複"]["-".join(map(str, sorted(int(post[x]) for x in fs)))] = float(p)
    for i, j, k in permutations(range(n), 3):
        p = float(cp["ordered_triple"][i, j, k])
        if p > 0:
            out["三連単"][f"{int(post[i])}→{int(post[j])}→{int(post[k])}"] = p
    return out


def _payouts(race_ids: list[str]) -> dict[tuple[str, str, str], float]:
    con = sqlite3.connect(f"file:{db_path()}?mode=ro", uri=True)
    out: dict[tuple[str, str, str], float] = {}
    for i in range(0, len(race_ids), 500):
        chunk = race_ids[i : i + 500]
        q = ",".join("?" * len(chunk))
        for race_id, bet_type, combo, amount in con.execute(
            f"SELECT race_id, bet_type, combo, amount FROM payouts WHERE race_id IN ({q})", chunk
        ):
            out[(race_id, bet_type, normalize_combo(str(combo)))] = float(amount) / 100.0
    con.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-end", default="2024-04-30")
    ap.add_argument("--valid-end", default="2024-10-31")
    ap.add_argument("--boot", type=int, default=3000)
    args = ap.parse_args()

    configure_logging()
    import lightgbm as lgb

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        frame = build_training_frame(session)
    engine.dispose()

    frame = frame[frame["finish_position"].notna() & frame["post_position"].notna()]
    frame = frame[frame["odds_win"].notna() & (frame["n_runners"] >= 8)].reset_index(drop=True)
    d = pd.to_datetime(frame["date"])
    train = frame[d <= pd.Timestamp(args.train_end)]
    valid = frame[(d > pd.Timestamp(args.train_end)) & (d <= pd.Timestamp(args.valid_end))]
    test = frame[d > pd.Timestamp(args.valid_end)].reset_index(drop=True)

    cols = [c for c in FEATURE_COLUMNS if c in frame.columns]
    models = {}
    for name in ("win", "place"):
        lab = (train["finish_position"] == 1) if name == "win" else (train["finish_position"] <= 3)
        vlab = (valid["finish_position"] == 1) if name == "win" else (valid["finish_position"] <= 3)
        models[name] = lgb.train(
            {
                "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.03,
                "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.8,
                "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1, "seed": 42,
            },
            lgb.Dataset(_prepare(train, cols), label=lab.astype(int)),
            num_boost_round=2000,
            valid_sets=[lgb.Dataset(_prepare(valid, cols), label=vlab.astype(int))],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

    def raw(part: pd.DataFrame, name: str) -> np.ndarray:
        return np.clip(models[name].predict(
            _prepare(part, cols), num_iteration=models[name].best_iteration), _EPS, None)

    # 下限は valid から (本番の recipe)
    rng = np.random.default_rng(0)
    v_raw = raw(valid, "win")
    best: dict[str, list[float]] = defaultdict(list)
    v = valid.reset_index(drop=True)
    for _rid, idx in v.groupby("race_id", sort=False).indices.items():
        s = v_raw[idx]
        probs = _combo_probs(np.log(s / s.sum()), v["post_position"].to_numpy()[idx], rng)
        for bt, m in probs.items():
            if m:
                best[bt].append(max(m.values()))
    floors = {bt: float(np.median(x)) * _MULTIPLIER for bt, x in best.items()}
    log.info("下限: %s", {k: round(x, 4) for k, x in floors.items()})

    t_win, t_place = raw(test, "win"), raw(test, "place")
    pay = _payouts(list(dict.fromkeys(test["race_id"])))
    # {券種: {race_id: [賭け金, 払戻]}}
    per_race: dict[str, dict[str, list[float]]] = {
        b: defaultdict(lambda: [0.0, 0.0]) for b in ("単勝", "複勝", *_COMBO_TYPES)
    }
    for n, (race_id, idx) in enumerate(test.groupby("race_id", sort=False).indices.items(), start=1):
        g = test.iloc[idx]
        post = g["post_position"].to_numpy()
        w, p = t_win[idx], t_place[idx]
        for bt, arr in (("単勝", w), ("複勝", p)):
            uma = int(post[int(np.argmax(arr))])
            slot = per_race[bt][race_id]
            slot[0] += 1
            slot[1] += pay.get((race_id, bt, str(uma)), 0.0)
        for bt, m in _combo_probs(np.log(w / w.sum()), post, rng).items():
            for combo, prob in m.items():
                if prob > floors[bt]:
                    slot = per_race[bt][race_id]
                    slot[0] += 1
                    slot[1] += pay.get((race_id, bt, normalize_combo(combo)), 0.0)
        if n % 1000 == 0:
            log.info("  %d レース", n)

    n_races = test["race_id"].nunique()
    print(f"\n回収率の測定精度 — test {n_races:,} レース"
          f"（{n_races / _RACES_PER_YEAR:.1f} 年分）\n")
    print(f"{'券種':8s} {'点/R':>6} {'回収率':>7} {'この標本の':>11} "
          f"{'±0.05 に要る':>13} {'±0.03 に要る':>13}")
    print(f"{'':8s} {'':6s} {'':7s} {'標準誤差':>11} {'年数':>13} {'年数':>13}")
    for bt, book in per_race.items():
        races = list(book.values())
        if not races:
            continue
        stakes = np.array([r[0] for r in races])
        rets = np.array([r[1] for r in races])
        roi = rets.sum() / stakes.sum()
        boot = []
        for _ in range(args.boot):
            pick = rng.integers(0, len(races), len(races))
            boot.append(rets[pick].sum() / max(stakes[pick].sum(), _EPS))
        se = float(np.std(boot))
        # 2 群の比較で差 d を 95% 検出: 2*1.96*se*sqrt(2) <= d
        def years_for(dd: float, se: float = se, n: int = len(races)) -> float:
            # 既定引数で束縛する。ループ変数を閉包で掴むと券種がずれる。
            return (2 * 1.96 * np.sqrt(2) * se / dd) ** 2 * n / _RACES_PER_YEAR
        print(f"{bt:8s} {stakes.sum() / len(races):>6.2f} {roi:>7.3f} {se:>11.4f} "
              f"{years_for(0.05):>13.1f} {years_for(0.03):>13.1f}")
    print(
        "\n「±0.05 に要る年数」= 2 つのモデルの回収率が 0.05 違うことを 95% で言うのに"
        "\n必要な期間。1 年 = 3,300 レース（2015〜2025 の実測）。"
        "\n**この年数より短い窓で出した差は、読んではいけない。**"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
