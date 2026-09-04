"""着順を直接学習せず、速度を予測して抽選で着順を作る。PL の代わりになるか。

`gbdt_exotic_check.py` で、勝率は市場と互角なのに PL で作った連系は負けることが
分かった。しかも**着順を 1 つ増やすたびに悪化する**（馬連 +0.023 → 馬単 +0.034 →
三連複 +0.103 → 三連単 +0.129）。PL の「2 着は勝ち馬を除いた再レース」という
独立性の仮定が、実際の着順の依存構造と合っていない。

ここでは別の作り方を試す:

    各馬の速度 μ_i と、そのばらつき σ_i を予測する
    → v_i ~ N(μ_i, σ_i) を引いて速い順に並べる = 1 回のレース
    → 何度も引いて、組合せごとの的中確率を数える

PL (Luce 型) と違い、これは Thurstone 型。**下位着順の依存構造が別物**になる。
「1 着馬を除いた残りで再レース」ではなく、全馬の速度が同時に決まる。

σ を馬ごとに predict するのが肝。安定して走る馬と、走ってみないと分からない馬を
同じばらつきで扱うと、PL とあまり変わらない形に戻ってしまう。

目的変数は **レース内の速度偏差** (distance/finish_time − そのレースの平均)。
馬場・ペース・距離といったレース全体の水準は予測できないし、着順には効かない。

σ の学習には valid の残差を使う (train の残差は過学習して小さく出るため)。

Usage:
    PYTHONPATH=src uv run python -m scripts.race_simulation_combo
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sqlite3
from itertools import permutations

import numpy as np
import pandas as pd

from core.logging import configure_logging, get_logger
from core.paths import db_path, odds_db_path
from db.session import make_engine, session_scope
from features.builder import CATEGORICAL_FEATURES, FEATURE_COLUMNS, build_training_frame

log = get_logger(__name__)
_EPS = 1e-12
_N_SIM = 20_000
#: Gumbel(0,1) の標準偏差 = π/√6。**Thurstone 型に Gumbel ノイズを入れると PL と
#: 数学的に一致する**ので、同じ μ に同じ広がりの Gaussian を入れれば「違うのは
#: 分布の形だけ」になり、着順の依存構造の効果だけを取り出せる。
_GUMBEL_SD = math.pi / math.sqrt(6.0)
# 券種 -> (使う頭数, 順序を見るか)
_BET_TYPES = {"馬連": (2, False), "馬単": (2, True), "三連複": (3, False), "三連単": (3, True)}


def _prepare(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = frame[cols].copy()
    for c in cols:
        if c in CATEGORICAL_FEATURES:
            x[c] = x[c].astype("category")
    return x


def _attach_finish_time(frame: pd.DataFrame) -> pd.DataFrame:
    """今走の finish_time を DB から引いて足す。

    build_training_frame は**今走のタイムを持たない** (予測に使えない量なので当然)。
    ここでは目的変数として要るので、entries から (race_id, horse_id) で引き直す。
    """
    con = sqlite3.connect(f"file:{db_path()}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT race_id, horse_id, finish_time FROM entries WHERE finish_time IS NOT NULL"
    ).fetchall()
    con.close()
    times = {(r, h): float(t) for r, h, t in rows if t}
    frame = frame.copy()
    frame["finish_time"] = [
        times.get((r, h), np.nan)
        for r, h in zip(frame["race_id"], frame["horse_id"], strict=True)
    ]
    return frame


def _speed_residual(frame: pd.DataFrame) -> pd.Series:
    """レース内の速度偏差 (m/s)。レース全体の水準は着順に効かないので抜く。"""
    speed = frame["distance"] / frame["finish_time"].replace(0, np.nan)
    return speed - speed.groupby(frame["race_id"]).transform("mean")


def _fit(train_x, train_y, valid_x, valid_y, seed: int = 42):
    import lightgbm as lgb

    return lgb.train(
        {
            "objective": "regression", "metric": "l2", "learning_rate": 0.03,
            "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1, "seed": seed,
        },
        lgb.Dataset(train_x, label=train_y),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(valid_x, label=valid_y)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )


def _simulate(mu: np.ndarray, sigma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """[n_sim, n] の着順 (0 = 1 着)。速い順に並べる。"""
    draws = rng.normal(mu, sigma, size=(_N_SIM, len(mu)))
    order = np.argsort(-draws, axis=1)          # 速い順の馬 index
    return order[:, :3]                          # 上位 3 頭だけあれば足りる


def _combo_probs_from_sim(top3: np.ndarray, post: np.ndarray) -> dict[str, dict[str, float]]:
    """抽選結果から券種ごとの {combo: 確率}。"""
    out: dict[str, dict[str, float]] = {b: {} for b in _BET_TYPES}
    n_sim = len(top3)
    a, b, c = top3[:, 0], top3[:, 1], top3[:, 2]
    p = post.astype(int)

    def _count(keys: np.ndarray) -> dict[str, float]:
        uniq, cnt = np.unique(keys, return_counts=True)
        return dict(zip(uniq.tolist(), (cnt / n_sim).tolist(), strict=True))

    lo, hi = np.minimum(p[a], p[b]), np.maximum(p[a], p[b])
    out["馬連"] = _count(np.char.add(np.char.add(lo.astype(str), "-"), hi.astype(str)))
    out["馬単"] = _count(np.char.add(np.char.add(p[a].astype(str), "→"), p[b].astype(str)))
    tri = np.sort(np.stack([p[a], p[b], p[c]], axis=1), axis=1)
    out["三連複"] = _count(np.char.add(np.char.add(
        np.char.add(np.char.add(tri[:, 0].astype(str), "-"), tri[:, 1].astype(str)), "-"),
        tri[:, 2].astype(str)))
    out["三連単"] = _count(np.char.add(np.char.add(
        np.char.add(np.char.add(p[a].astype(str), "→"), p[b].astype(str)), "→"),
        p[c].astype(str)))
    return out


def _top1_prob(probs: dict[str, dict[str, float]], horse: int) -> float:
    """抽選結果から 1 着確率を復元する。馬単の「その馬 → 誰か」を合計すればよい。"""
    return sum(p for c, p in probs["馬単"].items() if c.split("→")[0] == str(horse))


def _pl_combo(win: dict[int, float], horses: tuple[int, ...], ordered: bool) -> float:
    def one(order: tuple[int, ...]) -> float:
        remaining, out = 1.0, 1.0
        for h in order:
            if remaining <= _EPS:
                return _EPS
            out *= win[h] / remaining
            remaining -= win[h]
        return out

    return one(horses) if ordered else sum(one(pm) for pm in permutations(horses))


def _market(race_ids: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    con = sqlite3.connect(f"file:{odds_db_path()}?mode=ro", uri=True)
    out: dict[str, dict[str, dict[str, float]]] = {}
    want = set(_BET_TYPES)
    for i in range(0, len(race_ids), 500):
        chunk = race_ids[i : i + 500]
        q = ",".join("?" * len(chunk))
        for race_id, bet_type, blob in con.execute(
            f"SELECT race_id, bet_type, data FROM race_odds WHERE race_id IN ({q})", chunk
        ):
            if bet_type not in want:
                continue
            d = json.loads(gzip.decompress(blob))
            out.setdefault(race_id, {})[bet_type] = {
                k: float(v[0]) for k, v in d.items() if v and float(v[0]) > 0
            }
    con.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-end", default="2024-04-30")
    ap.add_argument("--valid-end", default="2024-10-31")
    ap.add_argument("--races", type=int, default=3000)
    args = ap.parse_args()

    configure_logging()
    engine = make_engine(db_path())
    with session_scope(engine) as session:
        frame = build_training_frame(session)
    engine.dispose()

    frame = frame[frame["finish_position"].notna() & frame["post_position"].notna()]
    frame = _attach_finish_time(frame)
    frame = frame[frame["finish_time"].notna() & (frame["finish_time"] > 0)]
    frame = frame[frame["odds_win"].notna() & (frame["n_runners"] >= 8)].reset_index(drop=True)
    frame["y_speed"] = _speed_residual(frame)
    frame = frame[frame["y_speed"].notna()].reset_index(drop=True)

    d = pd.to_datetime(frame["date"])
    train = frame[d <= pd.Timestamp(args.train_end)].reset_index(drop=True)
    valid = frame[(d > pd.Timestamp(args.train_end)) & (d <= pd.Timestamp(args.valid_end))]
    valid = valid.reset_index(drop=True)
    test_all = frame[d > pd.Timestamp(args.valid_end)]
    ids = list(dict.fromkeys(test_all["race_id"]))[: args.races]
    test = test_all[test_all["race_id"].isin(set(ids))].reset_index(drop=True)
    cols = [c for c in FEATURE_COLUMNS if c in frame.columns]
    log.info("train=%d / valid=%d / test=%d 行 (%d レース)",
             len(train), len(valid), len(test), len(ids))

    # μ: 速度偏差
    mu_model = _fit(_prepare(train, cols), train["y_speed"],
                    _prepare(valid, cols), valid["y_speed"])
    # σ: |残差|。**valid の残差で学習する** (train の残差は過学習して小さく出る)
    v_resid = np.abs(valid["y_speed"].to_numpy()
                     - mu_model.predict(_prepare(valid, cols),
                                        num_iteration=mu_model.best_iteration))
    cut = len(valid) // 2
    sigma_model = _fit(_prepare(valid.iloc[:cut], cols), v_resid[:cut],
                       _prepare(valid.iloc[cut:], cols), v_resid[cut:])
    log.info("μ=%d 本 / σ=%d 本 (残差の平均 %.4f m/s)",
             mu_model.best_iteration, sigma_model.best_iteration, float(v_resid.mean()))

    # 勝率 (PL 比較用)。同じ特徴量・同じ族で揃える。
    import lightgbm as lgb
    win_model = lgb.train(
        {
            "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.03,
            "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1, "seed": 42,
        },
        lgb.Dataset(_prepare(train, cols), label=(train["finish_position"] == 1).astype(int)),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(_prepare(valid, cols),
                                label=(valid["finish_position"] == 1).astype(int))],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    t_mu = mu_model.predict(_prepare(test, cols), num_iteration=mu_model.best_iteration)
    t_sigma = np.clip(sigma_model.predict(_prepare(test, cols),
                                          num_iteration=sigma_model.best_iteration),
                      1e-3, None)
    t_win_raw = np.clip(win_model.predict(_prepare(test, cols),
                                          num_iteration=win_model.best_iteration), _EPS, None)

    book = _market(ids)
    rng = np.random.default_rng(0)
    sources = ("速度sim", "勝率sim(Gauss)", "PL", "市場")
    nll: dict[tuple[str, str], list[float]] = {}
    for bt in _BET_TYPES:
        for src in sources:
            nll[(bt, src)] = []
    win_nll: dict[str, list[float]] = {s: [] for s in ("速度sim", "勝率sim(Gauss)", "PL")}

    for race_id, idx in test.groupby("race_id", sort=False).indices.items():
        g = test.iloc[idx]
        fin = g["finish_position"].to_numpy()
        top = g.iloc[np.argsort(fin)]
        if len(top) < 4 or list(top["finish_position"][:3]) != [1.0, 2.0, 3.0]:
            continue
        order = tuple(int(x) for x in top["post_position"][:3])
        post = g["post_position"].to_numpy()

        sim = _combo_probs_from_sim(_simulate(t_mu[idx], t_sigma[idx], rng), post)
        w = t_win_raw[idx]
        p_win = w / w.sum()
        win = {int(post[i]): float(p_win[i]) for i in range(len(idx))}
        # 依存構造だけを切り出す: PL と同じ μ・同じ広がりで、分布の形だけ Gaussian に
        gsim = _combo_probs_from_sim(
            _simulate(np.log(np.clip(p_win, _EPS, None)),
                      np.full(len(idx), _GUMBEL_SD), rng),
            post,
        )
        odds = book.get(race_id, {})

        # 単勝も出しておく。ここがずれていたら「依存構造の比較」になっていない
        winner = order[0]
        win_nll["速度sim"].append(-math.log(max(_top1_prob(sim, winner), 1e-7)))
        win_nll["勝率sim(Gauss)"].append(-math.log(max(_top1_prob(gsim, winner), 1e-7)))
        win_nll["PL"].append(-math.log(max(win.get(winner, _EPS), _EPS)))

        for bt, (k, ordered) in _BET_TYPES.items():
            key = ("→" if ordered else "-").join(
                str(x) for x in (order[:k] if ordered else tuple(sorted(order[:k])))
            )
            nll[(bt, "速度sim")].append(-math.log(max(sim[bt].get(key, 0.0), 1e-7)))
            nll[(bt, "勝率sim(Gauss)")].append(-math.log(max(gsim[bt].get(key, 0.0), 1e-7)))
            nll[(bt, "PL")].append(
                -math.log(max(_pl_combo(win, order[:k], ordered), _EPS)))
            m = odds.get(bt)
            if m and key in m:
                total = sum(1.0 / o for o in m.values())
                nll[(bt, "市場")].append(-math.log(max((1.0 / m[key]) / total, _EPS)))

    n_ok = len(nll[("馬連", "PL")])
    print(f"\n1 着の NLL（同じ μ を使えているかの確認）— {n_ok:,} レース\n")
    for src, vals in win_nll.items():
        print(f"  {src:16s} {float(np.mean(vals)):.4f}")

    print(f"\n当たり組合せの NLL（小さいほど良い）— {n_ok:,} レース\n")
    print(f"{'券種':8s} {'速度sim':>9} {'勝率sim':>9} {'PL':>9} {'市場':>9} {'勝率sim − PL':>13}")
    for bt in _BET_TYPES:
        v = {src: float(np.mean(nll[(bt, src)])) if nll[(bt, src)] else float("nan")
             for src in sources}
        print(f"{bt:8s} {v['速度sim']:>9.4f} {v['勝率sim(Gauss)']:>9.4f} {v['PL']:>9.4f} "
              f"{v['市場']:>9.4f} {v['勝率sim(Gauss)'] - v['PL']:>+13.4f}")
    print(
        "\n**勝率sim と PL は μ も広がりも同じで、違うのはノイズの分布の形だけ。**"
        "\nGumbel なら PL と数学的に一致するので、この列の差が「着順の依存構造」の効果。"
        "\n速度sim は μ の出どころも違う (速度回帰) ので、両方の効果が混ざっている。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
