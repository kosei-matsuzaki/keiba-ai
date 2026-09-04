"""本番の確率モデルの出力から、連系を直接学習する。いまの PL 経由と比べる。

`gbdt_combo_direct.py` は入力も GBDT だったので、**本番に入れたときどうなるか**は
分からなかった。ここは入力を本番と同じ確率モデル (settings.probability_model_path
の NN) にして、現行の PL 経由 (`compute_all_combination_probs`) と直接比べる。

    現行   NN のスコア → PL モンテカルロ → 券種ごとの確率
    直接   NN のスコア → 馬ごとの確率 → **組合せ単位の GBDT** → 券種ごとの確率

PL 確率も特徴量に入れる。モデルは PL からの「ずれ」だけを学べばよい。

**組合せモデルが使える期間は NN の未見期間だけ。** NN が学習に使った期間の出力は
過学習した値なので、それに対する補正を学んでも転移しない。よって 2024-10 以降の
22 ヶ月しか使えず、**確率モデル 31,000 レース対 組合せモデル 6,000 レース**という
開きは構造上避けられない。

そのうえで **拡大窓の交差適合**にする。22 ヶ月を時間順に K 分割し、fold k は
1..k-1 で学習して k を予測する。全期間を評価に使えるうえ、**学習量を増やしながら
測れる**ので「データ不足なのか、そもそも効かないのか」が分かる。

三連単は 1 レース 2,184 行になるので外す (三連複 364 行までにする)。

Usage:
    PYTHONPATH=src uv run python -m scripts.combo_direct_vs_pl_production
"""

from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
from itertools import combinations, permutations

import numpy as np
import pandas as pd

from ai.core.probabilities import compute_all_combination_probs
from ai.inference.predict import derive_wide_prob_from_triple
from ai.model.registry import load_model_full
from core.logging import configure_logging, get_logger
from core.paths import db_path, odds_db_path
from core.settings_store import SettingsStore, resolve_model_path
from db.session import make_engine, session_scope
from features.builder import build_training_frame

log = get_logger(__name__)
_EPS = 1e-12
_N_SAMPLES = 10_000
#: 券種 -> (当たりが何通りか)。ワイドだけ 1 レースで 3 通り当たる。
_BET_TYPES = {"馬連": 1, "馬単": 1, "ワイド": 3, "三連複": 1}
_FEATS = [
    "pl", "n_runners", "p_win_a", "p_win_b", "p_place_a", "p_place_b",
    "p_win_prod", "p_place_prod", "p_win_min", "p_win_max",
    "odds_a", "odds_b", "p_win_c", "p_place_c",
]


def _race_rows(g: pd.DataFrame, scores: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """1 レースを組合せ 1 行に展開する。本番と同じモンテカルロから PL 確率も取る。"""
    n = len(g)
    cp = compute_all_combination_probs(scores, k=3, n_samples=_N_SAMPLES, rng=rng)
    p_place = cp["place"]
    p_win = cp["position"][:, 0]
    wide = derive_wide_prob_from_triple(cp["triple"], n)
    post = [int(v) for v in g["post_position"]]
    fin = g["finish_position"].to_numpy()
    odds = g["odds_win"].to_numpy()

    rows = []

    def _add(bet_type, key, pl, idxs, hit):
        a, b = idxs[0], idxs[1]
        c = idxs[2] if len(idxs) > 2 else idxs[1]
        rows.append({
            "bet_type": bet_type, "combo": key, "hit": hit, "pl": pl,
            "n_runners": n,
            "p_win_a": p_win[a], "p_win_b": p_win[b], "p_win_c": p_win[c],
            "p_place_a": p_place[a], "p_place_b": p_place[b], "p_place_c": p_place[c],
            "p_win_prod": p_win[a] * p_win[b], "p_place_prod": p_place[a] * p_place[b],
            "p_win_min": min(p_win[a], p_win[b]), "p_win_max": max(p_win[a], p_win[b]),
            "odds_a": odds[a], "odds_b": odds[b],
        })

    for i, j in combinations(range(n), 2):
        lo, hi = sorted((post[i], post[j]))
        _add("馬連", f"{lo}-{hi}", float(cp["pair"][i, j]), (i, j),
             1 if {fin[i], fin[j]} == {1.0, 2.0} else 0)
        _add("ワイド", f"{lo}-{hi}", float(wide[i, j]), (i, j),
             1 if (fin[i] <= 3 and fin[j] <= 3) else 0)
    for i, j in permutations(range(n), 2):
        _add("馬単", f"{post[i]}→{post[j]}", float(cp["ordered_pair"][i, j]), (i, j),
             1 if (fin[i] == 1 and fin[j] == 2) else 0)
    for fs, p in cp["triple"].items():
        idxs = tuple(fs)
        key = "-".join(map(str, sorted(post[x] for x in idxs)))
        _add("三連複", key, float(p), idxs,
             1 if all(fin[x] <= 3 for x in idxs) else 0)
    return pd.DataFrame(rows)


def _market(race_ids: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    con = sqlite3.connect(f"file:{odds_db_path()}?mode=ro", uri=True)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for i in range(0, len(race_ids), 500):
        chunk = race_ids[i : i + 500]
        q = ",".join("?" * len(chunk))
        for race_id, bet_type, blob in con.execute(
            f"SELECT race_id, bet_type, data FROM race_odds WHERE race_id IN ({q})", chunk
        ):
            if bet_type not in _BET_TYPES:
                continue
            d = json.loads(gzip.decompress(blob))
            out.setdefault(race_id, {})[bet_type] = {
                k: float(v[0]) for k, v in d.items() if v and float(v[0]) > 0
            }
    con.close()
    return out


def _norm(df: pd.DataFrame, col: str, total: float) -> np.ndarray:
    s = pd.Series(np.clip(df[col].to_numpy(), _EPS, None), index=df.index)
    return np.clip((total * s / s.groupby(df["race_id"]).transform("sum")).to_numpy(),
                   _EPS, 1 - _EPS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2024-10-28", help="NN の未見期間の始まり")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--races", type=int, default=6000)
    args = ap.parse_args()

    configure_logging()
    import lightgbm as lgb

    from ai.inference.predict import _predict_race_nn

    prob_path = resolve_model_path(SettingsStore().load().get("probability_model_path"))
    assert prob_path is not None, "probability_model_path が未設定"

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        frame = build_training_frame(session)
        frame = frame[frame["finish_position"].notna() & frame["post_position"].notna()]
        frame = frame[frame["odds_win"].notna() & (frame["n_runners"] >= 8)]
        d = pd.to_datetime(frame["date"])
        part = frame[d >= pd.Timestamp(args.start)].reset_index(drop=True)
        ids = list(dict.fromkeys(part["race_id"]))[: args.races]
        part = part[part["race_id"].isin(set(ids))].reset_index(drop=True)

        bundle = load_model_full(prob_path)
        log.info("確率モデル: %s / 対象 %d レース", prob_path, len(ids))

        rng = np.random.default_rng(0)
        chunks = []
        groups = part.groupby("race_id", sort=False).indices
        for k, (race_id, idx) in enumerate(groups.items(), start=1):
            g = part.iloc[idx]
            try:
                preds = _predict_race_nn(bundle, g, session=session)
            except Exception:  # noqa: BLE001
                continue
            by_horse = dict(zip(preds["horse_id"], preds["score"], strict=True))
            scores = np.array([by_horse[h] for h in g["horse_id"]])
            rows = _race_rows(g.reset_index(drop=True), scores, rng)
            rows["race_id"] = race_id
            chunks.append(rows)
            if k % 500 == 0:
                log.info("  %d/%d レース", k, len(groups))
        data = pd.concat(chunks, ignore_index=True)

    engine.dispose()

    # 時間順に K 分割。**fold k は 1..k-1 で学習して k を予測する** (拡大窓)。
    order = list(dict.fromkeys(data["race_id"]))
    bounds = np.linspace(0, len(order), args.folds + 1).astype(int)
    fold_of = {}
    for f in range(args.folds):
        for r in order[bounds[f] : bounds[f + 1]]:
            fold_of[r] = f
    data["fold"] = data["race_id"].map(fold_of)

    book = _market(order)
    mkt = np.array([book.get(r, {}).get(bt, {}).get(c, np.nan)
                    for r, bt, c in zip(data["race_id"], data["bet_type"], data["combo"],
                                        strict=True)])
    data["mkt_odds"] = mkt

    print(f"\n当たり組合せの NLL（小さいほど良い）— 拡大窓 {args.folds} 分割 / "
          f"{len(order):,} レース\n")
    print(f"{'券種':8s} {'学習レース':>9} {'直接学習':>10} {'現行 PL':>10} {'市場':>10} "
          f"{'直接 − PL':>11}")
    for bet_type, n_hits in _BET_TYPES.items():
        sub = data[data["bet_type"] == bet_type]
        for f in range(1, args.folds):
            tr = sub[sub["fold"] < f]
            te = sub[sub["fold"] == f].reset_index(drop=True)
            if te.empty or tr.empty:
                continue
            # early stopping 用に、学習期間の **後ろ 20% のレース** を分ける。
            #
            # 以前は「学習の最後の 1 fold」を使っていたが、最初に評価する fold では
            # 学習が 1 fold しかなく、**検証セットが学習データそのもの**になっていた。
            # in-sample で早期停止を判定するので過学習し放題で、その行だけ NLL が
            # +0.5〜+1.5 と壊れていた。レース単位で切れば fold 数に依らず分離できる。
            tr_races = list(dict.fromkeys(tr["race_id"]))
            v_races = set(tr_races[int(len(tr_races) * 0.8):])
            v = tr["race_id"].isin(v_races)
            if not v.any() or not (~v).any():
                continue  # 分けられないほど小さい学習期間は測らない
            booster = lgb.train(
                {
                    "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
                    "num_leaves": 63, "min_data_in_leaf": 500, "feature_fraction": 0.8,
                    "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1, "seed": 42,
                },
                lgb.Dataset(tr[~v][_FEATS], label=tr[~v]["hit"]),
                num_boost_round=1500,
                valid_sets=[lgb.Dataset(tr[v][_FEATS], label=tr[v]["hit"])],
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            te = te.assign(
                direct=booster.predict(te[_FEATS], num_iteration=booster.best_iteration)
            )
            p_direct = _norm(te, "direct", float(n_hits))
            p_pl = _norm(te, "pl", float(n_hits))
            ok = np.isfinite(te["mkt_odds"].to_numpy())
            p_mkt = np.full(len(te), np.nan)
            if ok.any():
                s2 = te[ok].copy()
                s2["inv"] = 1.0 / s2["mkt_odds"]
                p_mkt[ok] = _norm(s2, "inv", float(n_hits))
            sel = te["hit"].to_numpy().astype(bool) & ok

            def nll(p: np.ndarray, sel: np.ndarray = sel) -> float:
                return float(np.mean(-np.log(np.clip(p[sel], _EPS, None))))

            label = bet_type if f == 1 else ""
            print(f"{label:8s} {tr['race_id'].nunique():>9,} {nll(p_direct):>10.4f} "
                  f"{nll(p_pl):>10.4f} {nll(p_mkt):>10.4f} "
                  f"{nll(p_direct) - nll(p_pl):>+11.4f}")

    print("\n学習レースを増やしても差が縮まらないなら、データ不足ではない。"
          "\n縮み続けているなら、22 ヶ月という上限が効いている。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
