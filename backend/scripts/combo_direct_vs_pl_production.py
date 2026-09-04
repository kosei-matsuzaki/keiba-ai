"""本番の確率モデルの出力から、連系を直接学習する。いまの PL 経由と比べる。

`gbdt_combo_direct.py` は入力も GBDT だったので、**本番に入れたときどうなるか**は
分からなかった。ここは入力を本番と同じ確率モデル (settings.probability_model_path
の NN) にして、現行の PL 経由 (`compute_all_combination_probs`) と直接比べる。

    現行   NN のスコア → PL モンテカルロ → 券種ごとの確率
    直接   NN のスコア → 馬ごとの確率 → **組合せ単位の GBDT** → 券種ごとの確率

PL 確率も特徴量に入れる。モデルは PL からの「ずれ」だけを学べばよい。

**NN は 2024-04 までで学習済みなので、それ以降は全部 out-of-sample。** 組合せ
モデルの学習に前半、評価に後半を使う。NN の推論が 1 レース約 1.6 秒かかるので
標本を絞る。

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
from features.builder import FEATURE_COLUMNS, build_training_frame

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
    ap.add_argument("--fit-start", default="2024-10-28")
    ap.add_argument("--fit-end", default="2025-10-31")
    ap.add_argument("--fit-races", type=int, default=1500)
    ap.add_argument("--test-races", type=int, default=1200)
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
        frame = frame[frame["odds_win"].notna() & (frame["n_runners"] >= 8)].reset_index(drop=True)
        d = pd.to_datetime(frame["date"])
        fit_all = frame[(d >= pd.Timestamp(args.fit_start)) & (d <= pd.Timestamp(args.fit_end))]
        test_all = frame[d > pd.Timestamp(args.fit_end)]

        def _sample(part: pd.DataFrame, n: int) -> pd.DataFrame:
            ids = list(dict.fromkeys(part["race_id"]))
            step = max(1, len(ids) // n)
            return part[part["race_id"].isin(set(ids[::step][:n]))].reset_index(drop=True)

        bundle = load_model_full(prob_path)
        log.info("確率モデル: %s", prob_path)
        cols = [c for c in FEATURE_COLUMNS if c in frame.columns]  # noqa: F841 (frame 由来)

        def build(part: pd.DataFrame, tag: str) -> pd.DataFrame:
            rng = np.random.default_rng(0)
            out = []
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
                out.append(rows)
                if k % 200 == 0:
                    log.info("  [%s] %d/%d レース", tag, k, len(groups))
            return pd.concat(out, ignore_index=True)

        fit = build(_sample(fit_all, args.fit_races), "学習")
        test = build(_sample(test_all, args.test_races), "評価")

    engine.dispose()
    log.info("学習 %d 行 / 評価 %d 行", len(fit), len(test))

    book = _market(list(dict.fromkeys(test["race_id"])))

    print(f"\n当たり組合せの NLL（小さいほど良い）— 評価 "
          f"{test['race_id'].nunique():,} レース\n")
    print(f"{'券種':8s} {'直接学習':>10} {'現行 PL':>10} {'市場':>10} {'直接 − PL':>11}")
    for bet_type, n_hits in _BET_TYPES.items():
        tr = fit[fit["bet_type"] == bet_type]
        te = test[test["bet_type"] == bet_type].reset_index(drop=True)
        if te.empty:
            continue
        cut = tr["race_id"].drop_duplicates()
        v_ids = set(cut.iloc[int(len(cut) * 0.8):])
        booster = lgb.train(
            {
                "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
                "num_leaves": 63, "min_data_in_leaf": 500, "feature_fraction": 0.8,
                "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1, "seed": 42,
            },
            lgb.Dataset(tr[~tr["race_id"].isin(v_ids)][_FEATS],
                        label=tr[~tr["race_id"].isin(v_ids)]["hit"]),
            num_boost_round=1500,
            valid_sets=[lgb.Dataset(tr[tr["race_id"].isin(v_ids)][_FEATS],
                                    label=tr[tr["race_id"].isin(v_ids)]["hit"])],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        te = te.assign(direct=booster.predict(te[_FEATS], num_iteration=booster.best_iteration))
        p_direct = _norm(te, "direct", float(n_hits))
        p_pl = _norm(te, "pl", float(n_hits))
        mkt = np.array([book.get(r, {}).get(bet_type, {}).get(c, np.nan)
                        for r, c in zip(te["race_id"], te["combo"], strict=True)])
        ok = np.isfinite(mkt)
        p_mkt = np.full(len(te), np.nan)
        if ok.any():
            sub = te[ok].copy()
            sub["inv"] = 1.0 / mkt[ok]
            p_mkt[ok] = _norm(sub, "inv", float(n_hits))

        y = te["hit"].to_numpy().astype(bool)
        sel = y & ok

        def nll(p: np.ndarray, sel: np.ndarray = sel) -> float:
            # sel を既定引数で束縛する。ループ変数を閉包で掴むと券種がずれる。
            return float(np.mean(-np.log(np.clip(p[sel], _EPS, None))))

        print(f"{bet_type:8s} {nll(p_direct):>10.4f} {nll(p_pl):>10.4f} {nll(p_mkt):>10.4f} "
              f"{nll(p_direct) - nll(p_pl):>+11.4f}")
        gain = booster.feature_importance("gain")
        top = sorted(zip(_FEATS, gain, strict=True), key=lambda x: -x[1])[:3]
        log.info("  %s: %s", bet_type, ", ".join(f"{k}({v / gain.sum():.0%})" for k, v in top))

    print("\n直接 − PL が負なら、本番の PL 経由をやめる価値がある。"
          "\n市場より小さくなって初めて、その券種にエッジがある。"
          "\n入力は本番の確率モデル (NN) なので、この差はそのまま本番の改善幅。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
