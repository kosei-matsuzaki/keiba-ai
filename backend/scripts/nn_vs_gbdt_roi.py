"""買い手として NN と GBDT のどちらが儲かるか。券種ごとに実払戻で比べる。

今日の一連の検証は**確率の質**しか見ていない。「買う馬を決めるモデル」としての
優劣は別の問いなので、ここで直接測る。

公平にするため **両モデルに同じ買い方**を当てる:

    単勝  勝率 1 位を 1 点
    複勝  3 着内率 1 位を 1 点
    連系  確率上位 2 点 (点数を揃えないと ROI が買い方の差で動く)

連系の確率は**両方とも PL** で作る。GBDT は 3 着内率ベースのほうが良いと分かって
いるが、それを使うとモデルの差と変換の差が混ざる。ここで知りたいのはモデルの差。

PL の全列挙は三連単で 1 レース 2,184 通りになるので、**確率上位 6 頭に絞って**
列挙する。上位 2 点が 7 位以下の馬を含むことはまず無い (PL は勝率の積なので)。

払戻は payouts の実額。**combo の表記ゆれに注意** — payouts 側は "1 - 10" のように
空白が入るので core.bet_types.normalize_combo を必ず通す。ここを素の == で比べると
連系が 1 件も当たらない (2026-09-01 に本番で踏んだバグ)。

NN は 1 レース約 1.6 秒かかる。--races で標本数を決める。

Usage:
    PYTHONPATH=src uv run python -m scripts.nn_vs_gbdt_roi --races 1500
"""

from __future__ import annotations

import argparse
import itertools
import sqlite3
from collections import defaultdict

import numpy as np
import pandas as pd

from ai.model.registry import get_active, load_model_full
from core.bet_types import normalize_combo
from core.logging import configure_logging, get_logger
from core.paths import db_path
from db.session import make_engine, session_scope
from features.builder import CATEGORICAL_FEATURES, FEATURE_COLUMNS, build_training_frame

log = get_logger(__name__)
_EPS = 1e-12
_TOP_HORSES = 6  # 連系の列挙に使う上位頭数
_POINTS = 2      # 連系で買う点数 (両モデル共通)

# 券種 -> (使う頭数, 順序を見るか)
_COMBOS = {
    "馬連": (2, False), "馬単": (2, True),
    "三連複": (3, False), "三連単": (3, True),
}


def _prepare(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = frame[cols].copy()
    for c in cols:
        if c in CATEGORICAL_FEATURES:
            x[c] = x[c].astype("category")
    return x


def _norm(frame: pd.DataFrame, raw: np.ndarray, total: float) -> np.ndarray:
    s = pd.Series(np.clip(raw, _EPS, None), index=frame.index)
    return np.clip((total * s / s.groupby(frame["race_id"]).transform("sum")).to_numpy(),
                   _EPS, 1 - _EPS)


def _pl(p: list[float]) -> float:
    """Plackett-Luce: この順に入る確率。"""
    remaining, out = 1.0, 1.0
    for pi in p:
        if remaining <= _EPS:
            return _EPS
        out *= pi / remaining
        remaining -= pi
    return out


def _top_combos(probs: dict[int, float], k: int, ordered: bool, n: int) -> list[tuple[str, float]]:
    """確率上位 n 点の (combo 文字列, 確率)。上位 _TOP_HORSES 頭から列挙する。"""
    top = sorted(probs, key=lambda h: -probs[h])[:_TOP_HORSES]
    scored = []
    for c in itertools.combinations(top, k):
        if ordered:
            for perm in itertools.permutations(c):
                scored.append(("→".join(map(str, perm)), _pl([probs[h] for h in perm])))
        else:
            scored.append((
                "-".join(map(str, sorted(c))),
                sum(_pl([probs[h] for h in perm]) for perm in itertools.permutations(c)),
            ))
    scored.sort(key=lambda x: -x[1])
    return scored[:n]


def _payouts(race_ids: list[str]) -> dict[tuple[str, str, str], float]:
    """{(race_id, 券種, 正規化 combo): 100 円あたりの払戻}。"""
    con = sqlite3.connect(f"file:{db_path()}?mode=ro", uri=True)
    out: dict[tuple[str, str, str], float] = {}
    for i in range(0, len(race_ids), 500):
        chunk = race_ids[i : i + 500]
        q = ",".join("?" * len(chunk))
        for race_id, bet_type, combo, amount in con.execute(
            f"SELECT race_id, bet_type, combo, amount FROM payouts WHERE race_id IN ({q})",
            chunk,
        ):
            out[(race_id, bet_type, normalize_combo(str(combo)))] = float(amount) / 100.0
    con.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-end", default="2024-04-30")
    ap.add_argument("--valid-end", default="2024-10-31")
    ap.add_argument("--races", type=int, default=1500)
    args = ap.parse_args()

    configure_logging()
    import lightgbm as lgb

    from ai.inference.predict import predict_race

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        frame = build_training_frame(session)
        frame = frame[frame["finish_position"].notna() & frame["post_position"].notna()]
        frame = frame[frame["odds_win"].notna() & (frame["n_runners"] >= 8)].reset_index(drop=True)
        d = frame["date"]
        train = frame[d <= args.train_end]
        valid = frame[(d > args.train_end) & (d <= args.valid_end)]
        test_all = frame[d > args.valid_end]

        ids = list(dict.fromkeys(test_all["race_id"]))
        step = max(1, len(ids) // args.races)
        picked = set(ids[::step][: args.races])
        test = test_all[test_all["race_id"].isin(picked)].reset_index(drop=True)
        log.info("test %d レース / %d 行", len(picked), len(test))

        cols = [c for c in FEATURE_COLUMNS if c in frame.columns]
        boosters = {}
        for name, lab in (("win", train["finish_position"] == 1),
                          ("place", train["finish_position"] <= 3)):
            vlab = (valid["finish_position"] == 1) if name == "win" \
                else (valid["finish_position"] <= 3)
            boosters[name] = lgb.train(
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
        g_win = _norm(test, np.asarray(boosters["win"].predict(
            _prepare(test, cols), num_iteration=boosters["win"].best_iteration)), 1.0)
        g_place = _norm(test, np.asarray(boosters["place"].predict(
            _prepare(test, cols), num_iteration=boosters["place"].best_iteration)), 3.0)
        log.info("gbdt win=%d / place=%d 本",
                 boosters["win"].best_iteration, boosters["place"].best_iteration)

        path = get_active(session)
        assert path is not None, "active モデルが無い"
        bundle = load_model_full(path)
        log.info("active: %s", path)

        pay = _payouts(list(picked))
        # {(モデル, 券種): [払戻, ...]}
        rets: dict[tuple[str, str], list[float]] = defaultdict(list)

        idx_by_race = test.groupby("race_id", sort=False).indices
        for n, (race_id, idx) in enumerate(idx_by_race.items(), start=1):
            g = test.iloc[idx]
            uma = {h: int(u) for h, u in zip(g["horse_id"], g["post_position"], strict=True)}
            sources: dict[str, tuple[dict[int, float], dict[int, float]]] = {
                "gbdt": ({uma[h]: float(v) for h, v in zip(g["horse_id"], g_win[idx], strict=True)},
                         {uma[h]: float(v) for h, v in zip(g["horse_id"], g_place[idx], strict=True)}),
            }
            try:
                preds = predict_race(bundle, g, session=session)
                sources["nn"] = (
                    {uma[h]: float(v) for h, v in zip(preds["horse_id"], preds["win_prob"], strict=True)},
                    {uma[h]: float(v) for h, v in zip(preds["horse_id"], preds["place_prob"], strict=True)},
                )
            except Exception:  # noqa: BLE001 — 1 レースの失敗で全体を止めない
                continue

            for model, (pw, pp) in sources.items():
                rets[(model, "単勝")].append(
                    pay.get((race_id, "単勝", str(max(pw, key=pw.get))), 0.0))
                rets[(model, "複勝")].append(
                    pay.get((race_id, "複勝", str(max(pp, key=pp.get))), 0.0))
                for bet_type, (k, ordered) in _COMBOS.items():
                    for combo, _p in _top_combos(pw, k, ordered, _POINTS):
                        rets[(model, bet_type)].append(
                            pay.get((race_id, bet_type, normalize_combo(combo)), 0.0))
            if n % 200 == 0:
                log.info("  %d/%d レース", n, len(idx_by_race))

    engine.dispose()

    print(f"\n同じ買い方での回収率 — {len(idx_by_race):,} レース"
          f"（単複は 1 位を 1 点 / 連系は確率上位 {_POINTS} 点）\n")
    print(f"{'券種':8s} {'点数':>8} {'NN':>8} {'GBDT':>8} {'差':>8}")
    for bet_type in ("単勝", "複勝", *_COMBOS):
        nn, gb = rets[("nn", bet_type)], rets[("gbdt", bet_type)]
        if not nn or not gb:
            continue
        r_nn, r_gb = float(np.mean(nn)), float(np.mean(gb))
        print(f"{bet_type:8s} {len(nn):>8,} {r_nn:>8.3f} {r_gb:>8.3f} {r_gb - r_nn:>+8.3f}")
    print("\n差が正なら GBDT のほうが買い手として良い。"
          "\n連系の確率は両方とも PL で作っている（モデルの差だけを見るため）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
