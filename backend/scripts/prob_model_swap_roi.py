"""確率モデルを GBDT に替えたら、active NN の回収率はどうなるか。

確率モデルは馬を選ばない。だが本番では 3 つの判断を握っていて、そこを通じて
回収率に効く:

  1. 複勝を買うか       `is_place_worth_buying(確信度, place_min_hit_prob)`
  2. 複勝を何点買うか   `points_for_confidence("複勝", 確信度)` = 5×(確信度/0.50)^2 を 1〜15
  3. どの連系を買うか   確率モデル由来の combo 確率が `combo_min_hit_prob` を超えたもの全部

**買う馬を選ぶのは active NN のまま。** ここで替えるのは確信度の出所だけ。

複勝は「active が選んだ馬」に対する確率モデルの 3 着内率を確信度に使う
(`pick_confidence` と同じ)。連系は確率モデル側が候補を出す
(`merge_combination_sources` が単複だけ active 側を使う、と決めているため)。

連系の PL 列挙は確率上位 6 頭に絞る。下限を超える買い目が 7 位以下の馬で構成される
ことはまず無い。払戻は payouts の実額で、combo は normalize_combo を通す。

回収率は**賭け金で重みづける** (複勝は点数が確信度で変わるため、点数を無視すると
「厚く張った買い目が外れた」ことが数字に出ない)。

Usage:
    PYTHONPATH=src uv run python -m scripts.prob_model_swap_roi --races 1000
"""

from __future__ import annotations

import argparse
import itertools
import sqlite3
from collections import defaultdict

import numpy as np
import pandas as pd

from ai.inference.confidence import is_place_worth_buying, points_for_confidence
from ai.model.registry import get_active, load_model_full
from core.bet_types import DEFAULT_COMBO_MIN_HIT_PROB, normalize_combo
from core.logging import configure_logging, get_logger
from core.paths import db_path
from core.settings_store import SettingsStore, resolve_model_path
from db.session import make_engine, session_scope
from features.builder import CATEGORICAL_FEATURES, FEATURE_COLUMNS, build_training_frame

log = get_logger(__name__)
_EPS = 1e-12
_TOP_HORSES = 6
_COMBOS = {"馬連": (2, False), "馬単": (2, True), "三連複": (3, False), "三連単": (3, True)}


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
    remaining, out = 1.0, 1.0
    for pi in p:
        if remaining <= _EPS:
            return _EPS
        out *= pi / remaining
        remaining -= pi
    return out


def _combos_above(probs: dict[int, float], k: int, ordered: bool,
                  floor: float) -> list[tuple[str, float]]:
    """的中確率が下限を超える買い目すべて。本番と同じく点数の上限は持たない。"""
    top = sorted(probs, key=lambda h: -probs[h])[:_TOP_HORSES]
    out = []
    for c in itertools.combinations(top, k):
        if ordered:
            for perm in itertools.permutations(c):
                p = _pl([probs[h] for h in perm])
                if p > floor:
                    out.append(("→".join(map(str, perm)), p))
        else:
            p = sum(_pl([probs[h] for h in perm]) for perm in itertools.permutations(c))
            if p > floor:
                out.append(("-".join(map(str, sorted(c))), p))
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
    ap.add_argument("--races", type=int, default=1000)
    args = ap.parse_args()

    configure_logging()
    import lightgbm as lgb

    from ai.inference.predict import predict_race

    settings = SettingsStore().load()
    place_floor = float(settings.get("place_min_hit_prob", settings.get("place_min_confidence", 0.6)))
    combo_floor = dict(DEFAULT_COMBO_MIN_HIT_PROB)
    log.info("複勝の下限=%.2f / 連系の下限=%s", place_floor, combo_floor)

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

        cols = [c for c in FEATURE_COLUMNS if c in frame.columns]
        gb = {}
        for name in ("win", "place"):
            lab = (train["finish_position"] == 1) if name == "win" else (train["finish_position"] <= 3)
            vlab = (valid["finish_position"] == 1) if name == "win" else (valid["finish_position"] <= 3)
            gb[name] = lgb.train(
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
        g_win = _norm(test, np.asarray(gb["win"].predict(
            _prepare(test, cols), num_iteration=gb["win"].best_iteration)), 1.0)
        g_place = _norm(test, np.asarray(gb["place"].predict(
            _prepare(test, cols), num_iteration=gb["place"].best_iteration)), 3.0)

        active = load_model_full(get_active(session))
        prob_path = resolve_model_path(settings.get("probability_model_path"))
        prob_nn = load_model_full(prob_path) if prob_path else None
        log.info("active=%s / prob=%s", get_active(session), prob_path)

        pay = _payouts(list(picked))
        # {(出所, 券種): (賭け金, 払戻, 点数)}
        acc: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])

        idx_by_race = test.groupby("race_id", sort=False).indices
        done = 0
        for race_id, idx in idx_by_race.items():
            g = test.iloc[idx]
            uma = {h: int(u) for h, u in zip(g["horse_id"], g["post_position"], strict=True)}
            try:
                a = predict_race(active, g, session=session)
                pn = predict_race(prob_nn, g, session=session) if prob_nn else None
            except Exception:  # noqa: BLE001
                continue
            done += 1

            # active が選ぶ馬 (ここは出所によらず同じ)
            top_place_horse = a.sort_values("place_prob", ascending=False).iloc[0]["horse_id"]

            sources = {
                "現行 NN": (
                    None if pn is None else
                    dict(zip(pn["horse_id"], pn["place_prob"], strict=True)),
                    None if pn is None else
                    {uma[h]: float(v) for h, v in zip(pn["horse_id"], pn["win_prob"], strict=True)},
                ),
                "GBDT": (
                    dict(zip(g["horse_id"], g_place[idx], strict=True)),
                    {uma[h]: float(v) for h, v in zip(g["horse_id"], g_win[idx], strict=True)},
                ),
            }
            for label, (place_by_horse, win_by_uma) in sources.items():
                if place_by_horse is None:
                    continue
                # 1+2: 複勝を買うか / 何点買うか
                conf = float(place_by_horse.get(top_place_horse, float("nan")))
                conf = None if not np.isfinite(conf) else conf
                if is_place_worth_buying(conf, place_floor):
                    pts = points_for_confidence("複勝", conf)
                    ret = pay.get((race_id, "複勝", str(uma[top_place_horse])), 0.0)
                    slot = acc[(label, "複勝")]
                    slot[0] += pts
                    slot[1] += pts * ret
                    slot[2] += pts
                # 3: どの連系を買うか
                for bet_type, (k, ordered) in _COMBOS.items():
                    for combo, _p in _combos_above(win_by_uma, k, ordered,
                                                   combo_floor.get(bet_type, 0.0)):
                        ret = pay.get((race_id, bet_type, normalize_combo(combo)), 0.0)
                        slot = acc[(label, bet_type)]
                        slot[0] += 1
                        slot[1] += ret
                        slot[2] += 1
            if done % 200 == 0:
                log.info("  %d/%d レース", done, len(idx_by_race))

    engine.dispose()

    print(f"\n確率モデルを替えたときの回収率 — {done:,} レース"
          "（買う馬を選ぶのは active NN のまま）\n")
    print(f"{'券種':8s} {'出所':8s} {'点数':>9} {'点/レース':>10} {'回収率':>8}")
    for bet_type in ("複勝", *_COMBOS):
        for label in ("現行 NN", "GBDT"):
            stake, ret, pts = acc[(label, bet_type)]
            if stake == 0:
                continue
            print(f"{bet_type:8s} {label:8s} {pts:>9,.0f} {pts / done:>10.2f} {ret / stake:>8.3f}")
    print("\n複勝は点数が確信度で変わるので、回収率は賭け金で重みづけている。"
          "\n連系は 1 組合せ 1 点で、何点買うかは的中確率の下限が決める（本番と同じ）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
