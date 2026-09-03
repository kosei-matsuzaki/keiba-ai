"""連系の下限を各モデルの確率分布から出し直して、公平に比べる。

`prob_model_swap_roi.py` は GBDT の連系が大きく負けたが、**その比較は不公平**
だった。`DEFAULT_COMBO_MIN_HIT_PROB`（馬連 0.075 / 三連単 0.019 …）は現行 NN の
確率分布から作った値なので、スケールの違う GBDT に流用すると線を超える買い目が
増えすぎる（実際 2.28 → 2.79 点/レースに増えた）。

導出の recipe は `core/bet_types.py` の記述どおり:

    券種ごとに「1 レースの最良の組合せ確率」を取り、その**レース間の中央値 × 1.25**

（ワイド 0.26 = 1.25 × 0.208、三連単 0.019 = 1.25 × 0.0152 と既存値に一致する）

**両モデルに同じ手順を当てる。** 片方だけ出し直すと「手で調整した NN」対
「新しく導出した GBDT」になってしまう。導出は test とは別の窓（valid 期間）で行う。
既存値が「評価に使ったのと同じ OOF から採っている」（bet_types.py の注意書き）
のに対し、ここは out-of-sample になっている。

なお両モデルとも valid 期間を early stopping に使っているので、そこは対称。

**重要な限界（2026-09-03 に判明）**: ここは NN 側に `win_prob`（温度スケール後の
softmax）を渡しているが、**本番は生の score を PL モンテカルロに渡している**
(`predict.py` の `compute_all_combination_probs(frame_scores, k=3, ...)`)。分布が
違うので、ここで出る下限は `DEFAULT_COMBO_MIN_HIT_PROB` と**直接は比べられない**
(実際 2〜5 倍ずれる)。GBDT と NN を同じ土俵で比べる分には内部整合しているが、
「既定値が古い」の根拠には**ならない**。本番忠実に測るには
`predict_race_with_combinations` を通すこと。

Usage:
    PYTHONPATH=src uv run python -m scripts.combo_threshold_refit --races 1000
"""

from __future__ import annotations

import argparse
import itertools
import sqlite3
from collections import defaultdict

import numpy as np
import pandas as pd

from ai.model.registry import load_model_full
from core.bet_types import DEFAULT_COMBO_MIN_HIT_PROB, normalize_combo
from core.logging import configure_logging, get_logger
from core.paths import db_path
from core.settings_store import SettingsStore, resolve_model_path
from db.session import make_engine, session_scope
from features.builder import CATEGORICAL_FEATURES, FEATURE_COLUMNS, build_training_frame

log = get_logger(__name__)
_EPS = 1e-12
_TOP_HORSES = 6
_MULTIPLIER = 1.25
_COMBOS = {"馬連": (2, False), "馬単": (2, True), "三連複": (3, False), "三連単": (3, True)}


def _prepare(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = frame[cols].copy()
    for c in cols:
        if c in CATEGORICAL_FEATURES:
            x[c] = x[c].astype("category")
    return x


def _norm(frame: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    s = pd.Series(np.clip(raw, _EPS, None), index=frame.index)
    return np.clip((s / s.groupby(frame["race_id"]).transform("sum")).to_numpy(), _EPS, 1 - _EPS)


def _pl(p: list[float]) -> float:
    remaining, out = 1.0, 1.0
    for pi in p:
        if remaining <= _EPS:
            return _EPS
        out *= pi / remaining
        remaining -= pi
    return out


def _all_combos(probs: dict[int, float], k: int, ordered: bool) -> list[tuple[str, float]]:
    top = sorted(probs, key=lambda h: -probs[h])[:_TOP_HORSES]
    out = []
    for c in itertools.combinations(top, k):
        if ordered:
            out += [("→".join(map(str, perm)), _pl([probs[h] for h in perm]))
                    for perm in itertools.permutations(c)]
        else:
            out.append(("-".join(map(str, sorted(c))),
                        sum(_pl([probs[h] for h in perm]) for perm in itertools.permutations(c))))
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
    ap.add_argument("--derive-races", type=int, default=600)
    ap.add_argument("--races", type=int, default=1000)
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
        valid_all = frame[(d > args.train_end) & (d <= args.valid_end)]
        test_all = frame[d > args.valid_end]

        def _sample(part: pd.DataFrame, n: int) -> pd.DataFrame:
            ids = list(dict.fromkeys(part["race_id"]))
            step = max(1, len(ids) // n)
            keep = set(ids[::step][:n])
            return part[part["race_id"].isin(keep)].reset_index(drop=True)

        derive = _sample(valid_all, args.derive_races)
        test = _sample(test_all, args.races)
        log.info("導出 %d レース / 評価 %d レース",
                 derive["race_id"].nunique(), test["race_id"].nunique())

        cols = [c for c in FEATURE_COLUMNS if c in frame.columns]
        booster = lgb.train(
            {
                "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.03,
                "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.8,
                "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1, "seed": 42,
            },
            lgb.Dataset(_prepare(train, cols), label=(train["finish_position"] == 1).astype(int)),
            num_boost_round=2000,
            valid_sets=[lgb.Dataset(_prepare(valid_all, cols),
                                    label=(valid_all["finish_position"] == 1).astype(int))],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        prob_path = resolve_model_path(SettingsStore().load().get("probability_model_path"))
        prob_nn = load_model_full(prob_path)
        log.info("確率モデル: %s", prob_path)

        def win_probs(part: pd.DataFrame) -> dict[str, dict[str, dict[int, float]]]:
            """{race_id: {出所: {馬番: 勝率}}}。"""
            g_win = _norm(part, np.asarray(booster.predict(
                _prepare(part, cols), num_iteration=booster.best_iteration)))
            out: dict[str, dict[str, dict[int, float]]] = {}
            for n, (race_id, idx) in enumerate(part.groupby("race_id", sort=False).indices.items(),
                                               start=1):
                g = part.iloc[idx]
                uma = {h: int(u) for h, u in zip(g["horse_id"], g["post_position"], strict=True)}
                try:
                    pn = predict_race(prob_nn, g, session=session)
                except Exception:  # noqa: BLE001
                    continue
                out[race_id] = {
                    "現行 NN": {uma[h]: float(v)
                                for h, v in zip(pn["horse_id"], pn["win_prob"], strict=True)},
                    "GBDT": {uma[h]: float(v)
                             for h, v in zip(g["horse_id"], g_win[idx], strict=True)},
                }
                if n % 200 == 0:
                    log.info("  %d/%d レース", n, len(part.groupby("race_id").indices))
            return out

        log.info("導出窓を予測中…")
        derive_p = win_probs(derive)
        log.info("評価窓を予測中…")
        test_p = win_probs(test)

    engine.dispose()

    # ── 下限を出し直す ────────────────────────────────────────────────────
    thresholds: dict[str, dict[str, float]] = defaultdict(dict)
    for label in ("現行 NN", "GBDT"):
        for bet_type, (k, ordered) in _COMBOS.items():
            best = [max(p for _c, p in _all_combos(src[label], k, ordered))
                    for src in derive_p.values() if label in src]
            thresholds[label][bet_type] = float(np.median(best)) * _MULTIPLIER

    print(f"\n出し直した下限（導出窓 {len(derive_p):,} レースの最良確率の中央値 × {_MULTIPLIER}）\n")
    print(f"{'券種':8s} {'既定 (NN 用)':>14} {'現行 NN 再導出':>16} {'GBDT 再導出':>14}")
    for bet_type in _COMBOS:
        print(f"{bet_type:8s} {DEFAULT_COMBO_MIN_HIT_PROB[bet_type]:>14.4f} "
              f"{thresholds['現行 NN'][bet_type]:>16.4f} {thresholds['GBDT'][bet_type]:>14.4f}")

    # ── 評価 ──────────────────────────────────────────────────────────────
    pay = _payouts(list(test_p))
    acc: dict[tuple[str, str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for race_id, src in test_p.items():
        for label in ("現行 NN", "GBDT"):
            if label not in src:
                continue
            for bet_type, (k, ordered) in _COMBOS.items():
                for rule, floor in (("既定", DEFAULT_COMBO_MIN_HIT_PROB[bet_type]),
                                    ("再導出", thresholds[label][bet_type])):
                    for combo, p in _all_combos(src[label], k, ordered):
                        if p <= floor:
                            continue
                        slot = acc[(label, bet_type, rule)]
                        slot[0] += 1
                        slot[1] += pay.get((race_id, bet_type, normalize_combo(combo)), 0.0)

    print(f"\n連系の回収率 — 評価 {len(test_p):,} レース\n")
    print(f"{'券種':8s} {'出所':8s} {'下限':>6} {'点/レース':>10} {'回収率':>8}")
    for bet_type in _COMBOS:
        for label in ("現行 NN", "GBDT"):
            for rule in ("既定", "再導出"):
                pts, ret = acc[(label, bet_type, rule)]
                if pts == 0:
                    continue
                print(f"{bet_type:8s} {label:8s} {rule:>6} "
                      f"{pts / len(test_p):>10.2f} {ret / pts:>8.3f}")
    print("\n『既定』は現行 NN の分布から作った値をそのまま使ったもの（前回の不公平な比較）。"
          "\n『再導出』は各モデル自身の分布から同じ手順で作り直したもの。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
