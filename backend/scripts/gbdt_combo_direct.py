"""連系を組合せ単位で直接学習する。PL 変換で失っている分を取り返せるか。

`gbdt_exotic_check.py` で、勝率は市場と互角なのに PL で作った連系は負けることが
分かった（馬連 +0.023 → 三連単 +0.129）。損しているのは勝率ではなく変換のほう。
PL は「2 着は勝ち馬を除いた再レース」と仮定するが、実際の着順には脚質・展開の
依存がある。ならば組合せを直接学習すれば取り返せるはず、というのがこの検証。

対象は**ペア系まで**。平均 14 頭で 1 レースあたり 馬連 91 / 馬単 182 / 三連複 364 /
三連単 2,184 行になり、三連単は 30,000 レースで 6,500 万行になって現実的でない。

**PL 確率を特徴量に入れる。** そうすればモデルは PL からの「ずれ」だけを学べばよく、
ゼロから組合せの確率を覚え直さずに済む。

リーク対策: 馬ごとのモデルと組合せモデルを**別の期間**で学習する。同じ期間だと
組合せモデルが「馬モデルが過学習した予測」を入力に取ることになる。

    期間 A (最古)  → 馬ごとの GBDT (勝率 / 3 着内率) を学習
    期間 B         → A のモデルで予測し、その出力を特徴量に組合せモデルを学習
    valid / test   → A のモデルで予測 → 組合せモデルで予測

Usage:
    PYTHONPATH=src uv run python -m scripts.gbdt_combo_direct
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import sqlite3

import numpy as np
import pandas as pd

from core.logging import configure_logging, get_logger
from core.paths import db_path, odds_db_path
from db.session import make_engine, session_scope
from features.builder import CATEGORICAL_FEATURES, FEATURE_COLUMNS, build_training_frame

log = get_logger(__name__)
_EPS = 1e-12

# 券種 -> (順序を見るか, 当たりが何通りあるか)
_PAIRS = {"馬連": (False, 1), "馬単": (True, 1), "ワイド": (False, 3)}


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


def _pl_pair(pi: float, pj: float, ordered: bool) -> float:
    """PL で「i,j が 1-2 着」。順不同なら両順を足す。"""
    a = pi * pj / max(1 - pi, _EPS)
    if ordered:
        return a
    return a + pj * pi / max(1 - pj, _EPS)


def _fit_horse_model(train: pd.DataFrame, valid: pd.DataFrame, cols: list[str], label: str):
    import lightgbm as lgb

    def y(part: pd.DataFrame) -> pd.Series:
        return (part["finish_position"] == 1).astype(int) if label == "win" \
            else (part["finish_position"] <= 3).astype(int)

    return lgb.train(
        {
            "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.03,
            "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1, "seed": 42,
        },
        lgb.Dataset(_prepare(train, cols), label=y(train)),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(_prepare(valid, cols), label=y(valid))],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )


def _combo_rows(part: pd.DataFrame, p_win: np.ndarray, p_place: np.ndarray,
                bet_type: str) -> pd.DataFrame:
    """1 レース分のペアを 1 行ずつに展開する。"""
    ordered, _ = _PAIRS[bet_type]
    part = part.reset_index(drop=True)
    out = []
    for race_id, idx in part.groupby("race_id", sort=False).indices.items():
        n = len(idx)
        if n < 4:
            continue
        uma = [int(v) for v in part["post_position"].to_numpy()[idx]]
        fin = part["finish_position"].to_numpy()[idx]
        pw, pp = p_win[idx], p_place[idx]
        odds = part["odds_win"].to_numpy()[idx]
        pairs = itertools.permutations(range(n), 2) if ordered else itertools.combinations(range(n), 2)
        for a, b in pairs:
            if ordered:
                hit = 1 if (fin[a] == 1 and fin[b] == 2) else 0
            elif _PAIRS[bet_type][1] == 3:  # ワイド: 両方 3 着以内
                hit = 1 if (fin[a] <= 3 and fin[b] <= 3) else 0
            else:  # 馬連
                hit = 1 if ({fin[a], fin[b]} == {1.0, 2.0}) else 0
            out.append({
                "race_id": race_id,
                "combo": ("→" if ordered else "-").join(
                    str(x) for x in ((uma[a], uma[b]) if ordered else sorted((uma[a], uma[b])))
                ),
                "hit": hit,
                "n_runners": n,
                "pl": _pl_pair(float(pw[a]), float(pw[b]), ordered),
                "p_win_a": pw[a], "p_win_b": pw[b],
                "p_place_a": pp[a], "p_place_b": pp[b],
                "p_win_prod": pw[a] * pw[b],
                "p_place_prod": pp[a] * pp[b],
                "p_win_min": min(pw[a], pw[b]), "p_win_max": max(pw[a], pw[b]),
                "odds_a": odds[a], "odds_b": odds[b],
            })
    return pd.DataFrame(out)


_COMBO_FEATS = ["n_runners", "pl", "p_win_a", "p_win_b", "p_place_a", "p_place_b",
                "p_win_prod", "p_place_prod", "p_win_min", "p_win_max", "odds_a", "odds_b"]


def _market(race_ids: list[str], bet_type: str) -> dict[str, dict[str, float]]:
    con = sqlite3.connect(f"file:{odds_db_path()}?mode=ro", uri=True)
    out: dict[str, dict[str, float]] = {}
    for i in range(0, len(race_ids), 500):
        chunk = race_ids[i : i + 500]
        q = ",".join("?" * len(chunk))
        for race_id, blob in con.execute(
            f"SELECT race_id, data FROM race_odds WHERE bet_type=? AND race_id IN ({q})",
            [bet_type, *chunk],
        ):
            d = json.loads(gzip.decompress(blob))
            out[race_id] = {k: float(v[0]) for k, v in d.items() if v and float(v[0]) > 0}
    con.close()
    return out


def _nll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, _EPS, 1 - _EPS)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horse-end", default="2021-12-31", help="期間 A の終わり")
    ap.add_argument("--combo-end", default="2024-04-30", help="期間 B の終わり")
    ap.add_argument("--valid-end", default="2024-10-31")
    ap.add_argument("--test-races", type=int, default=2500)
    args = ap.parse_args()

    configure_logging()
    import lightgbm as lgb

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        frame = build_training_frame(session)
    engine.dispose()

    frame = frame[frame["finish_position"].notna() & frame["post_position"].notna()]
    frame = frame[frame["odds_win"].notna() & (frame["n_runners"] >= 8)].reset_index(drop=True)
    d = frame["date"]
    a = frame[d <= args.horse_end]
    a_valid = a[a["date"] > str(pd.Timestamp(args.horse_end) - pd.DateOffset(months=6))]
    b = frame[(d > args.horse_end) & (d <= args.combo_end)]
    valid = frame[(d > args.combo_end) & (d <= args.valid_end)]
    test_all = frame[d > args.valid_end]
    ids = list(dict.fromkeys(test_all["race_id"]))[: args.test_races]
    test = test_all[test_all["race_id"].isin(set(ids))].reset_index(drop=True)
    log.info("A=%d 行 / B=%d 行 / valid=%d / test=%d 行 (%d レース)",
             len(a), len(b), len(valid), len(test), len(ids))

    cols = [c for c in FEATURE_COLUMNS if c in frame.columns]
    horse_win = _fit_horse_model(a, a_valid, cols, "win")
    horse_place = _fit_horse_model(a, a_valid, cols, "place")
    log.info("馬モデル win=%d / place=%d 本", horse_win.best_iteration, horse_place.best_iteration)

    def probs(part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        part = part.reset_index(drop=True)
        w = _norm(part, np.asarray(horse_win.predict(
            _prepare(part, cols), num_iteration=horse_win.best_iteration)), 1.0)
        p = _norm(part, np.asarray(horse_place.predict(
            _prepare(part, cols), num_iteration=horse_place.best_iteration)), 3.0)
        return w, p

    print(f"\n組合せの NLL（小さいほど良い）— test {len(ids):,} レース\n")
    print(f"{'券種':6s} {'行数':>10} {'直接 GBDT':>10} {'PL のみ':>9} {'市場':>9}")
    for bet_type in _PAIRS:
        parts = {}
        for name, part in (("b", b), ("valid", valid), ("test", test)):
            pw, pp = probs(part)
            parts[name] = _combo_rows(part, pw, pp, bet_type)
        tr, va, te = parts["b"], parts["valid"], parts["test"]
        if te.empty:
            continue

        booster = lgb.train(
            {
                "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.05,
                "num_leaves": 63, "min_data_in_leaf": 500, "feature_fraction": 0.8,
                "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1, "seed": 42,
            },
            lgb.Dataset(tr[_COMBO_FEATS], label=tr["hit"]),
            num_boost_round=1500,
            valid_sets=[lgb.Dataset(va[_COMBO_FEATS], label=va["hit"])],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        total = float(_PAIRS[bet_type][1])
        p_direct = _norm(te, np.asarray(
            booster.predict(te[_COMBO_FEATS], num_iteration=booster.best_iteration)), total)
        p_pl = _norm(te, te["pl"].to_numpy(), total)

        book = _market(list(dict.fromkeys(te["race_id"])), bet_type)
        mkt = np.array([book.get(r, {}).get(c, np.nan)
                        for r, c in zip(te["race_id"], te["combo"], strict=True)])
        ok = np.isfinite(mkt)
        y = te["hit"].to_numpy().astype(float)
        p_mkt = np.full(len(te), np.nan)
        p_mkt[ok] = _norm(te[ok], 1.0 / mkt[ok], total)

        print(f"{bet_type:6s} {len(te):>10,} {_nll(p_direct[ok], y[ok]):>10.4f} "
              f"{_nll(p_pl[ok], y[ok]):>9.4f} {_nll(p_mkt[ok], y[ok]):>9.4f}")
        gain = booster.feature_importance("gain")
        top = sorted(zip(_COMBO_FEATS, gain, strict=True), key=lambda x: -x[1])[:4]
        log.info("  %s の効いた特徴: %s", bet_type,
                 ", ".join(f"{k}({v / gain.sum():.0%})" for k, v in top))

    print("\n直接 GBDT が PL より小さければ「変換で損をしていた」が裏付けられる。"
          "\n市場より小さくなって初めて、その券種にエッジがある。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
