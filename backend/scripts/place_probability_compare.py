"""3 着内率の質を GBDT / 現行の確率モデル / 市場 で比べる。

確率モデルの枠 (`settings.probability_model_path`) が実際に使うのは **3 着内率**
(複勝を買うかの判定 `place_min_hit_prob` と、点数の重みづけ)。だから勝率ではなく
ここで比べないと、差し替えの是非は決まらない。

GBDT を枠に挿すには 8ff87dd で撤去された bundle 経路を作り直す必要がある。
**作る前に、作る価値があるかを測る**のがこのスクリプト。

比べるもの (3 着内に入るかの二値):
  gbdt        finish_position <= 3 を直接学習
  現行 prob   settings.probability_model_path の NN の place_prob
  市場        複勝オッズ (odds.db) の 1/オッズ

指標:
  NLL     二値の負の対数尤度。小さいほど良い (proper scoring rule)
  Brier   二乗誤差。較正のずれに敏感
  corr    予測確率と実際の 3 着内の相関

8 頭立て未満は除く (複勝が 2 着までになる / 発売が無い)。確率はレース内で
合計 3 に正規化する — ちょうど 3 頭が 3 着内に入るという制約を使う。

NN は 1 レース約 1.6 秒かかるので --races で標本を絞れる (既定 1200)。

Usage:
    PYTHONPATH=src uv run python -m scripts.place_probability_compare
"""

from __future__ import annotations

import argparse
import gzip
import json
import sqlite3

import numpy as np
import pandas as pd

from ai.model.registry import load_model_full
from core.logging import configure_logging, get_logger
from core.paths import db_path, odds_db_path
from db.session import make_engine, session_scope
from features.builder import CATEGORICAL_FEATURES, FEATURE_COLUMNS, build_training_frame

log = get_logger(__name__)
_EPS = 1e-9


def _prepare(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = frame[cols].copy()
    for c in cols:
        if c in CATEGORICAL_FEATURES:
            x[c] = x[c].astype("category")
    return x


def _normalise_to_three(frame: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    """レース内で合計 3 にそろえる。ちょうど 3 頭が 3 着内に入るため。"""
    s = pd.Series(np.clip(raw, _EPS, None), index=frame.index)
    total = s.groupby(frame["race_id"]).transform("sum")
    return np.clip((3.0 * s / total).to_numpy(), _EPS, 1 - _EPS)


def _market_place(frame: pd.DataFrame) -> np.ndarray | None:
    """odds.db の複勝オッズから 1/オッズ。取れない馬は NaN。"""
    con = sqlite3.connect(f"file:{odds_db_path()}?mode=ro", uri=True)
    race_ids = list(dict.fromkeys(frame["race_id"]))
    book: dict[str, dict[str, float]] = {}
    for i in range(0, len(race_ids), 500):
        chunk = race_ids[i : i + 500]
        q = ",".join("?" * len(chunk))
        for race_id, blob in con.execute(
            f"SELECT race_id, data FROM race_odds WHERE bet_type='複勝' AND race_id IN ({q})",
            chunk,
        ):
            d = json.loads(gzip.decompress(blob))
            # 複勝は [min, max, 人気]。期待値としては中間を採る
            book[race_id] = {
                k: (float(v[0]) + float(v[1])) / 2 if float(v[1]) > 0 else float(v[0])
                for k, v in d.items()
                if v and float(v[0]) > 0
            }
    con.close()
    out = np.full(len(frame), np.nan)
    for i, (race_id, umaban) in enumerate(
        zip(frame["race_id"], frame["post_position"], strict=True)
    ):
        o = book.get(race_id, {}).get(str(int(umaban))) if pd.notna(umaban) else None
        if o:
            out[i] = 1.0 / o
    return out


def _score(label: str, p: np.ndarray, y: np.ndarray, races: int) -> dict:
    ok = np.isfinite(p)
    p, y = np.clip(p[ok], _EPS, 1 - _EPS), y[ok]
    nll = float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
    return {
        "label": label,
        "races": races,
        "horses": int(ok.sum()),
        "nll": nll,
        "brier": float(np.mean((p - y) ** 2)),
        "corr": float(np.corrcoef(p, y)[0, 1]),
        "mean_p": float(np.mean(p)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-end", default="2024-04-30")
    ap.add_argument("--valid-end", default="2024-10-31")
    ap.add_argument("--races", type=int, default=1200, help="NN を回す test レース数")
    args = ap.parse_args()

    configure_logging()
    import lightgbm as lgb

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        frame = build_training_frame(session)

        frame = frame[frame["finish_position"].notna() & frame["post_position"].notna()]
        frame = frame[frame["n_runners"] >= 8].reset_index(drop=True)
        d = frame["date"]
        train = frame[d <= args.train_end]
        valid = frame[(d > args.train_end) & (d <= args.valid_end)]
        test_all = frame[d > args.valid_end]

        # NN が重いので test から等間隔に標本を取る
        race_ids = list(dict.fromkeys(test_all["race_id"]))
        step = max(1, len(race_ids) // args.races)
        picked = set(race_ids[::step][: args.races])
        test = test_all[test_all["race_id"].isin(picked)].reset_index(drop=True)
        y = (test["finish_position"] <= 3).to_numpy().astype(float)
        log.info("test %d レース / %d 行 (全 %d レースから抽出)",
                 len(picked), len(test), len(race_ids))

        rows = []

        # ── 市場 ──────────────────────────────────────────────────────────
        mkt = _market_place(test)
        have = np.isfinite(mkt)
        if have.any():
            m = np.full(len(test), np.nan)
            sub = test[have]
            m[have] = _normalise_to_three(sub, mkt[have])
            rows.append(_score("市場 (複勝オッズ)", m, y, len(picked)))

        # ── GBDT ──────────────────────────────────────────────────────────
        cols = [c for c in FEATURE_COLUMNS if c in frame.columns]
        booster = lgb.train(
            {
                "objective": "binary", "metric": "binary_logloss", "learning_rate": 0.03,
                "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.8,
                "bagging_fraction": 0.8, "bagging_freq": 1, "verbosity": -1, "seed": 42,
            },
            lgb.Dataset(_prepare(train, cols), label=(train["finish_position"] <= 3).astype(int)),
            num_boost_round=2000,
            valid_sets=[
                lgb.Dataset(_prepare(valid, cols), label=(valid["finish_position"] <= 3).astype(int))
            ],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        raw = booster.predict(_prepare(test, cols), num_iteration=booster.best_iteration)
        rows.append(_score("gbdt", _normalise_to_three(test, np.asarray(raw)), y, len(picked)))
        log.info("gbdt best_iteration=%d", booster.best_iteration)

        # ── 現行の確率モデル ───────────────────────────────────────────────
        from ai.inference.predict import predict_race
        from core.settings_store import SettingsStore, resolve_model_path

        path = resolve_model_path(SettingsStore().load().get("probability_model_path"))
        if path is None:
            log.warning("probability_model_path が未設定。NN の比較は飛ばす")
        else:
            bundle = load_model_full(path)
            log.info("確率モデル: %s", path)
            nn_p = np.full(len(test), np.nan)
            idx_by_race = test.groupby("race_id", sort=False).indices
            for n, (_race_id, idx) in enumerate(idx_by_race.items(), start=1):
                g = test.iloc[idx]
                try:
                    preds = predict_race(bundle, g, session=session)
                except Exception:  # noqa: BLE001 — 1 レースの失敗で全体を止めない
                    continue
                by_horse = dict(zip(preds["horse_id"], preds["place_prob"], strict=False))
                for j, hid in zip(idx, g["horse_id"], strict=True):
                    v = by_horse.get(hid)
                    if v is not None:
                        nn_p[j] = v
                if n % 200 == 0:
                    log.info("  NN %d/%d レース", n, len(idx_by_race))
            rows.append(_score("現行 prob (NN)", nn_p, y, len(picked)))

    engine.dispose()

    print(f"\n3 着内に入るかの二値評価 — {rows[0]['races']:,} レース\n")
    print(f"{'':20s} {'頭数':>8} {'NLL':>8} {'Brier':>8} {'corr':>7} {'平均確率':>9}")
    for r in rows:
        print(f"{r['label']:20s} {r['horses']:>8,} {r['nll']:>8.4f} {r['brier']:>8.4f} "
              f"{r['corr']:>7.3f} {r['mean_p']:>9.3f}")
    print("\nNLL / Brier は小さいほど良い。実際の 3 着内率は "
          f"{float(np.mean(test['finish_position'] <= 3)):.3f}。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
