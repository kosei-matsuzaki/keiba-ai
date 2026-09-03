"""連系の確率を NN と GBDT のどちらから作るか。期間をまたいで比べる。

単一窓 1,000 レースの連系回収率は判断材料にならない。今日 2 回、1.0 を超えた結果が
前進検証で消えている（複勝のバリューベット、オッズ帯の選別）。ここでは**評価窓を
転がして**、結果が期間をまたいで安定しているかを見る。

**本番と同じ経路を通す。** 前回の測定は NN に `win_prob`（温度スケール後）を渡して
いたが、本番は生の score を PL モンテカルロに渡している
(`compute_all_combination_probs(frame_scores, k=3)`)。ここは両モデルともその機械に
かける。GBDT には `log(p_win)` を渡す — softmax(log p) = p なので、同じ機械で
GBDT 自身の勝率分布を使ったことになる。

**モデルは両方とも固定。** NN の確率モデルは 2015〜2024-04 で学習された artifact で、
fold ごとに学習し直すと 1 fold 40 分かかる。そこで GBDT も同じ期間で固定し、
**評価窓だけを転がす**。retrain の前進検証ではなく「期間をまたいだ安定性」の検証。
今日 2 回とも壊れたのはそこなので、目的には合っている。

下限は導出窓（2024-05〜10、valid 期間）で各モデルごとに
「1 レースの最良組合せ確率の中央値 × 1.25」から作る（`core/bet_types.py` の recipe）。
評価窓は使わない。

Usage:
    PYTHONPATH=src uv run python -m scripts.combo_source_walk_forward
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict
from itertools import combinations, permutations

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from ai.core.probabilities import compute_all_combination_probs
from ai.model.registry import load_model_full
from core.bet_types import normalize_combo
from core.logging import configure_logging, get_logger
from core.paths import db_path
from core.settings_store import SettingsStore, resolve_model_path
from db.session import make_engine, session_scope
from features.builder import CATEGORICAL_FEATURES, FEATURE_COLUMNS, build_training_frame

log = get_logger(__name__)
_EPS = 1e-12
_MULTIPLIER = 1.25
_N_SAMPLES = 10_000
_BET_TYPES = ("馬連", "馬単", "三連複", "三連単")


def _prepare(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = frame[cols].copy()
    for c in cols:
        if c in CATEGORICAL_FEATURES:
            x[c] = x[c].astype("category")
    return x


def _combo_probs(scores: np.ndarray, post: np.ndarray,
                 rng: np.random.Generator) -> dict[str, dict[str, float]]:
    """本番と同じモンテカルロから券種ごとの {combo: 確率} を作る。"""
    n = len(scores)
    cp = compute_all_combination_probs(scores, k=3, n_samples=_N_SAMPLES, rng=rng)
    pair, opair = cp["pair"], cp["ordered_pair"]
    triple, otriple = cp["triple"], cp["ordered_triple"]
    out: dict[str, dict[str, float]] = {b: {} for b in _BET_TYPES}
    for i, j in combinations(range(n), 2):
        lo, hi = sorted((int(post[i]), int(post[j])))
        out["馬連"][f"{lo}-{hi}"] = float(pair[i, j])
    for i, j in permutations(range(n), 2):
        out["馬単"][f"{int(post[i])}→{int(post[j])}"] = float(opair[i, j])
    for fs, p in triple.items():
        nums = sorted(int(post[x]) for x in fs)
        out["三連複"]["-".join(map(str, nums))] = float(p)
    for i, j, k in permutations(range(n), 3):
        p = float(otriple[i, j, k])
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
    ap.add_argument("--window-months", type=int, default=4)
    ap.add_argument("--races-per-window", type=int, default=260)
    ap.add_argument("--derive-races", type=int, default=400)
    args = ap.parse_args()

    configure_logging()
    import lightgbm as lgb

    from ai.inference.predict import _predict_race_nn

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        frame = build_training_frame(session)
        frame = frame[frame["finish_position"].notna() & frame["post_position"].notna()]
        frame = frame[frame["odds_win"].notna() & (frame["n_runners"] >= 8)].reset_index(drop=True)
        d = pd.to_datetime(frame["date"])
        train = frame[d <= pd.Timestamp(args.train_end)]
        valid = frame[(d > pd.Timestamp(args.train_end)) & (d <= pd.Timestamp(args.valid_end))]

        cols = [c for c in FEATURE_COLUMNS if c in frame.columns]
        booster = lgb.train(
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
        prob_path = resolve_model_path(SettingsStore().load().get("probability_model_path"))
        prob_nn = load_model_full(prob_path)
        log.info("確率モデル: %s / gbdt=%d 本", prob_path, booster.best_iteration)

        def per_race(part: pd.DataFrame, tag: str) -> dict[str, dict[str, dict[str, float]]]:
            """{race_id: {出所: {券種: {combo: 確率}}}}"""
            rng = np.random.default_rng(0)
            out: dict[str, dict[str, dict[str, float]]] = {}
            groups = part.groupby("race_id", sort=False).indices
            for n, (race_id, idx) in enumerate(groups.items(), start=1):
                g = part.iloc[idx]
                post = g["post_position"].to_numpy()
                try:
                    preds = _predict_race_nn(prob_nn, g, session=session)
                except Exception:  # noqa: BLE001
                    continue
                # _predict_race_nn は score 降順で返すので馬 ID で並べ直す
                by_horse = dict(zip(preds["horse_id"], preds["score"], strict=True))
                nn_scores = np.array([by_horse[h] for h in g["horse_id"]])
                raw = np.clip(booster.predict(
                    _prepare(g, cols), num_iteration=booster.best_iteration), _EPS, None)
                gb_scores = np.log(raw / raw.sum())  # softmax(log p) = p
                out[race_id] = {
                    "NN": _combo_probs(nn_scores, post, rng),
                    "GBDT": _combo_probs(gb_scores, post, rng),
                }
                if n % 200 == 0:
                    log.info("  [%s] %d/%d レース", tag, n, len(groups))
            return out

        def sample(part: pd.DataFrame, n: int) -> pd.DataFrame:
            ids = list(dict.fromkeys(part["race_id"]))
            step = max(1, len(ids) // n)
            return part[part["race_id"].isin(set(ids[::step][:n]))].reset_index(drop=True)

        derive = per_race(sample(valid, args.derive_races), "導出")

        windows: list[tuple[str, dict]] = []
        start = pd.Timestamp(args.valid_end) + pd.Timedelta(days=1)
        data_end = pd.Timestamp(str(frame["date"].max()))
        while start < data_end:
            end = start + relativedelta(months=args.window_months)
            part = frame[(d >= start) & (d < end)]
            if part["race_id"].nunique() < 50:
                break
            label = f"{start.date()}..{min(end, data_end).date()}"
            windows.append((label, per_race(sample(part, args.races_per_window), label)))
            start = end

    engine.dispose()

    # ── 下限を導出窓から ──────────────────────────────────────────────────
    floors: dict[str, dict[str, float]] = defaultdict(dict)
    for src in ("NN", "GBDT"):
        for bet_type in _BET_TYPES:
            best = [max(r[src][bet_type].values()) for r in derive.values()
                    if r[src][bet_type]]
            floors[src][bet_type] = float(np.median(best)) * _MULTIPLIER
    print(f"\n下限（導出窓 {len(derive):,} レース・最良確率の中央値 × {_MULTIPLIER}）\n")
    print(f"{'券種':8s} {'NN':>10} {'GBDT':>10}")
    for bet_type in _BET_TYPES:
        print(f"{bet_type:8s} {floors['NN'][bet_type]:>10.4f} {floors['GBDT'][bet_type]:>10.4f}")

    pay = _payouts([r for _l, w in windows for r in w])
    print(f"\n窓ごとの連系回収率（{args.window_months} ヶ月刻み・下限は導出窓由来）\n")
    header = "".join(f"{lbl.split('..')[0][:7]:>10}" for lbl, _ in windows)
    print(f"{'券種':8s} {'出所':6s}{header}{'全体':>10} {'点/R':>7}")
    for bet_type in _BET_TYPES:
        for src in ("NN", "GBDT"):
            cells, tot_p, tot_r, tot_races = [], 0.0, 0.0, 0
            for _lbl, w in windows:
                pts = ret = 0.0
                for race_id, r in w.items():
                    for combo, p in r[src][bet_type].items():
                        if p > floors[src][bet_type]:
                            pts += 1
                            ret += pay.get((race_id, bet_type, normalize_combo(combo)), 0.0)
                cells.append(f"{ret / pts:>10.3f}" if pts else f"{'—':>10}")
                tot_p += pts
                tot_r += ret
                tot_races += len(w)
            print(f"{bet_type:8s} {src:6s}{''.join(cells)}"
                  f"{(tot_r / tot_p if tot_p else float('nan')):>10.3f} {tot_p / tot_races:>7.2f}")
    print("\n窓をまたいで符号が揃わないなら、その差は期間固有のノイズ。"
          "\n1 つの窓だけ 1.0 を超えても意味がない（今日 2 回それで消えた）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
