"""連系の現行仕様を、前進検証（別期間）で確かめる。

**なぜ要るか**: 「EV 条件を捨て、確率を確率専用モデルから出す」で連系計 0.849 →
0.877 になったが、これは test 19ヶ月という**単一窓**の測定で、しかもその窓は探索に
何度も使っている。同じ条件でオッズ帯は +0.48 が出て、4.5 年の前進検証で 0.789 まで
落ちて消えた。同じ轍を踏まないための検証。

各 fold は「その cutoff までで学習したモデル」で「まだ見ていない 6 ヶ月」を予測する。
買い目を決めるのは `multi` モデル、確率を出すのは `plackett_luce --pl-top-k 5` モデルで、
どちらもその fold のもの（本番の active / 確率モデルではない）。

**1 レースにつき推論は 2 回だけ**行い、そこから 2 構成を両方評価する。構成ごとに
シミュレーションを回すと推論が倍になり、Plackett-Luce の Monte Carlo が支配的な
このループでは時間も倍になる。

  旧: 買い目も確率も active 由来、``combo確率 × 推定オッズ > 1.1`` で選ぶ
  新: 買い目は active、確率は確率モデル由来、**EV を使わず**的中確率順に選ぶ

Usage:
  PYTHONIOENCODING=utf-8 PYTHONPATH=src python -m scripts.combo_walk_forward \
      --folds 2021-10-24,2022-04-24,2022-10-29,2023-04-29,2023-10-29
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from ai.betting.odds import compute_past_race_odds, compute_race_odds_with_sources
from ai.betting.strategy import recommend_for_race
from ai.inference.predict import (
    _combinations_from_base,
    _predict_race_nn,
    merge_combination_sources,
    predict_race,
    predict_race_with_combinations,
)
from ai.model.registry import load_model_full
from ai.simulation.engine import _settle_candidates
from core.paths import data_dir, db_path
from db.odds_db import init_odds_db, make_odds_engine
from db.session import make_engine, session_scope
from features.builder import build_training_frame

COMBO_TYPES = ["ワイド", "馬連", "馬単", "三連複", "三連単"]
UNITS = {"単勝": 500, "複勝": 500, "馬連": 100, "ワイド": 100,
         "馬単": 100, "三連複": 100, "三連単": 100}
RACE_BUDGET = 2_000
OLD_MIN_EV = 1.1


def _fold_models() -> dict[str, dict[str, str]]:
    """学習終了日 → {loss: モデルディレクトリ}。meta.json から復元する。

    walk_forward_oof は model_runs 行を消すので DB からは辿れない。ディスク上の
    meta.json が唯一の対応表になる。
    """
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for d in sorted((data_dir() / "models").glob("*-nn")):
        meta = d / "meta.json"
        if not meta.exists():
            continue
        m = json.loads(meta.read_text(encoding="utf-8"))
        tr = m.get("train_range") or ""
        if "/" not in tr:
            continue
        out[tr.split("/")[-1]][str(m.get("loss_type"))] = str(d)
    return out


def _payback(rows: list[dict], bet_types: list[str]) -> tuple[int, float]:
    sel = [r for r in rows if r["bet_type"] in bet_types]
    inv = sum(r["stake"] for r in sel)
    pay = sum(r["payout"] for r in sel)
    return len(sel), (pay / inv if inv else float("nan"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", required=True, help="カンマ区切りの学習終了日")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    models = _fold_models()
    engine = make_engine(db_path())
    odds_engine = make_odds_engine()
    init_odds_db(odds_engine)
    odds_session = Session(bind=odds_engine)

    settled: dict[str, list[dict]] = {"old": [], "new": []}
    per_fold: list[dict] = []

    for train_end in args.folds.split(","):
        pair = models.get(train_end, {})
        bet_dir, prob_dir = pair.get("multi"), pair.get("plackett_luce")
        if not bet_dir or not prob_dir:
            print(f"[skip] {train_end}: モデルが揃わない {list(pair)}", flush=True)
            continue

        meta = json.loads((Path(bet_dir) / "meta.json").read_text(encoding="utf-8"))
        test_start = (meta.get("test_range") or "/").split("/")[0]
        # 未見区間は 6 ヶ月ぶん。test_range の終端はフレーム末尾まで伸びるので使わない。
        end = (
            pd.Timestamp(test_start) + pd.DateOffset(months=6) - pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d")
        print(f"[fold] 学習 〜{train_end} / 検証 {test_start}..{end}", flush=True)

        bet_bundle = load_model_full(Path(bet_dir))
        prob_bundle = load_model_full(Path(prob_dir))

        with session_scope(engine) as session:
            frame = build_training_frame(session, train_start=test_start, train_end=end)
            race_ids = frame["race_id"].unique()
            fold_rows: dict[str, list[dict]] = {"old": [], "new": []}

            for i, rid in enumerate(race_ids):
                rf = frame[frame["race_id"] == rid]
                if len(rf) < 4:
                    continue
                finished = rf[rf["finish_position"].notna()]
                if finished.empty:
                    continue
                finish_to_pp: dict[int, int] = {}
                for _, row in finished.iterrows():
                    try:
                        finish_to_pp[int(row["finish_position"])] = int(row["post_position"])
                    except (ValueError, TypeError):
                        continue
                if not finish_to_pp:
                    continue

                try:
                    preds = predict_race(bet_bundle, rf, session=session)
                except Exception as exc:  # noqa: BLE001
                    print(f"  predict failed {rid}: {exc}", flush=True)
                    continue
                preds["post_position"] = preds["horse_id"].map(
                    dict(zip(rf["horse_id"], rf["post_position"], strict=True))
                )
                race_odds, sources = compute_race_odds_with_sources(
                    session, rid, odds_session=odds_session
                )
                # 推論は 1 レース 2 回だけ。ここから 2 構成を両方作る。
                combos_bet = predict_race_with_combinations(
                    bet_bundle, rf, session=session,
                    race_odds=race_odds, race_odds_sources=sources,
                )
                combos_prob = merge_combination_sources(
                    combos_bet,
                    _combinations_from_base(
                        base_df=_predict_race_nn(prob_bundle, rf, session=session),
                        frame=rf, n_samples=10_000, rng=None, top_k_combinations=None,
                        race_odds=race_odds, race_odds_sources=sources,
                    ),
                )
                past_odds = compute_past_race_odds(session, rid)

                for label, combos in (("old", combos_bet), ("new", combos_prob)):
                    c = dict(combos)
                    if label == "old":
                        # 旧仕様: 連系は EV 閾値で絞る
                        for bt in COMBO_TYPES:
                            c[bt] = [
                                x for x in c.get(bt, [])
                                if x.ev is not None and x.ev >= OLD_MIN_EV
                            ]
                    rec = recommend_for_race(
                        predictions=preds, combinations_by_type=c, race_id=rid,
                        race_budget=RACE_BUDGET, stake_unit=100,
                        stake_unit_by_bet_type=UNITS, win_min_odds=1.1,
                        top_n_horses=3, enabled_bet_types=["単勝", "複勝", *COMBO_TYPES],
                    )
                    bets = [x for x in rec.candidates if x.stake > 0]
                    fold_rows[label].extend(
                        _settle_candidates(bets, rid, finish_to_pp, past_odds)
                    )
                if (i + 1) % 200 == 0:
                    print(f"  {i + 1}/{len(race_ids)}", flush=True)

        row: dict = {"train_end": train_end, "window": f"{test_start}..{end}",
                     "races": int(len(race_ids))}
        for label in ("old", "new"):
            n, pb = _payback(fold_rows[label], COMBO_TYPES)
            row[f"{label}_n"], row[f"{label}_payback"] = n, round(pb, 4)
            settled[label].extend(fold_rows[label])
        per_fold.append(row)
        print(f"  -> 旧 {row['old_n']}点 {row['old_payback']:.3f} / "
              f"新 {row['new_n']}点 {row['new_payback']:.3f}", flush=True)

    print()
    header = "学習終了".ljust(12) + "検証区間".ljust(26)
    header += "旧EV 点数".rjust(10) + "旧EV".rjust(8) + "新 点数".rjust(10) + "新".rjust(8)
    print(header)
    for r in per_fold:
        print(f"  {r['train_end']:<12}{r['window']:<26}{r['old_n']:>10}"
              f"{r['old_payback']:>8.3f}{r['new_n']:>10}{r['new_payback']:>8.3f}")
    print()
    rng = np.random.default_rng(0)
    for label, name in (("old", "旧 (EV>1.1・active の確率)"),
                        ("new", "新 (EV なし・確率モデル)")):
        rows = settled[label]
        r = np.array(
            [x["payout"] / x["stake"] for x in rows if x["bet_type"] in COMBO_TYPES]
        )
        if r.size == 0:
            print(f"{name}: 点数なし")
            continue
        b = r[rng.integers(0, r.size, size=(3000, r.size))].mean(axis=1)
        print(f"{name:<28}{len(r):>8}点  回収率 {r.mean():.3f} "
              f"[{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]")
    if args.out:
        pd.DataFrame(per_fold).to_csv(args.out, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
