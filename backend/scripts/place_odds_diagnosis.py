"""複勝の推定オッズ・推定確率が実績とどれだけズレているかを測る。

backtest の複勝は 5,404 レースで 43,479 点 (8.1 点/レース) 発火して回収率 0.654 と、
1 番人気ベタ買いの複勝 0.850 にすら負けている。EV = 複勝確率 × 推定複勝オッズ の
**両側**が過大評価されうるので、どちらがどれだけ効いているかを分けて測る。

出力する診断:

  A. モデル複勝確率の較正
     予測確率で bin 分けし、実際の 3 着内率と比べる。1.0 より大きい比 = 過大評価。

  B. 推定複勝オッズの較正 (`_estimate_place_odds`)
     3 着内に入った馬について、推定オッズ vs 実際の払戻 (payout_place/100)。
     Harville (PL) 法はオッズ由来の複勝確率を**本命側で過大評価**することが知られており、
     そうなると推定オッズは本命側で**過小**、穴側で**過大**に出るはず。

  C. 市場の複勝確率の較正
     推定に使っている「オッズ→PL→3着内確率」自体を実績と比べる (B の原因切り分け)。

  D. EV 帯ごとの実現回収率
     EV>1.05 で買うという現行ルールが、どの EV 帯で損をしているか。

Usage:
  PYTHONIOENCODING=utf-8 PYTHONPATH=src python -m scripts.place_odds_diagnosis \\
      --start 2024-11-02 --end 2026-05-31 --out ../data/reports/place_diag.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ai.evaluation.backtest import _estimate_place_odds, _parse_payout_place
from ai.inference.predict import predict_race
from ai.model.registry import get_active, load_model_full
from core.paths import db_path
from db.session import make_engine, session_scope
from features.builder import build_training_frame


def _bin_report(df: pd.DataFrame, col: str, edges: list[float], label: str) -> list[dict]:
    """col で bin 分けし、予測平均 vs 実測 3 着内率を返す。"""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = df[(df[col] >= lo) & (df[col] < hi)]
        if m.empty:
            continue
        pred = float(m[col].mean())
        actual = float(m["placed"].mean())
        out.append({
            "band": f"{lo:.2f}-{hi:.2f}",
            "n": int(len(m)),
            f"pred_{label}": round(pred, 4),
            "actual": round(actual, 4),
            "ratio_pred_over_actual": round(pred / actual, 3) if actual > 0 else None,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--takeout", type=float, default=0.20)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    engine = make_engine(db_path())
    with session_scope(engine) as s0:
        model_path = get_active(s0)
    if model_path is None:
        raise SystemExit("no active model")
    bundle = load_model_full(model_path)
    print(f"model: {model_path}", flush=True)

    rows: list[pd.DataFrame] = []
    with session_scope(engine) as session:
        frame = build_training_frame(session, train_start=args.start, train_end=args.end)
        race_ids = frame["race_id"].unique()
        print(f"{len(race_ids)} races", flush=True)
        for i, rid in enumerate(race_ids):
            rf = frame[frame["race_id"] == rid]
            if len(rf) < 2:
                continue
            payout_map = {}
            if "payout_place" in rf.columns:
                vals = rf["payout_place"].dropna()
                if not vals.empty:
                    payout_map = _parse_payout_place(vals.iloc[0])
            if not payout_map:
                continue

            est = _estimate_place_odds(rf, takeout=args.takeout)
            if not est:
                continue
            preds = predict_race(bundle, rf, session=session)
            m = preds.merge(
                rf[["horse_id", "finish_position", "odds_win", "popularity"]],
                on="horse_id", how="left",
            )
            m["race_id"] = rid
            m["est_place_odds"] = m["horse_id"].map(est)
            fin = m["finish_position"]
            m["placed"] = ((fin.notna()) & (fin <= 3) & (fin == fin.round())).astype(float)
            # 実際の複勝払戻 (3 着内の馬のみ。100 円あたり → decimal odds)
            m["actual_place_odds"] = [
                payout_map.get(int(f)) / 100.0
                if pd.notna(f) and float(f) == int(f) and int(f) in payout_map
                else np.nan
                for f in fin
            ]
            rows.append(m)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(race_ids)}", flush=True)

    d = pd.concat(rows, ignore_index=True)
    d = d.dropna(subset=["est_place_odds", "place_prob"])
    # 市場 (オッズ) 由来の複勝確率 = (1-takeout)/推定オッズ の逆算
    d["market_place_prob"] = (1.0 - args.takeout) / d["est_place_odds"]
    d["ev"] = d["place_prob"] * d["est_place_odds"]

    result: dict = {
        "model": str(model_path),
        "window": [args.start, args.end],
        "n_horse_rows": int(len(d)),
        "n_races": int(d["race_id"].nunique()),
        "overall_place_rate": round(float(d["placed"].mean()), 4),
    }

    edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.01]
    result["A_model_place_prob_calibration"] = _bin_report(d, "place_prob", edges, "prob")
    result["C_market_place_prob_calibration"] = _bin_report(d, "market_place_prob", edges, "prob")

    # B. 3 着内に入った馬での推定オッズ vs 実払戻
    placed = d[d["placed"] > 0].dropna(subset=["actual_place_odds"])
    b: list[dict] = []
    for lo, hi in [(1.0, 1.3), (1.3, 1.6), (1.6, 2.2), (2.2, 3.5), (3.5, 6.0), (6.0, 1e9)]:
        m = placed[(placed["actual_place_odds"] >= lo) & (placed["actual_place_odds"] < hi)]
        if m.empty:
            continue
        b.append({
            "actual_odds_band": f"{lo}-{hi if hi < 1e9 else 'inf'}",
            "n": int(len(m)),
            "mean_est_odds": round(float(m["est_place_odds"].mean()), 3),
            "mean_actual_odds": round(float(m["actual_place_odds"].mean()), 3),
            "ratio_est_over_actual": round(
                float(m["est_place_odds"].mean() / m["actual_place_odds"].mean()), 3),
        })
    result["B_est_vs_actual_place_odds"] = b

    # D. EV 帯ごとの実現回収率 (1 点 100 円固定)
    dd: list[dict] = []
    for lo, hi in [(0.0, 0.9), (0.9, 1.0), (1.0, 1.05), (1.05, 1.2), (1.2, 1.5),
                   (1.5, 2.0), (2.0, 1e9)]:
        m = d[(d["ev"] >= lo) & (d["ev"] < hi)]
        if m.empty:
            continue
        ret = np.where(m["placed"] > 0, m["actual_place_odds"].fillna(0.0), 0.0)
        dd.append({
            "ev_band": f"{lo}-{hi if hi < 1e9 else 'inf'}",
            "n_bets": int(len(m)),
            "hit_rate": round(float(m["placed"].mean()), 4),
            "payback": round(float(ret.sum() / len(m)), 4),
        })
    result["D_payback_by_ev_band"] = dd

    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
