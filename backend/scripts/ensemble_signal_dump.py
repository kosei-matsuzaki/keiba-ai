"""2 モデルを「どの馬か」と「どれくらい確からしいか」で分担させ、その素材を書き出す。

  * 買う馬を決めるのは **active** (回収率で鍛えた `multi`)。順位は良いが、確率の
    大きさは実際の勝敗とほぼ無相関 (r=0.047、市場は 0.354)。
  * 確からしさを答えるのは **確率モデル** (proper scoring rule = `plackett_luce`)。
    順位/ROI は劣る想定だが、確率としての意味を持つことが期待される。

1 レース 1 行で、両モデルの予測と実際の結果を書き出す。**確率モデルの確率は
「確率モデル自身の本命」ではなく「active の本命の馬」について記録する** —
知りたいのは「active が買う馬をどれくらい信じてよいか」なので。

2 モデルが同じ馬を選んだかどうか (agreement) も残す。目的関数の違う 2 つが一致する
ことは、それ自体が確信度の signal になりうる。

Usage:
  PYTHONIOENCODING=utf-8 PYTHONPATH=src python -m scripts.ensemble_signal_dump \
      --bet-model ../data/models/20260613T114817-nn \
      --prob-model ../data/models/<PL の ts>-nn \
      --start 2024-11-02 --end 2026-05-31 --out ../data/analysis/ensemble_signals.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from ai.evaluation.backtest import _parse_payout_place
from ai.inference.predict import predict_race
from ai.model.registry import load_model_full
from core.paths import db_path
from db.session import make_engine, session_scope
from features.builder import build_training_frame


def _f(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _place_return(payout_map: dict[int, float], finish) -> float | None:
    if not payout_map or finish is None or pd.isna(finish):
        return None
    fin = int(finish)
    if fin in payout_map:
        return payout_map[fin] / 100.0
    return 0.0 if fin > 3 else None


def _row(rf: pd.DataFrame, bet_preds: pd.DataFrame, prob_preds: pd.DataFrame) -> dict | None:
    bet_top = bet_preds.iloc[0]
    row = rf[rf["horse_id"] == bet_top["horse_id"]]
    if row.empty:
        return None
    r = row.iloc[0]
    odds, fin = r.get("odds_win"), r.get("finish_position")
    if odds is None or pd.isna(odds) or pd.isna(fin):
        return None

    # 確率モデルが「active の本命」に与える確率 (自分の本命ではない)
    hit = prob_preds[prob_preds["horse_id"] == bet_top["horse_id"]]
    if hit.empty:
        return None
    prob_for_bet_pick = _f(hit.iloc[0].get("win_prob"))
    prob_rank_of_bet_pick = int(prob_preds.index.get_loc(hit.index[0]))

    prob_top = prob_preds.iloc[0]
    agree = bool(prob_top["horse_id"] == bet_top["horse_id"])

    payout_map: dict[int, float] = {}
    if "payout_place" in rf.columns:
        vals = rf["payout_place"].dropna()
        if not vals.empty:
            payout_map = _parse_payout_place(vals.iloc[0])

    # 確率モデルの本命を買った場合の結果 (比較用)
    prob_row = rf[rf["horse_id"] == prob_top["horse_id"]]
    prob_pick_odds = _f(prob_row.iloc[0].get("odds_win")) if not prob_row.empty else None
    prob_pick_fin = _f(prob_row.iloc[0].get("finish_position")) if not prob_row.empty else None

    return {
        "race_id": r.get("race_id"),
        "date": r.get("date"),
        "n_runners": _f(r.get("n_runners")),
        "race_class": r.get("race_class"),
        # active (買う馬を決める側)
        "bet_odds": _f(odds),
        "bet_popularity": _f(r.get("popularity")),
        "bet_win_prob": _f(bet_top.get("win_prob")),
        # 確率モデル (確からしさを答える側)
        "prob_model_p": prob_for_bet_pick,
        "prob_model_rank": prob_rank_of_bet_pick,
        "prob_model_ev": (prob_for_bet_pick or 0.0) * (_f(odds) or 0.0),
        "models_agree": agree,
        # 確率モデル自身の本命 (単体性能の比較用)
        "prob_pick_odds": prob_pick_odds,
        "prob_pick_win_return": (
            prob_pick_odds if prob_pick_odds is not None and prob_pick_fin == 1.0 else 0.0
        ),
        # 複勝は「3 着以内に来る馬」が報われるので、当てる力の高い確率モデルに
        # 馬を選ばせた場合も測る (active の本命の複勝と直接比べられる)。
        "prob_pick_finish": prob_pick_fin,
        "prob_pick_place_return": _place_return(payout_map, prob_pick_fin),
        "prob_pick_p": _f(prob_top.get("win_prob")),
        "prob_pick_place_prob": _f(prob_top.get("place_prob")),
        # 結果 (active の本命について)
        "finish": _f(fin),
        "win_return": float(odds) if float(fin) == 1.0 else 0.0,
        "place_return": _place_return(payout_map, fin),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bet-model", type=Path, required=True)
    ap.add_argument("--prob-model", type=Path, required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    engine = make_engine(db_path())
    bet_bundle = load_model_full(args.bet_model)
    prob_bundle = load_model_full(args.prob_model)
    print(f"bet={args.bet_model}\nprob={args.prob_model}", flush=True)

    rows: list[dict] = []
    # session はループ外で保持する (履歴 GRU が zero に degrade するのを防ぐ)
    with session_scope(engine) as session:
        frame = build_training_frame(session, train_start=args.start, train_end=args.end)
        race_ids = frame["race_id"].unique()
        print(f"{len(race_ids)} races", flush=True)
        for i, rid in enumerate(race_ids):
            rf = frame[frame["race_id"] == rid]
            if len(rf) < 2:
                continue
            try:
                bet_preds = predict_race(bet_bundle, rf, session=session)
                prob_preds = predict_race(prob_bundle, rf, session=session)
            except Exception as exc:  # noqa: BLE001
                print(f"  predict failed {rid}: {exc}", flush=True)
                continue
            row = _row(rf, bet_preds, prob_preds)
            if row is not None:
                rows.append(row)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(race_ids)}", flush=True)

    d = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(args.out, index=False, encoding="utf-8")
    print(f"wrote {len(d)} races -> {args.out}", flush=True)
    print(f"  active の本命: 単勝 {d['win_return'].mean():.4f}", flush=True)
    print(f"  確率モデルの本命: 単勝 {d['prob_pick_win_return'].mean():.4f}", flush=True)
    print(f"  2 モデルが一致した割合: {d['models_agree'].mean():.3f}", flush=True)
    print(f"  複勝: active の本命 {d['place_return'].dropna().mean():.4f} / "
          f"確率モデルの本命 {d['prob_pick_place_return'].dropna().mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
