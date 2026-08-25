"""レースごとの「賭ける前に分かる情報」と「実際のリターン」を 1 行ずつ書き出す。

戦略 (どのレースを買うか / いくら賭けるか) を考えるための素材。モデルは固定で、
推論を 1 回だけ回して CSV に落とし、以降の分析は CSV 上で何度でもやり直す。

出力 1 行 = 1 レース。列は 3 群:

  * 賭ける前に観測できる signal (クラス・頭数・オッズ・モデルの確信度・市場との一致)
  * 実際の結果 (本命の着順・単勝/複勝リターン)
  * 参照用ベースライン (1 番人気の単勝/複勝リターン)

Usage:
  PYTHONIOENCODING=utf-8 PYTHONPATH=src python -m scripts.strategy_signal_dump \
      --start 2024-11-02 --end 2026-05-31 --out ../data/analysis/strategy_signals.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ai.evaluation.backtest import _parse_payout_place
from ai.inference.predict import predict_race
from ai.model.registry import get_active, load_model_full
from core.paths import db_path
from db.session import make_engine, session_scope
from features.builder import build_training_frame
from features.race_info import race_info_coverage


def _first(frame: pd.DataFrame, col: str):
    if col not in frame.columns:
        return None
    vals = frame[col].dropna()
    return vals.iloc[0] if not vals.empty else None


def _place_return(payout_map: dict[int, float], finish) -> float | None:
    """複勝の払戻倍率。3 着以内でなければ 0.0、払戻が取れなければ None。"""
    if not payout_map or finish is None or pd.isna(finish):
        return None
    fin = int(finish)
    if fin in payout_map:
        return payout_map[fin] / 100.0
    return 0.0 if fin > 3 else None


def _row_for_race(rf: pd.DataFrame, preds: pd.DataFrame) -> dict | None:
    """1 レース分の signal + 結果。必要な情報が欠ければ None。"""
    top = preds.iloc[0]
    row = rf[rf["horse_id"] == top["horse_id"]]
    if row.empty:
        return None
    r = row.iloc[0]

    odds = r.get("odds_win")
    fin = r.get("finish_position")
    if odds is None or pd.isna(odds) or pd.isna(fin):
        return None

    payout_map: dict[int, float] = {}
    raw = _first(rf, "payout_place")
    if raw is not None:
        payout_map = _parse_payout_place(raw)

    probs = preds["win_prob"].to_numpy(dtype=float)
    probs = probs[np.isfinite(probs)]
    entropy = float(-(probs * np.log(np.clip(probs, 1e-12, None))).sum()) if probs.size else float("nan")
    scores = preds["score"].to_numpy(dtype=float)

    # 市場: 1 番人気の馬 (popularity==1)。同着人気があれば最小オッズを採る。
    fav = rf.sort_values("odds_win").iloc[0] if "odds_win" in rf.columns else None
    cov = race_info_coverage(rf)

    def _f(v):
        try:
            f = float(v)
            return f if math.isfinite(f) else None
        except (TypeError, ValueError):
            return None

    # ── 市場との突き合わせ ──
    # モデルと市場が「どこで・どれだけ食い違っているか」を測る列。EV (p̂×odds) は
    # 単調ですらなかったので、突き合わせ方を 1 本に決め打ちせず素材として複数残す。
    market = rf[["horse_id", "odds_win"]].dropna().copy()
    market["implied"] = 1.0 / market["odds_win"]
    overround = float(market["implied"].sum()) if not market.empty else float("nan")
    log_odds_std = float(np.log(market["odds_win"]).std()) if len(market) > 1 else float("nan")
    n_short = int((market["odds_win"] < 5.0).sum())

    # 市場の 1 番人気を、モデルが何位に置いているか (乖離の向きと大きさ)
    fav_model_rank = None
    fav_model_prob = None
    if fav is not None:
        hit = preds.index[preds["horse_id"] == fav.get("horse_id")]
        if len(hit):
            pos = int(preds.index.get_loc(hit[0]))
            fav_model_rank = pos
            fav_model_prob = _f(preds.iloc[pos].get("win_prob"))

    # モデルの 2・3 番手のオッズ (本命だけでなく上位の並び方を見る)
    def _odds_of(rank: int) -> float | None:
        if len(preds) <= rank:
            return None
        row2 = rf[rf["horse_id"] == preds.iloc[rank]["horse_id"]]
        return _f(row2.iloc[0].get("odds_win")) if not row2.empty else None

    implied_top1 = 1.0 / float(odds) if float(odds) > 0 else None

    return {
        # ── 識別 ──
        "race_id": r.get("race_id"),
        "date": r.get("date"),
        # ── 賭ける前に分かる signal ──
        "race_class": r.get("race_class"),
        "course": r.get("course"),
        "surface": r.get("surface"),
        "track_condition": r.get("track_condition"),
        "distance": _f(r.get("distance")),
        "n_runners": _f(r.get("n_runners")),
        "top1_odds": _f(odds),
        "top1_popularity": _f(r.get("popularity")),
        "top1_win_prob": _f(top.get("win_prob")),
        "top1_place_prob": _f(top.get("place_prob")),
        "top1_ev": (_f(top.get("win_prob")) or 0.0) * (_f(odds) or 0.0),
        # モデルの確信度: 1 位と 2 位の開き
        "prob_margin": _f(probs[0] - probs[1]) if probs.size > 1 else None,
        "score_margin": _f(scores[0] - scores[1]) if scores.size > 1 else None,
        "win_prob_entropy": _f(entropy),
        # 市場との一致 (本命が 1 番人気か / 市場の 1 番人気オッズ)
        "top1_is_favorite": bool(_f(r.get("popularity")) == 1.0),
        "fav_odds": _f(fav.get("odds_win")) if fav is not None else None,
        "debut_ratio": cov.debut_ratio,
        "mean_starts": cov.mean_starts,
        # ── 市場との乖離 ──
        "implied_top1": implied_top1,
        # モデル確率 ÷ 市場実装確率。1 より大きいほどモデルが市場より強気
        "prob_vs_market": (
            (_f(top.get("win_prob")) or 0.0) / implied_top1 if implied_top1 else None
        ),
        "fav_model_rank": fav_model_rank,
        "fav_model_prob": fav_model_prob,
        "top2_odds": _odds_of(1),
        "top3_odds": _odds_of(2),
        "overround": _f(overround),
        "log_odds_std": _f(log_odds_std),
        "n_short_odds": n_short,
        "score_std": _f(scores.std()) if scores.size > 1 else None,
        "top1_recent_starts": _f(r.get("recent_n_starts")),
        "top1_days_since_last": _f(r.get("days_since_last_race")),
        # ── 結果 ──
        "top1_finish": _f(fin),
        "win_return": float(odds) if float(fin) == 1.0 else 0.0,
        "place_return": _place_return(payout_map, fin),
        # ── ベースライン (1 番人気ベタ買い) ──
        "fav_win_return": (
            float(fav.get("odds_win")) if fav is not None and _f(fav.get("finish_position")) == 1.0 else 0.0
        ) if fav is not None else None,
        "fav_place_return": (
            _place_return(payout_map, fav.get("finish_position")) if fav is not None else None
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=None,
                    help="モデルディレクトリ。未指定なら active モデル")
    args = ap.parse_args()

    engine = make_engine(db_path())
    model_path = args.model
    if model_path is None:
        with session_scope(engine) as s0:
            model_path = get_active(s0)
    if model_path is None:
        raise SystemExit("no active model")
    bundle = load_model_full(model_path)
    print(f"model: {model_path}", flush=True)

    rows: list[dict] = []
    # session はループの外で開いたまま保持する。predict_race に渡さないと
    # 履歴 GRU が zero に degrade して単勝回収率が 0.9 台 → 0.8 台に落ちる。
    with session_scope(engine) as session:
        frame = build_training_frame(session, train_start=args.start, train_end=args.end)
        race_ids = frame["race_id"].unique()
        print(f"{len(race_ids)} races", flush=True)
        for i, rid in enumerate(race_ids):
            rf = frame[frame["race_id"] == rid]
            if len(rf) < 2:
                continue
            try:
                preds = predict_race(bundle, rf, session=session)
            except Exception as exc:  # noqa: BLE001
                print(f"  predict failed {rid}: {exc}", flush=True)
                continue
            row = _row_for_race(rf, preds)
            if row is not None:
                rows.append(row)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(race_ids)}", flush=True)

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8")
    print(f"wrote {len(out)} races -> {args.out}", flush=True)
    print(
        f"overall: win={out['win_return'].mean():.4f} "
        f"place={out['place_return'].dropna().mean():.4f} "
        f"fav_win={out['fav_win_return'].dropna().mean():.4f} "
        f"fav_place={out['fav_place_return'].dropna().mean():.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
