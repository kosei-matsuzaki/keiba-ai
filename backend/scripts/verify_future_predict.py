"""One-shot E2E check: predict a future (weekend) race with the active model.

live_odds 撤去後も「出馬表取込済みの未来レースを AI 予測できる」ことを確認する。
推論経路 (build_inference_frame → predict_race → compute_race_odds_with_sources) を
HTTP を介さず in-process で叩き、勝率ランキングと odds_source を表示する。
"""

from __future__ import annotations

import sys
import time

from ai.bet_odds import compute_race_odds_with_sources
from ai.predict import predict_race
from ai.registry import get_active, load_model_full
from core.paths import db_path
from db.session import make_engine, session_scope
from features.builder import build_inference_frame

race_id = sys.argv[1] if len(sys.argv) > 1 else "202605021101"
engine = make_engine(db_path())

with session_scope(engine) as session:
    active = get_active(session)
    print(f"active model: {active}")

    t0 = time.time()
    frame = build_inference_frame(session, race_id)
    print(f"build_inference_frame: {len(frame)} rows in {time.time()-t0:.1f}s")

    bundle = load_model_full(active)
    t1 = time.time()
    preds = predict_race(bundle, frame)
    print(f"predict_race: {time.time()-t1:.1f}s  model_type={bundle.model_type}")

    preds = preds.sort_values("win_prob", ascending=False).reset_index(drop=True)
    merged = preds.merge(
        frame[["horse_id", "post_position"]], on="horse_id", how="left"
    )
    print("\n=== AI 予測 (win_prob 降順 top8) ===")
    print(f"{'rank':>4} {'枠':>3} {'horse_id':>12} {'win_prob':>9} {'place_prob':>10}")
    for i, row in merged.head(8).iterrows():
        print(
            f"{i+1:>4} {int(row['post_position']) if row['post_position']==row['post_position'] else '-':>3} "
            f"{str(row['horse_id']):>12} {row['win_prob']:>9.3f} {row['place_prob']:>10.3f}"
        )

    odds, sources = compute_race_odds_with_sources(session, race_id)
    bt = sorted(odds.keys())
    print(f"\n=== compute_race_odds_with_sources ===\nbet_types: {bt}")
    print(f"単勝 combos: {len(odds.get('単勝', {}))}  | implied連系: "
          f"{sum(len(v) for k,v in odds.items() if k!='単勝')}")
