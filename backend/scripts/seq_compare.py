"""Subset-matched top1_hit comparison for the option-C experiment.

The sequence model only scores races whose winner has prior history (it can't
encode a debut horse), so it evaluates on a SUBSET of OOS races. A naive
comparison against the GBDT's full-OOS top1_hit is unfair: debut-winner races
(excluded by the seq model) are hard for everyone and drag the GBDT average
down. This recomputes the no-odds GBDT and the favorite top1_hit on the EXACT
same race set the sequence model used (criterion imported from seq_experiment),
so seq vs GBDT vs favorite are finally apples-to-apples.

Run with KEIBA_EXCLUDE_ODDS_FEATURES=1 (no-odds GBDT).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sqlalchemy import select

from ai.predict import predict_race
from ai.registry import load_model_full
from core.paths import db_path
from db.models.entry import Entry
from db.session import make_engine, session_scope
from features.builder import build_training_frame
from seq_experiment import group_by_race, load_history_and_samples  # same dir (scripts/)

OOS_START, OOS_END = "2025-10-01", "2026-04-30"
NO_ODDS_GBDT = "/mnt/c/Users/PC_User/private_production/keiba-ai/data/models/20260601-170042"


def main() -> None:
    # 1) seq-evaluable OOS race set — identical criterion to the seq experiment.
    data = load_history_and_samples(str(db_path()), None)
    seq_races = {race[0]["race_id"] for race in group_by_race(data["oos"])}
    print(f"seq-evaluable OOS races: {len(seq_races)}")

    # 2) per-race winner pp + favorite pp (lowest odds_win) from the DB.
    from db.models.race import Race
    engine = make_engine(db_path())
    winner_pp: dict[str, int] = {}
    fav_pp: dict[str, int] = {}
    with session_scope(engine) as s:
        rows = s.execute(
            select(Entry.race_id, Entry.post_position, Entry.finish_position, Entry.odds_win)
            .join(Race, Race.race_id == Entry.race_id)
            .where(Race.date >= OOS_START, Race.date <= OOS_END)
        ).all()
    best_odds: dict[str, float] = {}
    for rid, pp, fin, odds in rows:
        if fin == 1 and pp is not None:
            winner_pp[rid] = pp
        if odds is not None and pp is not None and odds < best_odds.get(rid, 1e9):
            best_odds[rid] = odds
            fav_pp[rid] = pp

    # 3) GBDT predictions per race (top-1 pick).
    bundle = load_model_full(Path(NO_ODDS_GBDT))
    gbdt_pick: dict[str, int] = {}
    with session_scope(engine) as s:
        frame = build_training_frame(s, train_start=OOS_START, train_end=OOS_END)
        for rid, race_frame in frame.groupby("race_id"):
            if len(race_frame) < 2:
                continue
            try:
                preds = predict_race(bundle, race_frame)
            except Exception:  # noqa: BLE001
                continue
            top = preds.sort_values("win_prob", ascending=False).iloc[0]
            pp_map = dict(zip(race_frame["horse_id"], race_frame["post_position"], strict=True))
            gbdt_pick[rid] = pp_map.get(top["horse_id"])

    # 4) top1_hit on ALL OOS races vs the seq-evaluable subset.
    def hit_rate(pick: dict[str, int], race_ids) -> tuple[float, int]:
        vals = [1.0 if pick.get(r) == winner_pp.get(r) else 0.0
                for r in race_ids if r in winner_pp and r in pick]
        return (float(np.mean(vals)) if vals else float("nan"), len(vals))

    all_oos = set(winner_pp)
    print("\n=== top1_hit (model's #1 pick won) ===")
    print(f"{'set':>16} {'GBDT(no-odds)':>14} {'favorite':>10} {'n':>6}")
    for label, rset in [("ALL OOS", all_oos), ("seq-evaluable", seq_races & all_oos)]:
        g, ng = hit_rate(gbdt_pick, rset)
        f, nf = hit_rate(fav_pp, rset)
        print(f"{label:>16} {g:>14.4f} {f:>10.4f} {ng:>6}")
    print("\n  sequence model (same seq-evaluable set): top1_hit=0.2901  races=1772")


if __name__ == "__main__":
    main()
