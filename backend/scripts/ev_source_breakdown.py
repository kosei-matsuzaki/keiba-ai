"""EV source breakdown — なぜ bet が極端に少ないのかを診断する。

simulation.py の選択ループをそのまま再現しつつ、min_ev フィルタを **掛ける前** の
候補を全部集め、(bet_type, est_odds_source) 別に EV 分布と各 min_ev 閾値での
通過数を出す。これで「どの source の EV が controlled な 1-takeout(≈0.8) に張り付き、
どこで 1.10 によって切られているか」を目視できる。

2 つのビューを出力する:
  View A (universe): predict_race_with_combinations が返す全 combo の EV 分布。
      市場効率性の構造（confirmed/implied は EV≈1-takeout に集中）を直接見る。
  View B (funnel):   recommend_for_race が実際に考慮する候補（top-N box/nagashi/
      formation, dedup 済み）の EV 分布。simulation の bet 発火に直結する母数。

Usage:
  PYTHONPATH=src python -m scripts.ev_source_breakdown \
      --model data/models/<ts> --start 2026-01-01 --end 2026-12-31 \
      [--strategy balanced]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from ai.bet_odds import compute_race_odds_with_sources
from ai.bet_strategy import recommend_for_race
from ai.predict import predict_race, predict_race_with_combinations
from ai.registry import load_model_full
from ai.simulation import DEFAULT_BET_TYPES, STRATEGY_PRESETS
from core.paths import db_path
from db.odds_db import init_odds_db, make_odds_engine
from db.session import make_engine, session_scope
from features.builder import build_training_frame

# 各 source / bet_type で EV がどの閾値を超えるかを見る列。
_THRESHOLDS = [1.00, 1.05, 1.10, 1.20, 1.30, 1.50]


def _percentiles(vals: list[float]) -> tuple[float, float, float, float, float]:
    a = np.asarray(vals, dtype=float)
    return (
        float(np.percentile(a, 50)),
        float(np.percentile(a, 90)),
        float(np.percentile(a, 99)),
        float(a.max()),
        float(a.mean()),
    )


def _print_breakdown(title: str, by_key: dict[tuple[str, str], list[float]]) -> None:
    print(f"\n=== {title} ===")
    if not by_key:
        print("  (no candidates)")
        return
    thr_hdr = "".join(f"  >={t:.2f}" for t in _THRESHOLDS)
    print(
        f"{'bet_type':>8} {'source':>10} {'n':>7} "
        f"{'med':>6} {'p90':>6} {'p99':>6} {'max':>7} {'mean':>6}{thr_hdr}"
    )
    # source ごとに合算した行も出すため、(bet_type, source) と (ALL, source) を集計
    src_totals: dict[str, list[float]] = defaultdict(list)
    for (bt, src), vals in sorted(by_key.items()):
        src_totals[src].extend(vals)
        med, p90, p99, mx, mean = _percentiles(vals)
        thr = "".join(
            f"{sum(1 for v in vals if v >= t):>7}" for t in _THRESHOLDS
        )
        print(
            f"{bt:>8} {src:>10} {len(vals):>7} "
            f"{med:>6.2f} {p90:>6.2f} {p99:>6.2f} {mx:>7.2f} {mean:>6.2f}{thr}"
        )
    print("  " + "-" * 100)
    for src, vals in sorted(src_totals.items()):
        med, p90, p99, mx, mean = _percentiles(vals)
        thr = "".join(
            f"{sum(1 for v in vals if v >= t):>7}" for t in _THRESHOLDS
        )
        print(
            f"{'ALL':>8} {src:>10} {len(vals):>7} "
            f"{med:>6.2f} {p90:>6.2f} {p99:>6.2f} {mx:>7.2f} {mean:>6.2f}{thr}"
        )


def run(model_path: Path, start: str, end: str, strategy: str) -> None:
    preset = STRATEGY_PRESETS[strategy]
    min_ev = preset["min_ev"]
    bundle = load_model_full(model_path)

    engine = make_engine(db_path())
    odds_engine = make_odds_engine()
    init_odds_db(odds_engine)
    odds_session = Session(bind=odds_engine)

    # (bet_type, source) -> list[ev]
    universe: dict[tuple[str, str], list[float]] = defaultdict(list)
    funnel: dict[tuple[str, str], list[float]] = defaultdict(list)
    # bet 発火カウント (funnel の stake>0)
    fired: dict[tuple[str, str], int] = defaultdict(int)

    n_races = 0
    n_with_any_candidate = 0
    n_with_pass = 0  # min_ev を 1 件以上通過したレース数

    with session_scope(engine) as session:
        frame = build_training_frame(session, train_start=start, train_end=end)
        race_ids = list(frame["race_id"].unique())
        print(f"window {start}..{end}: {len(race_ids)} races (strategy={strategy}, min_ev={min_ev})")

        for race_id in race_ids:
            rf = frame[frame["race_id"] == race_id]
            if len(rf) < 2:
                continue
            n_races += 1

            try:
                preds = predict_race(bundle, rf)
            except Exception:  # noqa: BLE001
                continue
            pp_map = dict(zip(rf["horse_id"].values, rf["post_position"].values, strict=True))
            preds["post_position"] = preds["horse_id"].map(pp_map)

            race_odds, race_odds_sources = compute_race_odds_with_sources(
                session, race_id, odds_session=odds_session
            )
            try:
                combos_by_type = predict_race_with_combinations(
                    bundle, rf, session=session,
                    race_odds=race_odds, race_odds_sources=race_odds_sources,
                )
            except Exception:  # noqa: BLE001
                continue

            # View A: full universe (all combos, EV not None)
            for bt, combos in combos_by_type.items():
                for c in combos:
                    if c.ev is not None:
                        universe[(bt, c.est_odds_source)].append(c.ev)

            # View B: funnel — recommend_for_race が実際に考慮する候補。
            # min_ev フィルタは掛けずに渡し、戻り値（keep_zero_stake で全件）の
            # ev/source/stake を見る。これで「考慮 → min_ev 通過 → stake付与」の
            # 各段が source 別に分かる。
            rec = recommend_for_race(
                predictions=preds,
                combinations_by_type=combos_by_type,
                race_id=race_id,
                bankroll=100_000,
                kelly_fraction=preset["kelly_fraction"],
                max_stake_per_race_pct=0.05,
                top_n_horses=3,
                enabled_bet_types=list(DEFAULT_BET_TYPES),
            )
            race_has_candidate = False
            race_has_pass = False
            for c in rec.candidates:
                if c.ev is None:
                    continue
                funnel[(c.bet_type, c.est_odds_source)].append(c.ev)
                race_has_candidate = True
                if c.ev >= min_ev:
                    race_has_pass = True
                if c.stake > 0:
                    fired[(c.bet_type, c.est_odds_source)] += 1
            if race_has_candidate:
                n_with_any_candidate += 1
            if race_has_pass:
                n_with_pass += 1

    odds_session.close()

    print(f"\nprocessed {n_races} races")
    print(f"  races with >=1 considered candidate : {n_with_any_candidate}")
    print(f"  races with >=1 candidate ev>=min_ev : {n_with_pass}")

    _print_breakdown("View A — full combo universe (EV by bet_type x source)", universe)
    _print_breakdown("View B — recommend funnel (considered candidates)", funnel)

    print("\n=== bets fired (stake>0) by bet_type x source ===")
    if not fired:
        print("  (none)")
    else:
        total = 0
        for (bt, src), n in sorted(fired.items()):
            print(f"  {bt:>8} {src:>10} {n:>5}")
            total += n
        print(f"  {'TOTAL':>8} {'':>10} {total:>5}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--strategy", default="balanced", choices=list(STRATEGY_PRESETS))
    args = ap.parse_args()
    run(args.model, args.start, args.end, args.strategy)


if __name__ == "__main__":
    main()
