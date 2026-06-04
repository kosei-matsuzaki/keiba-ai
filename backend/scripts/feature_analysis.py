"""Audit the 51 input features: importance, missingness, variance, redundancy.

Outputs, per feature:
  - gain importance from the with-odds lambdarank booster AND the no-odds one
    (so we see what carries signal once odds are removed),
  - % missing (NaN) and # distinct on the training frame,
  - membership in highly-correlated clusters (|r|>=0.9 → redundancy candidates).

Grouped by domain. Read-only; uses the cached training frame + existing models.
"""

from __future__ import annotations

import lightgbm as lgb

from core.paths import db_path
from db.session import make_engine, session_scope
from features.builder import CATEGORICAL_FEATURES, FEATURE_COLUMNS, build_training_frame

WITH_ODDS = "/mnt/c/Users/PC_User/private_production/keiba-ai/data/models/20260601-141858/model.txt"
NO_ODDS = "/mnt/c/Users/PC_User/private_production/keiba-ai/data/models/20260601-170042/model.txt"

GROUPS: dict[str, list[str]] = {
    "ODDS(市場)": ["odds_win", "popularity", "log_odds_win", "odds_win_rank", "odds_win_diff_from_favorite"],
    "RACE文脈": ["distance", "n_runners", "post_position", "post_position_ratio", "surface",
                 "course", "weather", "track_condition", "race_class"],
    "馬体/斤量": ["age", "sex", "horse_weight", "horse_weight_diff", "horse_weight_pct",
                  "weight_carried_pct", "weight_carried_diff"],
    "近走フォーム": ["recent_avg_finish", "recent_n_starts", "recent_avg_agari_3f", "recent_finish_1",
                    "recent_finish_2", "recent_finish_3", "recent_avg_margin", "recent_avg_finish_time_norm",
                    "recent_best_margin_in_top3", "recent_avg_position_change", "recent_passing_volatility",
                    "recent_early_position_ratio", "recent_late_position_ratio", "recent_best_agari_3f",
                    "days_since_last_race"],
    "クラス/適性": ["recent_avg_class_weight", "high_class_starts", "high_class_places", "class_change",
                    "starts_same_distance", "starts_same_course", "wins_same_course"],
    "騎手/厩舎": ["jockey_recent_win_rate", "jockey_recent_place_rate", "jockey_course_place_rate",
                  "trainer_course_place_rate", "jockey_recent_win_rate_vs_field", "course_place_rate_vs_field"],
    "血統": ["sire_progeny_win_rate", "dam_progeny_win_rate"],
}


def gain_pct(model_file: str) -> dict[str, float]:
    b = lgb.Booster(model_file=model_file)
    names = b.feature_name()
    gains = b.feature_importance(importance_type="gain")
    tot = gains.sum() or 1.0
    return {n: 100.0 * g / tot for n, g in zip(names, gains, strict=True)}


def main() -> None:
    wo = gain_pct(WITH_ODDS)
    no = gain_pct(NO_ODDS)

    engine = make_engine(db_path())
    with session_scope(engine) as s:
        frame = build_training_frame(s, train_start="2015-01-01", train_end="2024-12-31")
    n = len(frame)

    # numeric correlation clusters (redundancy)
    num_cols = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES and c in frame.columns]
    corr = frame[num_cols].corr().abs()
    redundant: dict[str, list[str]] = {}
    for i, a in enumerate(num_cols):
        for b_ in num_cols[i + 1:]:
            r = corr.loc[a, b_]
            if r >= 0.9:
                redundant.setdefault(a, []).append(f"{b_}({r:.2f})")

    print(f"frame rows={n}  (gain% = LightGBM split-gain share)\n")
    print(f"{'feature':<32}{'odds-gain%':>10}{'noOdds%':>9}{'miss%':>7}{'ndist':>7}")
    for grp, feats in GROUPS.items():
        print(f"\n── {grp} ──")
        # sort within group by no-odds importance (what matters without market)
        for f in sorted(feats, key=lambda x: -no.get(x, 0.0)):
            if f not in frame.columns:
                print(f"{f:<32}{'(absent from frame)':>33}")
                continue
            miss = 100.0 * frame[f].isna().mean()
            ndist = frame[f].nunique(dropna=True)
            print(f"{f:<32}{wo.get(f, 0.0):>10.2f}{no.get(f, 0.0):>9.2f}{miss:>7.1f}{ndist:>7}")

    print("\n=== 冗長候補 (|相関| >= 0.9) ===")
    if redundant:
        for a, partners in redundant.items():
            print(f"  {a}  ~  {', '.join(partners)}")
    else:
        print("  none")

    # summary: how concentrated is the signal
    wo_sorted = sorted(wo.values(), reverse=True)
    no_sorted = sorted(no.values(), reverse=True)
    print("\n=== 信号の集中度 ===")
    print(f"  odds入り: top-2 features = {sum(wo_sorted[:2]):.1f}% of gain, top-5 = {sum(wo_sorted[:5]):.1f}%")
    print(f"  no-odds : top-5 features = {sum(no_sorted[:5]):.1f}% of gain, top-10 = {sum(no_sorted[:10]):.1f}%")
    near_zero = [f for f in FEATURE_COLUMNS if no.get(f, 0.0) < 0.5 and wo.get(f, 0.0) < 0.5]
    print(f"  両モデルで gain<0.5% の希薄特徴 ({len(near_zero)}): {', '.join(near_zero) if near_zero else 'none'}")


if __name__ == "__main__":
    main()
