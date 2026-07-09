"""Permutation feature importance for the ACTIVE NN model (model-agnostic).

SHAP は廃止されているため、各特徴量を holdout 全体でシャッフル→指標劣化量を測る
permutation importance で「現行モデルの各特徴量の影響度」を実測する。

- 評価窓: 学習/検証で未使用の clean OOS (既定 2024-11-01〜2025-12-31)。
- 履歴 GRU テンソルはレース毎に 1 回だけ構築してキャッシュし、静的特徴量の
  permutation 時は前処理 + forward のみ回す (高速化)。
- 主要指標: top1_hit (最高スコア馬が1着の割合) / winner_in_top3 / tansho_roi。
- グループ集計: market(odds_win+popularity) / 履歴GRU / 非market静的 をまとめて落とす。

Usage:
    PYTHONPATH=src uv run python -m scripts.feature_importance_perm \
        --start 2024-11-01 --end 2025-12-31 --max-races 600 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np
import pandas as pd
import torch

from core.paths import db_path
from db.session import make_engine, session_scope
from ai.model.registry import get_active, load_model_full
from ai.inference.predict import _build_inference_history_tensors
from features.builder import (
    FEATURE_COLUMNS,
    ODDS_FEATURE_COLUMNS,
)


def _score_race(bundle, encoded_rows: pd.DataFrame, hist):
    """1 レース分の transform 済み行 + キャッシュ履歴テンソルから score 配列を返す。"""
    horse_cols = bundle.nn_horse_feature_cols or []
    race_cols = bundle.nn_race_feature_cols or []
    odds_cols = bundle.nn_odds_feature_cols or []
    n = len(encoded_rows)

    hf_np = encoded_rows[horse_cols].values.astype("float32") if horse_cols else np.zeros((n, 0), "float32")
    hf = torch.tensor(hf_np).unsqueeze(0)
    rf_np = encoded_rows[race_cols].iloc[0].values.astype("float32") if race_cols else np.zeros(0, "float32")
    rf = torch.tensor(rf_np).unsqueeze(0)
    mask = torch.ones(1, n, dtype=torch.bool)
    odds_t = None
    if odds_cols:
        odds_t = torch.tensor(encoded_rows[odds_cols].values.astype("float32")).unsqueeze(0)
    hist_seq, hist_len = hist if hist is not None else (None, None)
    with torch.no_grad():
        scores = bundle.nn_model(
            hf, rf, mask,
            history_seq=hist_seq, history_lengths=hist_len, odds_features=odds_t,
        )[0, :n].cpu().numpy()
    return scores


def _metrics(scores_by_race, races):
    """top1_hit / winner_in_top3 / tansho_roi を全レースで集計。"""
    top1 = top3 = 0
    bet = ret = 0.0
    nr = 0
    for rid, g in races:
        s = scores_by_race[rid]
        order = np.argsort(-s)  # 高スコア順
        fin = g["finish_position"].values
        odds = g["odds_win"].values
        winner_idx = np.where(fin == 1)[0]
        if len(winner_idx) == 0:
            continue
        nr += 1
        w = winner_idx[0]
        rank_of_winner = int(np.where(order == w)[0][0])
        top1 += 1 if rank_of_winner == 0 else 0
        top3 += 1 if rank_of_winner < 3 else 0
        pick = order[0]
        bet += 100.0
        if fin[pick] == 1 and not np.isnan(odds[pick]):
            ret += 100.0 * float(odds[pick])
    return {
        "top1_hit": top1 / nr,
        "winner_in_top3": top3 / nr,
        "tansho_roi": (ret / bet) if bet else float("nan"),
        "n_races": nr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-11-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--max-races", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=3, help="permutation 反復回数 (平均)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        active = get_active(session)
        if active is None:
            raise SystemExit("no active model")
        bundle = load_model_full(active)
        print(f"# active model: {active.name}")
        print(f"# history_feat_dim={bundle.nn_history_feat_dim} odds_cols={bundle.nn_odds_feature_cols}")

        from features.builder import build_training_frame
        df = build_training_frame(session, args.start, args.end)
        df = df[df["finish_position"].notna()].copy()

        # サンプリング (レース単位)
        race_ids = list(dict.fromkeys(df["race_id"].tolist()))
        if len(race_ids) > args.max_races:
            race_ids = rng.sample(race_ids, args.max_races)
        df = df[df["race_id"].isin(set(race_ids))].reset_index(drop=True)
        races = [(rid, g) for rid, g in df.groupby("race_id", sort=False)]
        print(f"# holdout: {len(races)} races, {len(df)} entries ({args.start}..{args.end})")

        # 履歴テンソルをレース毎に 1 回構築してキャッシュ
        print("# building history tensors (cached per race)...")
        hist_cache = {}
        if bundle.nn_history_feat_dim > 0:
            for rid, g in races:
                hist_cache[rid] = _build_inference_history_tensors(bundle, g, session, torch)
        else:
            hist_cache = {rid: None for rid, _ in races}

    # ---- baseline ----
    def eval_frame(frame):
        enc = bundle.nn_preprocessor.transform(frame)
        enc = enc.reset_index(drop=True)
        frame = frame.reset_index(drop=True)
        scores_by_race = {}
        loc = 0
        race_groups = []
        for rid, g in frame.groupby("race_id", sort=False):
            idx = g.index
            scores_by_race[rid] = _score_race(bundle, enc.loc[idx], hist_cache.get(rid))
            race_groups.append((rid, g))
        return _metrics(scores_by_race, race_groups)

    base = eval_frame(df)
    print(f"\n# BASELINE: top1={base['top1_hit']:.4f} top3={base['winner_in_top3']:.4f} "
          f"roi={base['tansho_roi']:.4f} (n={base['n_races']})")

    # ---- per-feature permutation ----
    feats = [c for c in FEATURE_COLUMNS if c in df.columns]
    results = []
    for f in feats:
        drops = []
        roi_drops = []
        for r in range(args.repeats):
            perm = df.copy()
            perm[f] = np_rng.permutation(perm[f].values)
            m = eval_frame(perm)
            drops.append(base["top1_hit"] - m["top1_hit"])
            roi_drops.append(base["tansho_roi"] - m["tansho_roi"])
        results.append({
            "feature": f,
            "top1_drop": float(np.mean(drops)),
            "roi_drop": float(np.mean(roi_drops)),
        })
        print(f"  {f:32s} top1_drop={np.mean(drops):+.4f}  roi_drop={np.mean(roi_drops):+.4f}")

    # ---- group ablations ----
    def perm_cols(cols):
        drops = []
        for r in range(args.repeats):
            perm = df.copy()
            for c in cols:
                if c in perm.columns:
                    perm[c] = np_rng.permutation(perm[c].values)
            m = eval_frame(perm)
            drops.append((base["top1_hit"] - m["top1_hit"], base["tansho_roi"] - m["tansho_roi"]))
        a = np.array(drops)
        return float(a[:, 0].mean()), float(a[:, 1].mean())

    non_market = [c for c in feats if c not in ODDS_FEATURE_COLUMNS]
    groups = {
        "market(odds_win+popularity)": ODDS_FEATURE_COLUMNS,
        "all_non_market_static": non_market,
    }
    print("\n# GROUP ablations (permute whole block):")
    group_res = {}
    for name, cols in groups.items():
        t, roi = perm_cols(cols)
        group_res[name] = {"top1_drop": t, "roi_drop": roi}
        print(f"  {name:34s} top1_drop={t:+.4f}  roi_drop={roi:+.4f}")

    # history GRU block: zero out history tensors
    if bundle.nn_history_feat_dim > 0:
        saved = dict(hist_cache)
        for rid in hist_cache:
            hs, hl = saved[rid]
            hist_cache[rid] = (torch.zeros_like(hs), torch.zeros_like(hl))
        m = eval_frame(df)
        hist_cache.update(saved)
        group_res["history_GRU(zeroed)"] = {
            "top1_drop": base["top1_hit"] - m["top1_hit"],
            "roi_drop": base["tansho_roi"] - m["tansho_roi"],
        }
        print(f"  {'history_GRU(zeroed)':34s} top1_drop={base['top1_hit']-m['top1_hit']:+.4f}  "
              f"roi_drop={base['tansho_roi']-m['tansho_roi']:+.4f}")

    out = {"model": active.name, "baseline": base,
           "per_feature": sorted(results, key=lambda x: -x["top1_drop"]),
           "groups": group_res}
    with open("/tmp/feature_importance.json", "w") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("\n# wrote /tmp/feature_importance.json")

    print("\n# === RANKED by top1_drop (top 20) ===")
    for r in sorted(results, key=lambda x: -x["top1_drop"])[:20]:
        print(f"  {r['feature']:32s} {r['top1_drop']:+.4f}")


if __name__ == "__main__":
    main()
