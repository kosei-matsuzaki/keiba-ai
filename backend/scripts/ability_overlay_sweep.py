"""Ability-overlay sweep — 「市場が過小評価している馬」を突く戦略の検証 (案A)。

docs/ai-model.md の「実験ノブと A/B 知見」は、特徴量・損失をどういじっても with-odds の
本番 ROI は改善しない (市場効率の壁) と結論し、**未検証の唯一の道**として
「odds を入力しない ability モデルの予測が odds と乖離する overlay を突く *戦略側*」を
挙げている。arch-3 は ability エンコーダに odds が入らず head で concat するだけなので、
``odds_features=None`` で forward すれば **再学習なしで ability-only スコアが取れる**
(標準化済みオッズの平均 = 市場が無意見だった場合の反実仮想スコア)。

本スクリプトは 1 パスで per-horse テーブルを作り、後段のスイープを純 pandas でやる:

  1. calib 窓 (= 学習時の valid) で ability スコアの温度 T を **NLL 最小化**で当てる
     (既存 TemperatureScaler は payback のグリッド探索なので、ここで使うと
      校正に ROI 追求が混入する。overlay の検証では確率を正直にしたい)
  2. test 窓で p_ability・市場内包確率 q・overlay 比 r = p/q・EV = p·o を出す
  3. EV 閾値 / overlay 比 / 人気帯でスイープし、単勝 payback を race 単位 bootstrap CI 付きで出す
  4. 比較対象として本番ルール (with-odds の EV>閾値) と 1 番人気ベタ買いも同じレース集合で出す

Usage:
  PYTHONIOENCODING=utf-8 PYTHONPATH=src python -m scripts.ability_overlay_sweep \\
      --calib-start 2024-05-04 --calib-end 2024-10-27 \\
      --start 2024-11-02 --end 2026-05-31 \\
      --out data/reports/ability_overlay.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ai.inference.predict import _build_inference_history_tensors
from ai.model.registry import get_active, load_model_full
from core.paths import db_path
from db.session import make_engine, session_scope
from features.builder import build_training_frame

EPS = 1e-12


# ---------------------------------------------------------------------------
# per-horse テーブルの構築
# ---------------------------------------------------------------------------

def _race_scores(bundle, frame: pd.DataFrame, session) -> tuple[np.ndarray, np.ndarray]:
    """1 レースを 1 回だけ前処理・履歴構築して (本番スコア, ability スコア) を返す。

    ``predict_race`` を 2 回呼ぶと履歴テンソルの構築 (DB 参照) が二重に走るので、
    ここだけ ``_predict_race_nn`` の中身を最小限なぞって forward を 2 回にする。
    """
    horse_cols = bundle.nn_horse_feature_cols or []
    race_cols = bundle.nn_race_feature_cols or []
    encoded = bundle.nn_preprocessor.transform(frame)
    n = len(encoded)

    hf = torch.tensor(
        encoded[horse_cols].values.astype("float32") if horse_cols
        else np.zeros((n, 0), dtype="float32"),
        dtype=torch.float32,
    ).unsqueeze(0)
    rf = torch.tensor(
        encoded[race_cols].iloc[0].values.astype("float32") if race_cols
        else np.zeros(0, dtype="float32"),
        dtype=torch.float32,
    ).unsqueeze(0)
    mask = torch.ones(1, n, dtype=torch.bool)

    odds_cols = bundle.nn_odds_feature_cols or []
    odds_t = None
    if odds_cols:
        odds_t = torch.tensor(
            encoded[odds_cols].values.astype("float32"), dtype=torch.float32
        ).unsqueeze(0)

    hist_seq = hist_len = None
    if bundle.nn_history_feat_dim > 0 and session is not None and "date" in frame.columns:
        hist_seq, hist_len = _build_inference_history_tensors(bundle, frame, session, torch)

    model = bundle.nn_model
    with torch.no_grad():
        prod = model(
            hf, rf, mask, history_seq=hist_seq, history_lengths=hist_len,
            odds_features=odds_t,
        )[0, :n].cpu().numpy()
        # odds_features=None → head は標準化平均 (=0) を食う = 市場が無意見の反実仮想
        abil = model(
            hf, rf, mask, history_seq=hist_seq, history_lengths=hist_len,
            odds_features=None,
        )[0, :n].cpu().numpy()
    return prod, abil


def collect(bundle, session, start: str, end: str, label: str) -> pd.DataFrame:
    frame = build_training_frame(session, train_start=start, train_end=end)
    if frame.empty:
        return pd.DataFrame()
    print(f"[{label}] {len(frame)} rows / {frame['race_id'].nunique()} races", flush=True)

    rows: list[pd.DataFrame] = []
    race_ids = frame["race_id"].unique()
    for i, rid in enumerate(race_ids):
        rf = frame[frame["race_id"] == rid]
        if len(rf) < 2:
            continue
        try:
            prod, abil = _race_scores(bundle, rf, session)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {rid}: {exc}", flush=True)
            continue
        rows.append(
            pd.DataFrame({
                "race_id": rid,
                "horse_id": rf["horse_id"].values,
                "score_prod": prod,
                "score_abil": abil,
                "odds_win": rf["odds_win"].values,
                "popularity": rf["popularity"].values if "popularity" in rf else np.nan,
                "finish_position": rf["finish_position"].values,
            })
        )
        if (i + 1) % 500 == 0:
            print(f"  [{label}] {i + 1}/{len(race_ids)}", flush=True)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# 温度 (NLL 最小化)
# ---------------------------------------------------------------------------

def _softmax_by_race(df: pd.DataFrame, col: str, T: float) -> np.ndarray:
    z = df[col].values / T
    out = np.empty(len(df), dtype=float)
    for _, idx in df.groupby("race_id", sort=False).indices.items():
        zi = z[idx]
        e = np.exp(zi - zi.max())
        out[idx] = e / e.sum()
    return out


def fit_temperature_nll(df: pd.DataFrame, col: str) -> tuple[float, float]:
    """勝ち馬の負の対数尤度を最小化する T を grid で選ぶ。(T, NLL) を返す。"""
    won = (df["finish_position"] == 1).values
    best_T, best_nll = 1.0, float("inf")
    for T in np.concatenate([np.arange(0.2, 2.01, 0.05), np.arange(2.2, 6.01, 0.2)]):
        p = _softmax_by_race(df, col, float(T))
        nll = -np.log(np.clip(p[won], EPS, None)).mean()
        if nll < best_nll:
            best_T, best_nll = float(T), float(nll)
    return best_T, best_nll


# ---------------------------------------------------------------------------
# 評価
# ---------------------------------------------------------------------------

def _payback(bets: pd.DataFrame) -> dict:
    """bets: 1 行 = 1 点 (100 円固定)。回収率 = 払戻合計 / 賭け金合計。"""
    n = len(bets)
    if n == 0:
        return {"n_bets": 0, "n_races": 0, "hit": None, "payback": None}
    won = bets["won"].values
    payout = np.where(won, bets["odds_win"].values, 0.0).sum()
    return {
        "n_bets": int(n),
        "n_races": int(bets["race_id"].nunique()),
        "hit": float(won.mean()),
        "payback": float(payout / n),
    }


def _bootstrap_payback(bets: pd.DataFrame, all_races: np.ndarray, iters: int, seed: int) -> tuple:
    """race 単位リサンプルで回収率の 95% CI。賭けの無いレースも母集団に含める。"""
    if iters <= 0 or bets.empty:
        return (None, None)
    rng = np.random.default_rng(seed)
    by_race = bets.groupby("race_id").apply(
        lambda g: (len(g), float(np.where(g["won"], g["odds_win"], 0.0).sum())),
        include_groups=False,
    )
    stake_map = {r: v[0] for r, v in by_race.items()}
    pay_map = {r: v[1] for r, v in by_race.items()}
    stakes = np.array([stake_map.get(r, 0) for r in all_races], dtype=float)
    pays = np.array([pay_map.get(r, 0.0) for r in all_races], dtype=float)

    out = []
    n = len(all_races)
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        s = stakes[idx].sum()
        if s > 0:
            out.append(pays[idx].sum() / s)
    if not out:
        return (None, None)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def sweep(
    df: pd.DataFrame,
    ev_col: str,
    thresholds: list[float],
    bootstrap: int,
    seed: int,
    min_pop: int | None = None,
    max_pop: int | None = None,
    top_k: int | None = None,
) -> list[dict]:
    """EV/overlay 閾値ごとの単勝 payback。

    top_k を指定すると **ability スコア上位 k 頭**だけを買い対象にする。
    ability の確率分布は本番より平坦になりがちで、平坦な p × 大穴 odds は
    簡単に EV>1 を作ってしまう (実力ではなく校正のゆるさが EV を生む)。
    上位に絞った系列と絞らない系列を並べないと、この罠を切り分けられない。
    """
    all_races = df["race_id"].unique()
    d = df.dropna(subset=["odds_win"])
    d = d[d["odds_win"] > 0]
    if min_pop is not None:
        d = d[d["popularity"] >= min_pop]
    if max_pop is not None:
        d = d[d["popularity"] <= max_pop]
    if top_k is not None:
        d = d[d["rank_abil"] <= top_k]

    out = []
    for t in thresholds:
        bets = d[d[ev_col] > t]
        row = {"threshold": t, **_payback(bets)}
        lo, hi = _bootstrap_payback(bets, all_races, bootstrap, seed)
        row["payback_ci"] = [lo, hi]
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=None, help="モデルディレクトリ (既定: active)")
    ap.add_argument("--calib-start", required=True, help="温度較正窓 (= 学習時の valid) 開始")
    ap.add_argument("--calib-end", required=True)
    ap.add_argument("--start", required=True, help="評価窓 (= 学習時の test) 開始")
    ap.add_argument("--end", required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None, help="結果 JSON の保存先")
    args = ap.parse_args()

    engine = make_engine(db_path())
    with session_scope(engine) as s0:
        model_path = args.model or get_active(s0)
    if model_path is None:
        raise SystemExit("no active model — --model で指定するか active を設定")
    bundle = load_model_full(model_path)
    print(f"model: {model_path}", flush=True)
    if not bundle.nn_odds_feature_cols:
        raise SystemExit(
            "この bundle は odds を head に入れていない (exclude-odds 学習) ので "
            "ability/production の対比ができない"
        )

    with session_scope(engine) as session:
        calib = collect(bundle, session, args.calib_start, args.calib_end, "calib")
        test = collect(bundle, session, args.start, args.end, "test")

    if calib.empty or test.empty:
        raise SystemExit("データが空")

    # 1) 温度 (NLL 最小化)
    T_abil, nll_abil = fit_temperature_nll(calib, "score_abil")
    T_prod, nll_prod = fit_temperature_nll(calib, "score_prod")
    print(f"T_ability={T_abil:.2f} (NLL {nll_abil:.4f}) / T_prod={T_prod:.2f} (NLL {nll_prod:.4f})", flush=True)

    # 2) 確率・overlay
    test = test.copy()
    test["p_abil"] = _softmax_by_race(test, "score_abil", T_abil)
    test["p_prod"] = _softmax_by_race(test, "score_prod", T_prod)
    test["won"] = test["finish_position"] == 1

    inv = np.where(test["odds_win"].values > 0, 1.0 / test["odds_win"].values, np.nan)
    test["inv_odds"] = inv
    denom = test.groupby("race_id")["inv_odds"].transform("sum")
    test["q_market"] = test["inv_odds"] / denom          # 控除率を除いた市場内包確率
    test["ev_abil"] = test["p_abil"] * test["odds_win"]  # ability の期待値
    test["ev_prod"] = test["p_prod"] * test["odds_win"]  # 本番ルールの期待値
    test["overlay"] = test["p_abil"] / test["q_market"]  # 市場に対する乖離比
    # ability スコアのレース内順位 (1 = ability 1 番手)
    test["rank_abil"] = (
        test.groupby("race_id")["score_abil"].rank(ascending=False, method="first").astype(int)
    )

    # 3) スイープ
    ev_grid = [1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0]
    ov_grid = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]

    result = {
        "model": str(model_path),
        "calib_window": [args.calib_start, args.calib_end],
        "test_window": [args.start, args.end],
        "test_races": int(test["race_id"].nunique()),
        "temperature": {
            "T_ability": T_abil, "nll_ability": nll_abil,
            "T_production": T_prod, "nll_production": nll_prod,
        },
        "sweeps": {
            "ability_ev": sweep(test, "ev_abil", ev_grid, args.bootstrap, args.seed),
            "ability_ev_top1": sweep(
                test, "ev_abil", ev_grid, args.bootstrap, args.seed, top_k=1
            ),
            "ability_ev_top3": sweep(
                test, "ev_abil", ev_grid, args.bootstrap, args.seed, top_k=3
            ),
            "ability_overlay_ratio": sweep(test, "overlay", ov_grid, args.bootstrap, args.seed),
            "ability_overlay_ratio_top3": sweep(
                test, "overlay", ov_grid, args.bootstrap, args.seed, top_k=3
            ),
            "production_ev (baseline)": sweep(test, "ev_prod", ev_grid, args.bootstrap, args.seed),
        },
    }

    # 人気帯別 (ability EV 1.1 固定)
    bands = {"pop_1_3": (1, 3), "pop_4_8": (4, 8), "pop_9_": (9, None)}
    result["ability_ev_by_popularity"] = {
        name: sweep(test, "ev_abil", [1.1], args.bootstrap, args.seed, lo, hi)[0]
        for name, (lo, hi) in bands.items()
    }

    # 1 番人気ベタ買いベースライン
    fav = test[test["popularity"] == 1].dropna(subset=["odds_win"])
    result["baseline_favorite"] = {
        **_payback(fav),
        "payback_ci": list(
            _bootstrap_payback(fav, test["race_id"].unique(), args.bootstrap, args.seed)
        ),
    }

    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
