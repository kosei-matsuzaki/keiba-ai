"""Post-hoc に NN モデルの combo 確率 calibrator を fit して保存する。

active NN モデルが `combo_calibrators.pkl` を持たずに学習された場合
(meta.json: has_combo_calibrators=false)、解析的 Plackett-Luce で組んだ
連系 combo 確率がロングショットで過信気味になり、EV=prob×odds が過大評価され、
バックテストで「見かけ EV>1.10 だが実的中は数% → ROI 崩壊」を招く。

このスクリプトは **再学習せずに** valid 区間（モデルが学習に使っていない期間）で
(PL_prob, hit) を集めて isotonic 補正を fit し、`combo_calibrators.pkl` を
モデルディレクトリに追加 + meta.json を更新する。registry は NN bundle でも
この pkl があれば自動で積む (predict_race_with_combinations の _calibrate)。

valid 区間は meta.json の `valid_range` から自動取得する（--valid-start/-end で上書き可）。
test/backtest 区間で fit すると楽観バイアスが乗るので **valid を使うこと**。

Usage:
  KEIBA_DATA_DIR=/tmp/keiba-snap PYTHONPATH=src \
    python -m scripts.fit_nn_combo_calibrators --model data/models/<ts>-nn
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from ai.calibrate import fit_combo_calibrators_bundle
from ai.registry import load_model_full
from core.paths import db_path
from db.session import make_engine, session_scope
from features.builder import build_training_frame


def _valid_window(model_path: Path, cli_start: str | None, cli_end: str | None) -> tuple[str, str]:
    """valid 区間を決定。CLI 優先、無ければ meta.json の valid_range (start/end)。"""
    if cli_start and cli_end:
        return cli_start, cli_end
    meta = json.loads((model_path / "meta.json").read_text())
    vr = meta.get("valid_range")
    if not vr or "/" not in vr:
        raise SystemExit(
            "meta.json に valid_range が無いので --valid-start / --valid-end を指定してください"
        )
    start, end = vr.split("/", 1)
    return cli_start or start, cli_end or end


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="NN モデルディレクトリ")
    ap.add_argument("--valid-start", default=None, help="校正用 valid 開始 (既定: meta.valid_range)")
    ap.add_argument("--valid-end", default=None, help="校正用 valid 終了 (既定: meta.valid_range)")
    ap.add_argument("--n-samples", type=int, default=5_000, help="combo 予測の PL MC サンプル数")
    ap.add_argument(
        "--use-conditional", action="store_true",
        help="surface × n_runners 別の条件付き isotonic を使う (既定: 全体 iso)",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="fit して fitted bet_types を表示するだけ。pkl/meta は書かない",
    )
    args = ap.parse_args()

    model_path: Path = args.model
    if not (model_path / "meta.json").exists():
        raise SystemExit(f"meta.json が見つかりません: {model_path}")

    meta = json.loads((model_path / "meta.json").read_text())
    if meta.get("model_type") != "nn":
        raise SystemExit(f"NN モデルではありません (model_type={meta.get('model_type')})")

    start, end = _valid_window(model_path, args.valid_start, args.valid_end)
    print(f"model      : {model_path}")
    print(f"valid win  : {start} .. {end}")
    print(f"n_samples  : {args.n_samples}  use_conditional={args.use_conditional}")

    # combo_calibrators=None の生 bundle を読む（raw PL prob を集めるため）。
    bundle = load_model_full(model_path)
    if bundle.combo_calibrators is not None:
        print("⚠️  既に combo_calibrators が読み込まれています。raw 収集には影響しません"
              " (fit 用に内部で combo_calibrators を使わない予測経路を通ります) が、"
              "上書き保存になります。")

    engine = make_engine(db_path())
    print("building valid feature frame …")
    with session_scope(engine) as session:
        valid_frame = build_training_frame(session, train_start=start, train_end=end)
    n_races = valid_frame["race_id"].nunique() if not valid_frame.empty else 0
    print(f"valid frame: {len(valid_frame)} rows / {n_races} races")
    if valid_frame.empty:
        raise SystemExit("valid frame が空です。期間を確認してください")

    print("fitting ComboCalibrators (this runs per-race combo prediction) …")
    cal = fit_combo_calibrators_bundle(
        valid_frame, bundle,
        n_samples=args.n_samples,
        use_conditional=args.use_conditional,
    )
    fitted = cal.fitted_bet_types
    print(f"fitted bet_types: {fitted}")
    if not fitted:
        raise SystemExit("どの bet_type も fit されませんでした (各 <100 サンプル?)。中断します")

    if args.dry_run:
        print("[dry-run] pkl/meta は書きませんでした")
        return

    pkl_path = model_path / "combo_calibrators.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(cal, f)
    print(f"saved: {pkl_path}")

    # meta.json を更新（has_combo_calibrators / bet_types）
    meta["has_combo_calibrators"] = True
    meta["combo_calibrators_bet_types"] = fitted
    (model_path / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )
    print("meta.json updated: has_combo_calibrators=true")


if __name__ == "__main__":
    main()
