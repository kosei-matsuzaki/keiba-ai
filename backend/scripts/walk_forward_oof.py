"""二段目 (active への信用度) の学習データを作る — 一段目の out-of-sample 出力。

**なぜ必要か**: active は 2015〜2024-04 で学習しているので、その期間の active の出力は
答えを部分的に覚えた後のものであり、本番より遥かに当たって見える。そこで信用度を
学習すると「active は信用してよい」という実運用では成立しない関係を覚える
(stacking の典型的な失敗)。

そこで cutoff を進めながら一段目を学習し直し、**毎回まだ見ていない区間だけ**を
予測して積み上げる。得られるのは「本番と同じ条件で出された一段目の出力」で、
これなら二段目を正しく学習できる。副産物として標本が増える (test 窓の 5,390 →
fold 数 × 約 2,000)。検出力不足も同時に解ける。

特徴量フレームは**全期間 1 個を組んで使い回す** (fold ごとに組み直すと 1 回 60 分)。

Usage:
  PYTHONIOENCODING=utf-8 PYTHONPATH=src python -m scripts.walk_forward_oof \
      --cutoffs 2019-06-30,2019-12-31 --losses multi,plackett_luce
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta
from sqlalchemy import text

from ai.inference.predict import predict_race
from ai.model.registry import load_model_full
from ai.training.train_nn import train_nn
from core.paths import data_dir, db_path
from db.session import make_engine, session_scope
from features.builder import build_training_frame

# 損失ごとの監視指標。一段目は本番と同じ選び方をさせる。
MONITOR = {
    "multi": "valid_tansho_roi",
    "log_growth": "valid_tansho_roi",
    "plackett_luce": "valid_ndcg3",
}


def _fold_dates(cutoff: str, valid_months: int, test_months: int) -> tuple[str, str, str]:
    """学習終了日 `cutoff` から、train_nn に渡す基準日と OOF 区間を出す。

    **`train_nn(train_end=...)` は「学習の終了日」ではない。** `ai/core/splits.time_split`
    は基準日から逆算する:

        test_start  = train_end - test_months
        valid_start = test_start - valid_months
        train       = [min, valid_start)
        test        = [test_start, **フレームの最後まで**]

    つまり学習が終わるのは `train_end - test_months - valid_months` で、test には
    上限が無い。素直に cutoff を渡すと、学習が 2 年短くなったうえに「未見区間」が
    数年分に膨らんで OOF が汚染される (実際にこれを踏んだ)。

    そこで逆に解いて基準日を作り、OOF 区間は自分で閉じる。

    Returns:
        (train_nn に渡す基準日, OOF 区間の開始, OOF 区間の終了)
    """
    c = date.fromisoformat(cutoff)
    oof_start = c + relativedelta(months=valid_months)
    oof_end = oof_start + relativedelta(months=test_months)
    ref = oof_end  # test_start = ref - test_months = oof_start となるように
    return ref.isoformat(), oof_start.isoformat(), oof_end.isoformat()


def _drop_model_run(model_dir: str) -> None:
    """walk-forward が作った model_runs 行を消す (Models 画面を汚さないため)。

    active 行には絶対に触らない。モデルのディレクトリは残す (再予測に使えるため)。
    """
    engine = make_engine(db_path())
    with session_scope(engine) as s:
        s.execute(
            text("DELETE FROM model_runs WHERE model_path = :p AND is_active = 0"),
            {"p": str(model_dir)},
        )
    engine.dispose()


def _predict_slice(model_dir: Path, frame: pd.DataFrame, start: str, end: str,
                   loss: str, cutoff: str) -> list[dict]:
    """その fold の未見区間を 1 頭 1 行で予測する。

    二段目は全馬を並べ直せる形が良い (1 レース 1 行より 14 倍のラベル情報が使える)
    ので、本命だけでなく全馬を書き出す。
    """
    bundle = load_model_full(model_dir)
    sl = frame[(frame["date"] >= start) & (frame["date"] <= end)]
    rows: list[dict] = []
    engine = make_engine(db_path())
    # session はループ外で保持する (履歴 GRU が zero に degrade するのを防ぐ)
    with session_scope(engine) as session:
        race_ids = sl["race_id"].unique()
        for i, rid in enumerate(race_ids):
            rf = sl[sl["race_id"] == rid]
            if len(rf) < 2:
                continue
            try:
                preds = predict_race(bundle, rf, session=session)
            except Exception as exc:  # noqa: BLE001
                print(f"    predict failed {rid}: {exc}", flush=True)
                continue
            rank = {h: r for r, h in enumerate(preds["horse_id"])}
            pmap = preds.set_index("horse_id")
            for _, row in rf.iterrows():
                hid = row["horse_id"]
                if hid not in pmap.index:
                    continue
                p = pmap.loc[hid]
                rows.append({
                    "race_id": rid,
                    "date": row["date"],
                    "horse_id": hid,
                    "post_position": row.get("post_position"),
                    "odds_win": row.get("odds_win"),
                    "popularity": row.get("popularity"),
                    "n_runners": row.get("n_runners"),
                    "race_class": row.get("race_class"),
                    "finish_position": row.get("finish_position"),
                    # 一段目の出力
                    "model_loss": loss,
                    "model_cutoff": cutoff,
                    # 二段目に「この一段目がどれだけ学習したか」を渡すための列
                    "train_races": int(frame[frame["date"] <= cutoff]["race_id"].nunique()),
                    "score": float(p["score"]),
                    "win_prob": float(p["win_prob"]),
                    "place_prob": float(p["place_prob"]),
                    "model_rank": int(rank[hid]),
                })
            if (i + 1) % 500 == 0:
                print(f"    {i + 1}/{len(race_ids)}", flush=True)
    engine.dispose()
    return rows


def _check_oof(rows: list[dict], cutoff: str, t_start: str, t_end: str) -> None:
    """書き出す直前に OOF が意図どおりかを検証する。

    **なぜ要るか**: 2026-08-25 に、停止したはずの旧プロセス (nohup 越しなので
    タスクを止めてもプロセスは生きていた) が 2 時間半後に正しい出力を上書きし、
    学習期間ごと混ざった OOF (23,960 レース / 2019-05〜2026-08) が紛れ込んだ。
    区間の一覧を目視するまで気づけず、そのまま二段目を学習していたら結論は
    全て無意味になっていた。プロセス管理は間違えうるので、**データ側で弾く**。

    Raises:
        SystemExit: 区間が cutoff より前に食い込んでいる / 想定区間を超えている /
            レース数が 1 fold としてありえない場合。
    """
    if not rows:
        raise SystemExit(f"OOF が空 (cutoff={cutoff})")
    dates = [r["date"] for r in rows]
    lo, hi = min(dates), max(dates)
    n_races = len({r["race_id"] for r in rows})
    if lo < t_start or hi > t_end:
        raise SystemExit(
            f"OOF 区間が想定外 (cutoff={cutoff}): 実際 {lo}..{hi} / 想定 {t_start}..{t_end}。"
            "別プロセスの出力が混ざっていないか確認すること。"
        )
    if lo <= cutoff:
        raise SystemExit(
            f"OOF が学習期間に食い込んでいる (cutoff={cutoff}, 最古 {lo})"
        )
    # 6 ヶ月の fold は概ね 1,400〜2,000 レース。桁違いなら何かが混ざっている。
    if n_races > 3_000:
        raise SystemExit(
            f"OOF のレース数が多すぎる (cutoff={cutoff}, {n_races} レース)。"
            "区間が閉じていない可能性がある。"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cutoffs", required=True, help="カンマ区切りの train_end (YYYY-MM-DD)")
    ap.add_argument("--losses", default="multi", help="カンマ区切りの損失")
    ap.add_argument("--valid-months", type=int, default=6)
    ap.add_argument("--test-months", type=int, default=6)
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or (data_dir() / "analysis" / "oof")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 全期間のフレームを 1 回だけ組む (以降 fold ごとに使い回す)
    engine = make_engine(db_path())
    with session_scope(engine) as session:
        print("building the full feature frame (cached after the first run)...", flush=True)
        frame = build_training_frame(session)
    print(f"frame: {len(frame)} rows / {frame['race_id'].nunique()} races", flush=True)

    for cutoff in args.cutoffs.split(","):
        for loss in args.losses.split(","):
            tag = f"{cutoff}_{loss}"
            out = out_dir / f"oof_{tag}.csv"
            if out.exists():
                print(f"[skip] {tag} (already done)", flush=True)
                continue
            print(f"[train] cutoff={cutoff} loss={loss}", flush=True)
            ref, t_start, t_end = _fold_dates(cutoff, args.valid_months, args.test_months)
            print(f"  学習 〜{cutoff} / OOF 区間 {t_start}..{t_end} (基準日 {ref})", flush=True)
            metrics = train_nn(
                prebuilt_frame=frame,
                train_end=ref,
                valid_months=args.valid_months,
                test_months=args.test_months,
                loss=loss,
                monitor=MONITOR.get(loss, "valid_tansho_roi"),
                max_epochs=args.max_epochs,
                device="cpu",
                persist=True,          # 予測にモデルが要るので保存する
            )
            model_dir = Path(metrics["model_dir"])
            meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
            actual_train_end = (meta.get("train_range") or "/").split("/")[-1]
            if actual_train_end > cutoff:
                # 逆算がずれていたら OOF が汚染されるので、書かずに止める。
                raise SystemExit(
                    f"学習が cutoff を超えた (train_end={actual_train_end} > {cutoff})。"
                    "_fold_dates と time_split の対応を確認すること。"
                )
            print(f"  実際の学習範囲 {meta.get('train_range')}", flush=True)
            rows = _predict_slice(model_dir, frame, t_start, t_end, loss, cutoff)
            _check_oof(rows, cutoff, t_start, t_end)
            pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
            print(f"  wrote {len(rows)} rows -> {out}", flush=True)
            _drop_model_run(str(model_dir))

    engine.dispose()
    print("done", flush=True)


if __name__ == "__main__":
    main()
