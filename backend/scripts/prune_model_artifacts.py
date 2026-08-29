"""model_runs に行の無いモデルディレクトリ（孤児）を棚卸しする。

**孤児 = ゴミではない。** walk-forward（`scripts/walk_forward_oof.py`）は fold ごとに
モデルを学習し、Models 画面を汚さないよう `model_runs` の行だけ消してディレクトリは
残す。`scripts/combo_walk_forward.py` はその `meta.json` から fold とモデルの対応を
復元するので、消すと前進検証をやり直せなくなる。

そのため既定は**一覧を出すだけ**。削除は `--delete` を明示したときだけ行い、
さらに fold モデル（学習期間が本番 active より前で、前進検証に使いうるもの）は
`--include-folds` を付けない限り残す。

Usage:
  # 棚卸し
  PYTHONIOENCODING=utf-8 PYTHONPATH=src python -m scripts.prune_model_artifacts

  # fold 以外の孤児を消す
  PYTHONIOENCODING=utf-8 PYTHONPATH=src python -m scripts.prune_model_artifacts --delete
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from sqlalchemy import text

from core.paths import data_dir, db_path
from db.session import make_engine, session_scope


def _known_paths() -> set[str]:
    """model_runs が参照しているディレクトリ名（basename で比較）。"""
    engine = make_engine(db_path())
    with session_scope(engine) as s:
        rows = s.execute(text("SELECT model_path FROM model_runs")).scalars().all()
    engine.dispose()
    return {Path(str(p)).name for p in rows}


def _fold_cutoffs() -> set[str]:
    """前進検証で使う fold の学習終了日。fold_models.txt があればそこからも拾う。"""
    cutoffs: set[str] = set()
    log = data_dir() / "analysis" / "oof" / "fold_models.txt"
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if parts:
                cutoffs.add(parts[0].split("_")[0])
    return cutoffs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delete", action="store_true", help="孤児を実際に削除する")
    ap.add_argument(
        "--include-folds", action="store_true",
        help="前進検証の fold モデルも削除対象にする（やり直せなくなるので既定は除外）",
    )
    args = ap.parse_args()

    known = _known_paths()
    fold_cutoffs = _fold_cutoffs()
    models_dir = data_dir() / "models"

    orphans: list[tuple[Path, str, str, bool]] = []
    total = 0
    for d in sorted(models_dir.glob("*-nn")):
        total += 1
        if d.name in known:
            continue
        meta_path = d / "meta.json"
        loss, train_end = "?", "?"
        if meta_path.exists():
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            loss = str(m.get("loss_type"))
            train_end = (m.get("train_range") or "/").split("/")[-1]
        # fold かどうか: walk-forward の cutoff と一致するか、本番 active より前の学習
        is_fold = train_end in fold_cutoffs or train_end < "2024-04-28"
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        orphans.append((d, f"{loss} / 学習〜{train_end}", f"{size / 1e6:.1f}MB", is_fold))

    print(f"モデルディレクトリ {total} 個 / model_runs 行 {len(known)} 個 "
          f"→ 孤児 {len(orphans)} 個")
    print()
    print(f'{"ディレクトリ":<24}{"内容":<34}{"サイズ":>8}  区分')
    for d, info, size, is_fold in orphans:
        print(f"  {d.name:<24}{info:<34}{size:>8}  {'fold (残す)' if is_fold else '不明'}")

    targets = [o for o in orphans if args.include_folds or not o[3]]
    print()
    if not args.delete:
        print(f"削除対象は {len(targets)} 個。実際に消すには --delete を付ける。")
        if not args.include_folds:
            print("（前進検証の fold モデルは既定で除外。含めるなら --include-folds）")
        return

    freed = 0
    for d, _, _, _ in targets:
        freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
        shutil.rmtree(d)
        print(f"  削除: {d.name}")
    print(f"{len(targets)} 個を削除し、{freed / 1e6:.1f}MB を解放した。")


if __name__ == "__main__":
    main()
