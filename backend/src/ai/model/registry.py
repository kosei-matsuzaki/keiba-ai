"""Model registry — save, list, load, and activate trained NN models.

Each NN model is stored under data/models/<YYYYMMDDTHHMMSS>-nn/ with:
  model.pt              — PyTorch state_dict
  preprocessor.pkl      — カテゴリ map + 数値標準化 (optional)
  temperature_scaler.pkl — 温度スケーリング (optional)
  meta.json             — model_type="nn", params, feature columns, etc.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from ai.core.temperature import TemperatureScaler
from ai.model._artifacts_nn import load_nn_artifacts, save_nn_artifacts
from core.paths import data_dir
from db.models.model_run import ModelRun
from features.builder import FEATURE_COLUMNS

if TYPE_CHECKING:
    import torch.nn

    from ai.model.preprocess import NNPreprocessor


@dataclass
class ModelMeta:
    path: Path
    timestamp: str
    params: dict
    train_range: str | None
    valid_range: str | None
    metrics: dict
    feature_columns: list[str]
    loss_type: str | None = None
    conditional_calibration: bool = False
    model_type: str = "nn"


def _models_dir() -> Path:
    d = data_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_nn_model(
    state_dict_path: Path,
    meta_dict: dict,
    model_dir_name: str | None = None,
) -> Path:
    """Register an already-saved NN model into the registry layout.

    train_nn.py saves model.pt and meta.json directly.  This helper provides
    an alternative entry point that places artifacts under data/models/ and
    ensures meta.json is written consistently.

    Args:
        state_dict_path: Path to the existing model.pt file.
        meta_dict: Metadata dict (must include model_type, horse_feature_cols,
            race_feature_cols, params, metrics, feature_columns, etc.).
        model_dir_name: Optional subdirectory name.  Defaults to the parent
            directory name of state_dict_path (so train_nn's timestamp dir is
            reused when the file is already inside data/models/).

    Returns:
        Path to the model directory (parent of model.pt).
    """
    if model_dir_name is None:
        model_dir = state_dict_path.parent
    else:
        model_dir = _models_dir() / model_dir_name
        model_dir.mkdir(parents=True, exist_ok=True)
        save_nn_artifacts(state_dict_path, model_dir)

    # Ensure model_type is set
    meta_dict.setdefault("model_type", "nn")

    (model_dir / "meta.json").write_text(
        json.dumps(meta_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return model_dir


def list_models() -> list[ModelMeta]:
    """Return all model generations found in data/models/, sorted by timestamp."""
    base = _models_dir()
    results: list[ModelMeta] = []
    for candidate in sorted(base.iterdir()):
        meta_file = candidate / "meta.json"
        if not meta_file.exists():
            continue
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        results.append(
            ModelMeta(
                path=candidate,
                timestamp=meta.get("timestamp", candidate.name),
                params=meta.get("params", {}),
                train_range=meta.get("train_range"),
                valid_range=meta.get("valid_range"),
                metrics=meta.get("metrics", {}),
                feature_columns=meta.get("feature_columns", FEATURE_COLUMNS),
                loss_type=meta.get("loss_type"),
                conditional_calibration=bool(meta.get("conditional_calibration", False)),
                model_type=meta.get("model_type", "nn"),
            )
        )
    return results


@dataclass
class ModelBundle:
    """All artifacts saved alongside a NN model directory.

        model_type:            常に "nn"
        model_dir:             モデルディレクトリのパス
        meta:                  meta.json の内容 (dict)
        feature_columns:       全特徴量列
        nn_model:              RaceModel インスタンス (eval 済み)
        nn_horse_feature_cols: 馬ごとの特徴量列
        nn_race_feature_cols:  レースレベルの特徴量列
        nn_preprocessor:       カテゴリ map + 数値標準化 (train で fit 済み)。
                               旧モデルでは None (推論側で legacy fallback)。
        temperature_scaler:    温度スケーリング (optional)
    """

    model_type: str  # 常に "nn"
    model_dir: Path
    meta: dict
    feature_columns: list[str]
    # torch は遅延 import のため型は文字列注釈のみ
    nn_model: torch.nn.Module | None = None
    nn_horse_feature_cols: list[str] | None = None
    nn_race_feature_cols: list[str] | None = None
    nn_preprocessor: NNPreprocessor | None = None
    # 温度スケーリング (optional)
    temperature_scaler: TemperatureScaler | None = None
    # odds-at-scoring head (v3): odds は encoder でなく head で使う。None = v2 (encoder)。
    nn_odds_feature_cols: list[str] | None = None
    # per-race 履歴エンコーダ (serving): 推論時に過去走系列を構築し標準化するための情報。
    nn_history_norm: tuple | None = None  # (mean, std) ndarray
    nn_history_max_len: int = 0
    nn_history_feat_dim: int = 0


def load_model_full(path: Path) -> ModelBundle:
    """Load a NN ModelBundle from a model directory.

    RaceModel (eval mode) + horse/race feature column lists + optional
    preprocessor / temperature scaler。

    torch がインストールされていない環境ではロードに失敗する (ImportError)。
    """
    if not path.is_dir():
        raise ValueError(f"model path must be a directory, got {path}")

    meta_path = path / "meta.json"
    meta: dict = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    feature_columns: list[str] = meta.get("feature_columns", list(FEATURE_COLUMNS))

    nn_artifacts = load_nn_artifacts(path, meta)
    return ModelBundle(
        model_type="nn",
        model_dir=path,
        meta=meta,
        feature_columns=feature_columns,
        nn_model=nn_artifacts["nn_model"],
        nn_horse_feature_cols=nn_artifacts["nn_horse_feature_cols"],
        nn_race_feature_cols=nn_artifacts["nn_race_feature_cols"],
        nn_preprocessor=nn_artifacts["nn_preprocessor"],
        temperature_scaler=nn_artifacts["temperature_scaler"],
        nn_odds_feature_cols=nn_artifacts.get("nn_odds_feature_cols"),
        nn_history_norm=nn_artifacts.get("nn_history_norm"),
        nn_history_max_len=nn_artifacts.get("nn_history_max_len", 0),
        nn_history_feat_dim=nn_artifacts.get("nn_history_feat_dim", 0),
    )


def set_active(model_path: Path, session: Session) -> None:
    """Mark the model at model_path as active; deactivate all others.

    パス比較は basename (timestamp ディレクトリ名) ベースで行う。これにより
    WSL で保存した `/mnt/c/...` を Windows サイドカーから activate しても
    正しく一致する (Path() の str() 化で起こる区切り文字の差を回避)。
    """
    target_name = Path(model_path).name
    runs = session.query(ModelRun).all()
    for run in runs:
        run.is_active = 1 if Path(run.model_path).name == target_name else 0
    session.flush()


def set_active_by_id(model_id: int, session: Session) -> None:
    """Activate by ModelRun.id directly. パス比較不要なので最も堅牢。"""
    runs = session.query(ModelRun).all()
    for run in runs:
        run.is_active = 1 if run.id == model_id else 0
    session.flush()


def delete_model_files(stored_path: str) -> None:
    """data/models/<ts>/ ディレクトリを再帰削除する。

    stored_path は ModelRun.model_path の値 (WSL/Windows のいずれで保存されたパスでも可)。
    _resolve_model_path と同様 basename ベースで現プラットフォーム上に解決し、存在すれば
    rmtree で消す。存在しなければ no-op。
    """
    import shutil

    resolved = _resolve_model_path(stored_path)
    if resolved.is_dir():
        shutil.rmtree(resolved)


def renumber_model_ids(session: Session) -> None:
    """ModelRun.id を created_at 昇順で 1, 2, 3, ... に振り直す。

    削除を繰り返すと autoincrement のせいで「モデル 1 個しかないのに id=13」のような
    飛び番が残りがちなので、削除後に呼んで詰める。

    simulation_runs.model_run_id が ModelRun.id を FK 参照する (ON UPDATE CASCADE)
    ため、ここで id を振り直すと子の参照も自動追従する。FK enforcement は
    db/session.py の PRAGMA foreign_keys=ON 前提。

    主キー UPDATE で衝突しないよう、一度全 id を +1_000_000 にオフセットしてから
    1..N の本番値を割り当てる。AUTOINCREMENT を使うテーブルがある場合は
    sqlite_sequence の seq も併せて更新して、次回 INSERT が N+1 から始まるようにする。
    """
    from sqlalchemy import select, text

    runs = session.scalars(
        select(ModelRun).order_by(ModelRun.created_at, ModelRun.id)
    ).all()

    if runs:
        offset = 1_000_000
        for r in runs:
            r.id = r.id + offset
        session.flush()
        for new_id, r in enumerate(runs, start=1):
            r.id = new_id
        session.flush()

    # sqlite_sequence は AUTOINCREMENT 宣言があるスキーマでのみ存在するため
    # IF EXISTS 相当のガードを入れる (未使用なら触らない)。
    has_sequence_table = session.execute(
        text("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'")
    ).first()
    if has_sequence_table:
        if runs:
            session.execute(
                text(
                    "INSERT OR REPLACE INTO sqlite_sequence(name, seq) "
                    "VALUES('model_runs', :seq)"
                ),
                {"seq": len(runs)},
            )
        else:
            session.execute(text("DELETE FROM sqlite_sequence WHERE name='model_runs'"))


def _resolve_model_path(stored_path: str) -> Path:
    """DB に格納された model_path を、現在の data_dir で解決し直す。

    WSL で保存した `/mnt/c/.../data/models/<ts>` のようなパスを Windows サイド
    カーから扱うとき、Path 解釈で実ファイルにたどり着けないことがある。
    basename (= timestamp ディレクトリ名) を取り出して `data_dir() / "models"
    / <ts>` に再配置することで、現プラットフォームに依存しない解決ができる。
    再配置先が存在しなければ元の path をそのまま返す (後方互換)。
    """
    raw = Path(stored_path)
    fallback = data_dir() / "models" / raw.name
    if fallback.exists():
        return fallback
    return raw


def get_active(session: Session) -> Path | None:
    """Return the path of the currently active model, or None."""
    run = session.query(ModelRun).filter(ModelRun.is_active == 1).first()
    if run is None:
        return None
    return _resolve_model_path(run.model_path)
