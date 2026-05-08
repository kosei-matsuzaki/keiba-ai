"""Model registry — save, list, load, and activate trained LightGBM models.

Each model is stored under data/models/<YYYYMMDD-HHMMSS>/ with:
  model.txt        — LightGBM lambdarank Booster (順位用、必須)
  binary.txt       — LightGBM binary classifier (勝率用、Phase 2 で追加; optional)
  calibrator.pkl   — IsotonicCalibrator (binary 出力の post-hoc 補正; optional)
  meta.json        — params, ranges, metrics, feature columns
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
from sqlalchemy.orm import Session

from keiba_ai.ai.calibrate import IsotonicCalibrator
from keiba_ai.core.paths import data_dir
from keiba_ai.db.models.model_run import ModelRun
from keiba_ai.features.builder import FEATURE_COLUMNS


@dataclass
class ModelMeta:
    path: Path
    timestamp: str
    params: dict
    train_range: str | None
    valid_range: str | None
    metrics: dict
    feature_columns: list[str]


def _models_dir() -> Path:
    d = data_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_model(
    model: lgb.Booster,
    params: dict,
    train_range: str | None,
    valid_range: str | None,
    metrics: dict,
    notes: str | None = None,
    feature_columns: list[str] | None = None,
    binary_model: lgb.Booster | None = None,
    calibrator: IsotonicCalibrator | None = None,
) -> Path:
    """Persist model + (optional) binary classifier + calibrator and metadata.

    Args:
        model: lambdarank Booster (順位用、必須)。
        binary_model: 同じ feature で学習した is_winner 二項分類器 (任意)。
        calibrator: binary_model 出力を post-hoc 補正する isotonic regression
            (任意)。binary_model と calibrator は **両方揃って初めて意味がある**。
        feature_columns: 学習で実際に使った特徴量列。None のときは LightGBM の
            feature_name() を使う。
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = _models_dir() / ts
    model_dir.mkdir(parents=True, exist_ok=True)

    model_txt = model_dir / "model.txt"
    model.save_model(str(model_txt))

    if feature_columns is None:
        feature_columns = list(model.feature_name())

    has_binary = binary_model is not None
    has_calibrator = calibrator is not None

    if has_binary:
        binary_model.save_model(str(model_dir / "binary.txt"))
    if has_calibrator:
        with (model_dir / "calibrator.pkl").open("wb") as f:
            pickle.dump(calibrator, f)

    meta = {
        "timestamp": ts,
        "params": params,
        "train_range": train_range,
        "valid_range": valid_range,
        "metrics": metrics,
        "feature_columns": feature_columns,
        "notes": notes,
        "has_binary_model": has_binary,
        "has_calibrator": has_calibrator,
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    return model_dir


def list_models() -> list[ModelMeta]:
    """Return all model generations found in data/models/, sorted by timestamp."""
    base = _models_dir()
    results: list[ModelMeta] = []
    for candidate in sorted(base.iterdir()):
        meta_file = candidate / "meta.json"
        if not meta_file.exists():
            continue
        meta = json.loads(meta_file.read_text())
        results.append(
            ModelMeta(
                path=candidate,
                timestamp=meta.get("timestamp", candidate.name),
                params=meta.get("params", {}),
                train_range=meta.get("train_range"),
                valid_range=meta.get("valid_range"),
                metrics=meta.get("metrics", {}),
                feature_columns=meta.get("feature_columns", FEATURE_COLUMNS),
            )
        )
    return results


def load_model(path: Path) -> lgb.Booster:
    """Load the lambdarank Booster (model.txt) from a model directory or path.

    For backwards compatibility this only returns the lambdarank model.
    Use load_model_full() to additionally retrieve the binary classifier and
    calibrator when present (Phase 2 onward).
    """
    model_txt = path / "model.txt" if path.is_dir() else path
    return lgb.Booster(model_file=str(model_txt))


@dataclass
class ModelBundle:
    """All artifacts saved alongside a model directory.

    lambdarank: 順位用 Booster (必須)
    binary:     勝率用二項分類器 (Phase 2 以降のモデルで設定)
    calibrator: binary 出力の post-hoc 補正 (binary とセット)
    """

    lambdarank: lgb.Booster
    binary: lgb.Booster | None
    calibrator: IsotonicCalibrator | None


def load_model_full(path: Path) -> ModelBundle:
    """Load lambdarank + (optional) binary classifier + calibrator.

    旧モデル (model.txt のみ) でも安全にロード可能。
    binary.txt / calibrator.pkl が無ければ None を返す。
    """
    if not path.is_dir():
        # path が model.txt 直接指定なら lambdarank のみで返す
        return ModelBundle(
            lambdarank=lgb.Booster(model_file=str(path)),
            binary=None,
            calibrator=None,
        )

    lambdarank = lgb.Booster(model_file=str(path / "model.txt"))

    binary_path = path / "binary.txt"
    binary = lgb.Booster(model_file=str(binary_path)) if binary_path.exists() else None

    calibrator_path = path / "calibrator.pkl"
    calibrator = None
    if calibrator_path.exists():
        with calibrator_path.open("rb") as f:
            calibrator = pickle.load(f)

    return ModelBundle(lambdarank=lambdarank, binary=binary, calibrator=calibrator)


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
