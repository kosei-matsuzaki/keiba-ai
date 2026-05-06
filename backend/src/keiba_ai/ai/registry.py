"""Model registry — save, list, load, and activate trained LightGBM models.

Each model is stored under data/models/<YYYYMMDD-HHMMSS>/ with:
  model.txt  — LightGBM Booster serialized text
  meta.json  — params, ranges, metrics, feature columns
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
from sqlalchemy.orm import Session

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
) -> Path:
    """Persist model and metadata; return the model directory path.

    feature_columns: 学習で実際に使った特徴量列。None のときは LightGBM の
    feature_name() を使う（lightgbm が学習時に設定済み）。
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = _models_dir() / ts
    model_dir.mkdir(parents=True, exist_ok=True)

    model_txt = model_dir / "model.txt"
    model.save_model(str(model_txt))

    if feature_columns is None:
        feature_columns = list(model.feature_name())

    meta = {
        "timestamp": ts,
        "params": params,
        "train_range": train_range,
        "valid_range": valid_range,
        "metrics": metrics,
        "feature_columns": feature_columns,
        "notes": notes,
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
    """Load a LightGBM Booster from a model directory or model.txt file."""
    model_txt = path / "model.txt" if path.is_dir() else path
    return lgb.Booster(model_file=str(model_txt))


def set_active(model_path: Path, session: Session) -> None:
    """Mark the model at model_path as active; deactivate all others."""
    path_str = str(model_path)
    runs = session.query(ModelRun).all()
    for run in runs:
        run.is_active = 1 if run.model_path == path_str else 0
    session.flush()


def get_active(session: Session) -> Path | None:
    """Return the path of the currently active model, or None."""
    run = session.query(ModelRun).filter(ModelRun.is_active == 1).first()
    if run is None:
        return None
    return Path(run.model_path)
