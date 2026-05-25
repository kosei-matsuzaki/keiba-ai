"""Model registry — save, list, load, and activate trained models.

Supports both LightGBM GBDT models and PyTorch NN models.

Each GBDT model is stored under data/models/<YYYYMMDD-HHMMSS>/ with:
  model.txt        — LightGBM lambdarank Booster (順位用、必須)
  binary.txt       — LightGBM binary classifier (勝率用、Phase 2 で追加; optional)
  calibrator.pkl   — IsotonicCalibrator (binary 出力の post-hoc 補正; optional)
  combo_calibrators.pkl — ComboCalibrators (連系 馬券種補正; optional)
  meta.json        — params, ranges, metrics, feature columns

Each NN model is stored under data/models/<YYYYMMDDTHHMMSS>-nn/ with:
  model.pt         — PyTorch state_dict
  meta.json        — model_type="nn", params, feature columns, etc.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import lightgbm as lgb
from sqlalchemy.orm import Session

from ai._registry_gbdt import load_gbdt_artifacts, save_gbdt_artifacts
from ai._registry_nn import load_nn_artifacts, save_nn_artifacts
from ai.calibrate import ComboCalibrators, IsotonicCalibrator
from ai.temperature import TemperatureScaler
from core.paths import data_dir
from db.models.model_run import ModelRun
from features.builder import FEATURE_COLUMNS

if TYPE_CHECKING:
    import torch.nn

    from ai.nn.preprocess import NNPreprocessor


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
    model_type: str = "gbdt"  # "gbdt" or "nn" — default preserves backward compat


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
    combo_calibrators: ComboCalibrators | None = None,
    loss_type: str | None = None,
    conditional_calibration: bool = False,
    model_type: str = "gbdt",
    temperature_scaler: TemperatureScaler | None = None,
) -> Path:
    """Persist model + (optional) binary classifier + calibrator and metadata.

    Args:
        model: lambdarank Booster (順位用、必須)。
        binary_model: 同じ feature で学習した is_winner 二項分類器 (任意)。
        calibrator: binary_model 出力を post-hoc 補正する isotonic regression
            (任意)。binary_model と calibrator は **両方揃って初めて意味がある**。
        feature_columns: 学習で実際に使った特徴量列。None のときは LightGBM の
            feature_name() を使う。
        model_type: "gbdt" (default) or "nn". meta.json に記録する。
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = _models_dir() / ts
    model_dir.mkdir(parents=True, exist_ok=True)

    if feature_columns is None:
        feature_columns = list(model.feature_name())

    has_flags = save_gbdt_artifacts(
        model_dir,
        model=model,
        binary_model=binary_model,
        calibrator=calibrator,
        combo_calibrators=combo_calibrators,
        temperature_scaler=temperature_scaler,
    )

    meta = {
        "model_type": model_type,
        "timestamp": ts,
        "params": params,
        "train_range": train_range,
        "valid_range": valid_range,
        "metrics": metrics,
        "feature_columns": feature_columns,
        "notes": notes,
        "loss_type": loss_type,
        "conditional_calibration": conditional_calibration,
        **has_flags,
    }
    (model_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return model_dir


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
                model_type=meta.get("model_type", "gbdt"),
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

    GBDT 経路:
        lambdarank:        順位用 Booster (GBDT の場合は必須、NN では None)
        binary:            勝率用二項分類器 (Phase 2 以降のモデルで設定)
        calibrator:        binary 出力の post-hoc 補正 (binary とセット)
        combo_calibrators: 連系 馬券種ごとの PL prob 補正 (#44, optional)

    NN 経路:
        nn_model:              RaceModel インスタンス (eval 済み)
        nn_horse_feature_cols: 馬ごとの特徴量列
        nn_race_feature_cols:  レースレベルの特徴量列
        nn_preprocessor:       カテゴリ map + 数値標準化 (train で fit 済み)。
                               旧モデルでは None (推論側で legacy fallback)。

    共通:
        model_type:     "gbdt" or "nn"
        model_dir:      モデルディレクトリのパス
        meta:           meta.json の内容 (dict)
        feature_columns: 全特徴量列 (GBDT / NN 共通インターフェース)
    """

    model_type: str  # "gbdt" or "nn"
    model_dir: Path
    meta: dict
    feature_columns: list[str]
    # GBDT 経路
    lambdarank: lgb.Booster | None = None
    binary: lgb.Booster | None = None
    calibrator: IsotonicCalibrator | None = None
    combo_calibrators: ComboCalibrators | None = None
    # NN 経路 (torch は遅延 import のため型は文字列注釈のみ)
    nn_model: "torch.nn.Module | None" = None
    nn_horse_feature_cols: list[str] | None = None
    nn_race_feature_cols: list[str] | None = None
    nn_preprocessor: NNPreprocessor | None = None
    # GBDT stacking: NN was trained on features augmented with this GBDT
    # bundle's predictions.  Inference must apply the same augmentation
    # before calling the NN.
    nn_gbdt_bundle: "ModelBundle | None" = None
    # GBDT ensemble (inference-time blending, no training-side coupling):
    # blend the per-horse (win_prob, place_prob) between NN and this GBDT
    # using ``nn_ensemble_weight`` (1.0 = pure NN, 0.0 = pure GBDT).
    # Ranking score stays as the NN's, so ordering doesn't change.
    nn_ensemble_gbdt_bundle: "ModelBundle | None" = None
    nn_ensemble_weight: float = 1.0
    # Post-hoc isotonic calibrator for NN win_prob. Applied after the temperature
    # scaler / softmax to fix systematic over-confidence on longshots. None for
    # GBDT bundles (GBDT uses `calibrator` above) and for NN models trained
    # before the calibrator path was introduced.
    nn_calibrator: IsotonicCalibrator | None = None
    # Post-hoc isotonic calibrator for place_prob (GBDT / NN 共通). place_prob は
    # PL Monte Carlo 由来で未校正のため穴の複勝率を過大評価しがち。win とは別に
    # place 結果で fit した monotonic 補正を適用する (race 内で合計 1 にならない
    # ので normalise=False)。optional — 無い旧モデルは未校正のまま。
    place_calibrator: IsotonicCalibrator | None = None
    # 温度スケーリング (GBDT / NN 共通; optional)
    temperature_scaler: TemperatureScaler | None = None


def load_model_full(path: Path) -> ModelBundle:
    """Load a ModelBundle from a model directory.

    meta.json の model_type を見て GBDT / NN 経路を自動選択する。

    GBDT:
        lambdarank + (optional) binary classifier + calibrator + combo calibrators
    NN:
        RaceModel (eval mode) + horse/race feature column lists

    旧モデル (meta.json に model_type 無し) は "gbdt" として扱う。
    torch がインストールされていない環境では NN モデルのロードのみ失敗する
    (ImportError)。GBDT 経路は torch 不要。
    """
    if not path.is_dir():
        # path が model.txt 直接指定なら lambdarank のみで返す (後方互換)
        return ModelBundle(
            model_type="gbdt",
            model_dir=path.parent,
            meta={},
            feature_columns=list(FEATURE_COLUMNS),
            lambdarank=lgb.Booster(model_file=str(path)),
        )

    meta_path = path / "meta.json"
    meta: dict = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    model_type = meta.get("model_type", "gbdt")
    feature_columns: list[str] = meta.get("feature_columns", list(FEATURE_COLUMNS))

    # place_calibrator.pkl は GBDT / NN 共通の optional artifact。
    place_calibrator = _load_place_calibrator(path)

    if model_type == "nn":
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
            nn_gbdt_bundle=nn_artifacts["nn_gbdt_bundle"],
            temperature_scaler=nn_artifacts["temperature_scaler"],
            combo_calibrators=nn_artifacts["combo_calibrators"],
            nn_calibrator=nn_artifacts.get("nn_calibrator"),
            place_calibrator=place_calibrator,
        )

    gbdt_artifacts = load_gbdt_artifacts(path)
    return ModelBundle(
        model_type="gbdt",
        model_dir=path,
        meta=meta,
        feature_columns=feature_columns,
        lambdarank=gbdt_artifacts["lambdarank"],
        binary=gbdt_artifacts["binary"],
        calibrator=gbdt_artifacts["calibrator"],
        combo_calibrators=gbdt_artifacts["combo_calibrators"],
        temperature_scaler=gbdt_artifacts["temperature_scaler"],
        place_calibrator=place_calibrator,
    )


def _load_place_calibrator(path: Path) -> IsotonicCalibrator | None:
    """Load place_calibrator.pkl from a model dir if present (optional)."""
    from ai._registry_gbdt import _pickle_load  # noqa: PLC0415 — shared unpickler

    cal_path = path / "place_calibrator.pkl"
    if not cal_path.exists():
        return None
    with cal_path.open("rb") as f:
        return _pickle_load(f)


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
