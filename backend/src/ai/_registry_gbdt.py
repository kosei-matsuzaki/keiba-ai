"""GBDT model artifact I/O (internal).

registry.py から呼ばれる低レベル GBDT 保存/読み込みプリミティブ。
公開 API ではない — 使う側は ai.registry の save_model / load_model_full
を経由すること。

含むもの:
  - _LegacyUnpickler / _pickle_load: keiba_ai.* 旧パス対応の pickle 読み込み
  - save_gbdt_artifacts: model.txt / binary.txt / *.pkl を書き出す
  - load_gbdt_artifacts: 同じファイル群を読み出し dict で返す
"""

from __future__ import annotations

import pickle
from pathlib import Path

import lightgbm as lgb

from ai.calibrate import ComboCalibrators, IsotonicCalibrator
from ai.temperature import TemperatureScaler


class _LegacyUnpickler(pickle.Unpickler):
    """`keiba_ai.*` 旧パスで pickle 化された artifact を、refactor 後の新パスへ
    透過的にリマップする Unpickler。再学習なしで旧 .pkl を読めるようにする。

    対象は calibrator.pkl / combo_calibrators.pkl / temperature_scaler.pkl 等。
    GBM 固有 (train/tune/pl_loss) はクラスを pickle しないので keiba_ai.ai → ai
    の単純な prefix 除去で十分。
    """

    def find_class(self, module: str, name: str):
        if module.startswith("keiba_ai."):
            module = module[len("keiba_ai."):]
        elif module == "keiba_ai":
            raise ImportError("keiba_ai は廃止済みパッケージです")
        return super().find_class(module, name)


def _pickle_load(fp) -> object:
    """旧 keiba_ai.* パス対応の pickle.load ラッパー。"""
    return _LegacyUnpickler(fp).load()


def save_gbdt_artifacts(
    model_dir: Path,
    model: lgb.Booster,
    binary_model: lgb.Booster | None,
    calibrator: IsotonicCalibrator | None,
    combo_calibrators: ComboCalibrators | None,
    temperature_scaler: TemperatureScaler | None,
) -> dict[str, bool | list[str]]:
    """Persist GBDT artifact files into model_dir.

    Returns:
        meta 用 has_* フラグの dict。呼び出し側が meta.json にマージする。
    """
    model.save_model(str(model_dir / "model.txt"))

    has_binary = binary_model is not None
    has_calibrator = calibrator is not None
    if has_binary:
        binary_model.save_model(str(model_dir / "binary.txt"))
    if has_calibrator:
        with (model_dir / "calibrator.pkl").open("wb") as f:
            pickle.dump(calibrator, f)

    has_combo_calibrators = (
        combo_calibrators is not None and len(combo_calibrators.fitted_bet_types) > 0
    )
    if has_combo_calibrators:
        with (model_dir / "combo_calibrators.pkl").open("wb") as f:
            pickle.dump(combo_calibrators, f)

    has_temperature_scaler = temperature_scaler is not None
    if has_temperature_scaler:
        with (model_dir / "temperature_scaler.pkl").open("wb") as f:
            pickle.dump(temperature_scaler, f)

    return {
        "has_binary_model": has_binary,
        "has_calibrator": has_calibrator,
        "has_combo_calibrators": has_combo_calibrators,
        "combo_calibrators_bet_types": (
            combo_calibrators.fitted_bet_types if has_combo_calibrators else []
        ),
        "has_temperature_scaler": has_temperature_scaler,
    }


def load_gbdt_artifacts(path: Path) -> dict[str, object]:
    """Read GBDT artifact files from a model directory.

    Returns:
        {lambdarank, binary, calibrator, combo_calibrators, temperature_scaler}
        — 存在しないファイルに対応するキーは None。
    """
    lambdarank = lgb.Booster(model_file=str(path / "model.txt"))

    binary_path = path / "binary.txt"
    binary = lgb.Booster(model_file=str(binary_path)) if binary_path.exists() else None

    calibrator = None
    calibrator_path = path / "calibrator.pkl"
    if calibrator_path.exists():
        with calibrator_path.open("rb") as f:
            calibrator = _pickle_load(f)

    combo_calibrators = None
    combo_cal_path = path / "combo_calibrators.pkl"
    if combo_cal_path.exists():
        with combo_cal_path.open("rb") as f:
            combo_calibrators = _pickle_load(f)

    temperature_scaler = None
    temperature_scaler_path = path / "temperature_scaler.pkl"
    if temperature_scaler_path.exists():
        with temperature_scaler_path.open("rb") as f:
            temperature_scaler = _pickle_load(f)

    return {
        "lambdarank": lambdarank,
        "binary": binary,
        "calibrator": calibrator,
        "combo_calibrators": combo_calibrators,
        "temperature_scaler": temperature_scaler,
    }
