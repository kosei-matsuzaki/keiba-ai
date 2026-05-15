"""NN (PyTorch) model artifact I/O (internal).

registry.py から呼ばれる低レベル NN 保存/読み込みプリミティブ。
公開 API ではない — 使う側は ai.registry の save_nn_model / load_model_full
を経由すること。

torch は遅延 import — torch が入っていない環境でも、GBDT 経路は
このモジュール側に触れないので影響しない。
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ai._registry_gbdt import _pickle_load


def save_nn_artifacts(state_dict_path: Path, model_dir: Path) -> None:
    """model.pt を model_dir 直下に配置する。

    train_nn.py が直接 data/models/<ts>-nn/model.pt に書き込むケースが多いので、
    source と target が同じ場所のときは no-op。
    """
    target_pt = model_dir / "model.pt"
    if state_dict_path.resolve() != target_pt.resolve():
        shutil.copy2(state_dict_path, target_pt)


def load_nn_artifacts(path: Path, meta: dict) -> dict[str, object]:
    """Read NN artifact files from a model directory.

    Returns:
        {nn_model, nn_horse_feature_cols, nn_race_feature_cols, temperature_scaler}

    Raises:
        ImportError: torch がインストールされていない環境では発生する。
    """
    import torch  # noqa: PLC0415 — intentional lazy import

    from ai.nn.model import RaceModel  # noqa: PLC0415

    params = meta.get("params", {})
    horse_feature_cols: list[str] = meta.get("horse_feature_cols", [])
    race_feature_cols: list[str] = meta.get("race_feature_cols", [])

    horse_feat_dim = len(horse_feature_cols)
    race_feat_dim = len(race_feature_cols)

    hidden_dim: int = params.get("hidden_dim", 64)
    embed_dim: int = params.get("embed_dim", 32)
    n_heads: int = params.get("n_heads", 4)

    race_model = RaceModel(
        horse_feat_dim=horse_feat_dim,
        race_feat_dim=race_feat_dim,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
    )

    state_dict = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
    race_model.load_state_dict(state_dict)
    race_model.eval()

    temperature_scaler = None
    temperature_scaler_path = path / "temperature_scaler.pkl"
    if temperature_scaler_path.exists():
        with temperature_scaler_path.open("rb") as f:
            temperature_scaler = _pickle_load(f)

    return {
        "nn_model": race_model,
        "nn_horse_feature_cols": horse_feature_cols,
        "nn_race_feature_cols": race_feature_cols,
        "temperature_scaler": temperature_scaler,
    }
