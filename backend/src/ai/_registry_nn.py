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
        {nn_model, nn_horse_feature_cols, nn_race_feature_cols,
         nn_preprocessor, temperature_scaler, combo_calibrators}

    Raises:
        ImportError: torch がインストールされていない環境では発生する。
    """
    import torch  # noqa: PLC0415 — intentional lazy import

    params = meta.get("params", {})
    horse_feature_cols: list[str] = meta.get("horse_feature_cols", [])
    race_feature_cols: list[str] = meta.get("race_feature_cols", [])

    horse_feat_dim = len(horse_feature_cols)
    race_feat_dim = len(race_feature_cols)

    hidden_dim: int = params.get("hidden_dim", 64)
    embed_dim: int = params.get("embed_dim", 32)
    n_heads: int = params.get("n_heads", 4)

    arch_version = int(meta.get("arch_version", 1))
    if arch_version >= 2:
        from ai.nn.model import RaceTransformerModel  # noqa: PLC0415

        cat_meta: dict = meta.get("cat_metadata", {}) or {}
        race_model: torch.nn.Module = RaceTransformerModel(
            horse_feat_dim=horse_feat_dim,
            race_feat_dim=race_feat_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            horse_cat_positions=list(cat_meta.get("horse_cat_positions", [])),
            horse_cat_cardinalities=list(cat_meta.get("horse_cat_cardinalities", [])),
            race_cat_positions=list(cat_meta.get("race_cat_positions", [])),
            race_cat_cardinalities=list(cat_meta.get("race_cat_cardinalities", [])),
            cat_embed_dim=int(params.get("cat_embed_dim", 4)),
            n_transformer_layers=int(params.get("n_transformer_layers", 2)),
        )
    else:
        from ai.nn.model import RaceModel  # noqa: PLC0415

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

    nn_preprocessor = None
    preprocessor_path = path / "preprocessor.pkl"
    if preprocessor_path.exists():
        from ai.nn.preprocess import NNPreprocessor  # noqa: PLC0415
        nn_preprocessor = NNPreprocessor.load(preprocessor_path)

    combo_calibrators = None
    combo_cal_path = path / "combo_calibrators.pkl"
    if combo_cal_path.exists():
        with combo_cal_path.open("rb") as f:
            combo_calibrators = _pickle_load(f)

    # GBDT stacking: if the NN was trained with --gbdt-model-path, load that
    # GBDT bundle so the inference path can augment incoming frames with the
    # same gbdt_* columns the NN saw at train time.
    nn_gbdt_bundle = None
    gbdt_path_str = meta.get("gbdt_model_path")
    if gbdt_path_str:
        gbdt_path = Path(gbdt_path_str)
        if gbdt_path.exists():
            from ai.registry import load_model_full  # noqa: PLC0415 — lazy to break cycle
            nn_gbdt_bundle = load_model_full(gbdt_path)

    return {
        "nn_model": race_model,
        "nn_horse_feature_cols": horse_feature_cols,
        "nn_race_feature_cols": race_feature_cols,
        "nn_preprocessor": nn_preprocessor,
        "temperature_scaler": temperature_scaler,
        "combo_calibrators": combo_calibrators,
        "nn_gbdt_bundle": nn_gbdt_bundle,
    }
