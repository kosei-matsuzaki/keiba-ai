"""NN (PyTorch) model artifact I/O (internal).

registry.py から呼ばれる低レベル NN 保存/読み込みプリミティブ。
公開 API ではない — 使う側は ai.model.registry の save_nn_model / load_model_full
を経由すること。

torch は遅延 import — torch が入っていない環境 (scraper / ingest のみ) でも、
このモジュールに触れなければ影響しない。
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ai.model._pickle_compat import legacy_pickle_load as _pickle_load


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
         nn_preprocessor, temperature_scaler}

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

    # odds-at-scoring head + per-race 履歴の再構築情報 (dimensions)。
    # odds_feat_dim=0 は exclude-odds (ability-only)、history_feat_dim=0 は履歴なし。
    odds_feat_dim = int(meta.get("odds_feat_dim", 0))
    history_feat_dim = int(meta.get("history_feat_dim", 0))

    from ai.model.net import RaceTransformerModel  # noqa: PLC0415

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
        history_feat_dim=history_feat_dim,
        odds_feat_dim=odds_feat_dim,
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
        from ai.model.preprocess import NNPreprocessor  # noqa: PLC0415
        nn_preprocessor = NNPreprocessor.load(preprocessor_path)

    # per-race 履歴の正規化器 (mean/std)。推論時に学習時と同じ標準化を行う。
    nn_history_norm = None
    history_norm_path = path / "history_norm.pkl"
    if history_feat_dim > 0 and history_norm_path.exists():
        with history_norm_path.open("rb") as f:
            hn = _pickle_load(f)
        nn_history_norm = (hn["mean"], hn["std"])
    history_meta: dict = meta.get("history_meta", {}) or {}

    # B1: 履歴トークンの speed_fig を推論で再構築するための train-fit par テーブル。
    # meta の has_speed_figure が真かつ artifact が在るときのみロード。
    nn_speed_model = None
    speed_path = path / "speed_figure.pkl"
    if meta.get("has_speed_figure") and speed_path.exists():
        with speed_path.open("rb") as f:
            nn_speed_model = _pickle_load(f)

    return {
        "nn_model": race_model,
        "nn_horse_feature_cols": horse_feature_cols,
        "nn_race_feature_cols": race_feature_cols,
        "nn_preprocessor": nn_preprocessor,
        "temperature_scaler": temperature_scaler,
        "nn_odds_feature_cols": meta.get("odds_feature_cols", []) if odds_feat_dim > 0 else [],
        "nn_history_norm": nn_history_norm,
        "nn_history_max_len": int(history_meta.get("max_len", 0)),
        "nn_history_feat_dim": history_feat_dim,
        "nn_speed_model": nn_speed_model,
    }
