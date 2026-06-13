"""target 構成 (per-race 履歴 + odds-at-scoring head) を persist=True で学習・保存。

ユーザー採用構成。ability(集約+履歴, odds除く) → value(odds は head)。seed 固定で
再現性を確保。保存物: model.pt / preprocessor.pkl / temperature_scaler.pkl /
history_norm.pkl / meta.json + model_runs 行 (is_active=0)。アクティブ化は別途。

使い方:
  UV_PROJECT_ENVIRONMENT=/tmp/keiba-linux-venv PYTHONPATH=src \
      uv run python scripts/train_active_target.py
"""

from __future__ import annotations

from lightning.pytorch import seed_everything

from ai.training.train_nn import train_nn

seed_everything(0, workers=True)
res = train_nn(
    train_end="2025-04-30",
    valid_months=6,
    test_months=6,
    loss="multi",
    monitor="valid_tansho_roi",
    device="cuda",
    max_epochs=50,
    early_stopping_patience=8,
    use_history=True,
    history_seq_len=15,
    use_odds_head=True,
    fit_temperature=True,  # 推論で win/place 確率に温度スケーリングを使う
    persist=True,
)
print("SAVED_MODEL_DIR:", res.get("model_dir"), flush=True)
print(
    "metrics:",
    {
        k: res.get(k)
        for k in ("test_tansho_roi", "test_fukusho_roi", "test_tansho_hit", "test_fukusho_hit")
    },
    flush=True,
)
