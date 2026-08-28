"""設定を null に戻せること。

`exclude_none=True` だと「明示的に null を送った」と「そのキーを送らなかった」が
区別できず、**値を未設定に戻せない**。確率モデルの割り当て解除（Models 画面の
「確率を解除」ボタン）がこれで黙って効かなかった。

設定ストアはテスト間で共有されるので、この 2 本は**自前の一時ストアに差し替えて**
実行する。共有のまま書き換えると、既定値を検証している他のテストを壊す。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.deps import get_settings_store
from core.settings_store import SettingsStore


@pytest.fixture
def isolated_client(app_with_temp_db: FastAPI, tmp_path: Path):
    store = SettingsStore(tmp_path / "settings.json")
    app_with_temp_db.dependency_overrides[get_settings_store] = lambda: store
    try:
        with TestClient(app_with_temp_db) as client:
            yield client
    finally:
        app_with_temp_db.dependency_overrides.pop(get_settings_store, None)


def test_probability_model_can_be_cleared(isolated_client: TestClient):
    isolated_client.put("/api/settings", json={"probability_model_path": "models/foo-nn"})
    assert (
        isolated_client.get("/api/settings").json()["probability_model_path"]
        == "models/foo-nn"
    )

    isolated_client.put("/api/settings", json={"probability_model_path": None})
    assert isolated_client.get("/api/settings").json()["probability_model_path"] is None


def test_omitted_keys_are_left_alone(isolated_client: TestClient):
    """送らなかったキーは変えない（null で潰さない）。"""
    isolated_client.put("/api/settings", json={"probability_model_path": "models/foo-nn"})
    isolated_client.put("/api/settings", json={"rate_min_seconds": 7.0})

    data = isolated_client.get("/api/settings").json()
    assert data["rate_min_seconds"] == 7.0
    assert data["probability_model_path"] == "models/foo-nn"
