"""確率モデルの役割 — 一覧での表示と、削除からの保護。

「使用中のモデル」が 2 種類になった:
  * active (model_runs.is_active) — どの馬・買い目を選ぶか
  * 確率モデル (settings.probability_model_path) — 複勝の確信度 / 連系の確率

住んでいる場所が DB と設定ファイルに分かれているので、片方しか見ないと
「確率モデルを消せてしまい、設定は存在しないパスを指したまま残る」ことになる。
その状態では推論側が警告ログを出して黙って旧挙動に戻るため、画面上は何も
起きていないように見えて複勝の絞り込みと連系の確率が無効化される。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.deps import get_settings_store
from core.settings_store import SettingsStore, is_probability_model


class TestIsProbabilityModel:
    def test_matches_on_basename(self):
        """パス表記の差 (Windows / WSL、絶対 / 相対) を跨いで一致すること。"""
        settings = {"probability_model_path": "models/20260101T000000-nn"}
        assert is_probability_model("/mnt/c/keiba/data/models/20260101T000000-nn", settings)
        assert is_probability_model(r"C:\keiba\data\models\20260101T000000-nn", settings)

    def test_different_model_does_not_match(self):
        settings = {"probability_model_path": "models/20260101T000000-nn"}
        assert not is_probability_model("models/20269999T999999-nn", settings)

    def test_unset_means_no_probability_model(self):
        assert not is_probability_model("models/anything-nn", {})
        assert not is_probability_model("models/anything-nn", {"probability_model_path": None})


def _seed_model(session, path: str, is_active: int = 0) -> int:
    from db.models.model_run import ModelRun
    run = ModelRun(
        created_at="2026-01-01T00:00:00", model_path=path,
        train_range=None, valid_range=None, params_json=None,
        metrics_json=None, is_active=is_active, model_type="nn",
    )
    session.add(run)
    session.flush()
    return run.id


def test_probability_model_cannot_be_deleted(app_with_temp_db: FastAPI, tmp_path: Path):
    from core.paths import db_path
    from db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        model_id = _seed_model(session, "models/20260827T140017-nn")

    store = SettingsStore(tmp_path / "settings.json")
    store.save({"probability_model_path": "models/20260827T140017-nn"})
    app_with_temp_db.dependency_overrides[get_settings_store] = lambda: store
    try:
        with TestClient(app_with_temp_db) as client:
            resp = client.delete(f"/api/models/{model_id}")
    finally:
        app_with_temp_db.dependency_overrides.pop(get_settings_store, None)

    assert resp.status_code == 409
    assert "確率モデル" in resp.json()["detail"]


def test_model_list_marks_the_probability_model(app_with_temp_db: FastAPI, tmp_path: Path):
    from core.paths import db_path
    from db.session import make_engine, session_scope

    engine = make_engine(db_path())
    with session_scope(engine) as session:
        prob_id = _seed_model(session, "models/20260827T140017-nn")
        other_id = _seed_model(session, "models/20260613T114817-nn")

    store = SettingsStore(tmp_path / "settings.json")
    store.save({"probability_model_path": "models/20260827T140017-nn"})
    app_with_temp_db.dependency_overrides[get_settings_store] = lambda: store
    try:
        with TestClient(app_with_temp_db) as client:
            rows = {m["id"]: m for m in client.get("/api/models").json()}
    finally:
        app_with_temp_db.dependency_overrides.pop(get_settings_store, None)

    assert rows[prob_id]["is_probability_model"] is True
    assert rows[other_id]["is_probability_model"] is False
