"""テスト用アプリの設定ストアが tmp_path に隔離されていること。

1 つのテストが書いた設定が次のテストに漏れると、原因の分かりにくい失敗になる
（実際に「EV 閾値が 1.1 のはずが 1.2」で一度踏んだ）。隔離を環境変数のタイミング
任せにせず、fixture で明示的に向けている。
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI


def test_settings_store_lives_under_tmp_path(app_with_temp_db: FastAPI, tmp_path: Path):
    store_path = app_with_temp_db.state.settings_store._path
    assert tmp_path in Path(store_path).parents


def test_writes_do_not_touch_the_real_settings(app_with_temp_db: FastAPI, tmp_path: Path):
    """書き込んでもリポジトリの data/settings.json を汚さない。"""
    store = app_with_temp_db.state.settings_store
    store.save({**store.load(), "rate_min_seconds": 99.0})
    assert (tmp_path / "data" / "settings.json").exists()
    assert store.load()["rate_min_seconds"] == 99.0
