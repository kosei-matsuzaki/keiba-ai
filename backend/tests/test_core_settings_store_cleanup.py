"""settings_store — 廃止キーの掃除とモデルパスの解決。"""

from __future__ import annotations

import json

from core.settings_store import SettingsStore, resolve_model_path


class TestDropsRetiredKeys:
    def test_retired_keys_do_not_survive_a_load(self, tmp_path):
        """設定を削除しても保存済み JSON に残り続ける、という状態を作らない。

        win_ev_threshold / place_ev_threshold は実際にこれで死に設定になり、
        「画面に無いのにファイルにはある」状態が続いた。
        """
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({
            "win_ev_threshold": 1.1,
            "place_ev_threshold": 1.05,
            "rate_min_seconds": 4.0,
        }), encoding="utf-8")
        data = SettingsStore(path).load()
        assert "win_ev_threshold" not in data
        assert "place_ev_threshold" not in data
        assert data["rate_min_seconds"] == 4.0  # 生きているキーは残る

    def test_legacy_kelly_keys_still_migrate_before_being_dropped(self, tmp_path):
        """掃除は移行の**後**に走ること (順番を逆にすると金額感を失う)。"""
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({
            "bankroll": 200_000, "max_stake_per_race_pct": 0.05,
        }), encoding="utf-8")
        data = SettingsStore(path).load()
        assert data["race_budget"] == 10_000     # 200,000 × 0.05 を引き継ぐ
        assert "bankroll" not in data


class TestResolveModelPath:
    def test_relative_path_is_resolved_against_the_data_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("KEIBA_DATA_DIR", str(tmp_path))
        got = resolve_model_path("models/20260101T000000-nn")
        assert got is not None
        assert got.is_absolute()
        assert got.parts[-2:] == ("models", "20260101T000000-nn")

    def test_absolute_path_is_left_alone(self, tmp_path):
        p = tmp_path / "elsewhere" / "model-nn"
        assert resolve_model_path(str(p)) == p

    def test_empty_means_not_configured(self):
        assert resolve_model_path(None) is None
        assert resolve_model_path("") is None
