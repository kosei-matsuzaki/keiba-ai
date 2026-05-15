"""Tests for core/settings_store.py — focusing on the bankroll / Kelly fields
added in Issue #126."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.settings_store import _DEFAULTS, SettingsStore


class TestSettingsStoreDefaults:
    def test_bankroll_default(self, tmp_path: Path) -> None:
        store = SettingsStore(tmp_path / "settings.json")
        data = store.load()
        assert data["bankroll"] == 100_000

    def test_kelly_fraction_default(self, tmp_path: Path) -> None:
        store = SettingsStore(tmp_path / "settings.json")
        data = store.load()
        assert data["kelly_fraction"] == 0.25

    def test_max_stake_per_race_pct_default(self, tmp_path: Path) -> None:
        store = SettingsStore(tmp_path / "settings.json")
        data = store.load()
        assert data["max_stake_per_race_pct"] == 0.05

    def test_enabled_bet_types_default(self, tmp_path: Path) -> None:
        store = SettingsStore(tmp_path / "settings.json")
        data = store.load()
        assert data["enabled_bet_types"] == ["単勝", "複勝", "ワイド", "馬連"]

    def test_legacy_settings_json_gets_new_fields_filled(self, tmp_path: Path) -> None:
        """A settings.json that pre-dates Issue #126 is missing the new fields.
        load() must forward-fill them from _DEFAULTS."""
        old_settings = {
            "user_agent": "old-agent",
            "rate_min_seconds": 2.0,
            "rate_max_seconds": 5.0,
            "night_min_seconds": 4.0,
            "win_ev_threshold": 1.15,
            "place_ev_threshold": 1.08,
            "scraper_stopped": False,
        }
        settings_path = tmp_path / "settings.json"
        settings_path.write_text(json.dumps(old_settings), encoding="utf-8")

        store = SettingsStore(settings_path)
        data = store.load()

        # Legacy values preserved
        assert data["user_agent"] == "old-agent"
        assert data["rate_min_seconds"] == 2.0

        # New fields filled from defaults
        assert data["bankroll"] == 100_000
        assert data["kelly_fraction"] == 0.25
        assert data["max_stake_per_race_pct"] == 0.05
        assert data["enabled_bet_types"] == ["単勝", "複勝", "ワイド", "馬連"]


class TestSettingsStoreReadWrite:
    def test_write_and_read_bankroll(self, tmp_path: Path) -> None:
        store = SettingsStore(tmp_path / "settings.json")
        data = store.load()
        data["bankroll"] = 500_000
        store.save(data)

        reloaded = store.load()
        assert reloaded["bankroll"] == 500_000

    def test_write_and_read_kelly_fraction(self, tmp_path: Path) -> None:
        store = SettingsStore(tmp_path / "settings.json")
        data = store.load()
        data["kelly_fraction"] = 0.5
        store.save(data)

        reloaded = store.load()
        assert reloaded["kelly_fraction"] == 0.5

    def test_write_and_read_enabled_bet_types(self, tmp_path: Path) -> None:
        store = SettingsStore(tmp_path / "settings.json")
        data = store.load()
        data["enabled_bet_types"] = ["馬単", "三連複"]
        store.save(data)

        reloaded = store.load()
        assert reloaded["enabled_bet_types"] == ["馬単", "三連複"]

    def test_partial_update_preserves_other_keys(self, tmp_path: Path) -> None:
        store = SettingsStore(tmp_path / "settings.json")
        data = store.load()
        data["bankroll"] = 200_000
        store.save(data)

        data2 = store.load()
        data2["kelly_fraction"] = 0.1
        store.save(data2)

        final = store.load()
        assert final["bankroll"] == 200_000
        assert final["kelly_fraction"] == 0.1

    def test_corrupt_json_falls_back_to_defaults(self, tmp_path: Path) -> None:
        settings_path = tmp_path / "settings.json"
        settings_path.write_text("{ invalid json }", encoding="utf-8")
        store = SettingsStore(settings_path)
        data = store.load()
        assert data["bankroll"] == _DEFAULTS["bankroll"]


class TestSettingsApiValidation:
    """Validation of the new fields at the API schema layer (SettingsUpdate)."""

    def test_bankroll_ge_100_valid(self) -> None:
        from api.schemas import SettingsUpdate
        su = SettingsUpdate(bankroll=100)
        assert su.bankroll == 100

    def test_bankroll_lt_100_raises(self) -> None:
        from pydantic import ValidationError

        from api.schemas import SettingsUpdate
        with pytest.raises(ValidationError, match="bankroll"):
            SettingsUpdate(bankroll=99)

    def test_kelly_fraction_zero_raises(self) -> None:
        from pydantic import ValidationError

        from api.schemas import SettingsUpdate
        with pytest.raises(ValidationError, match="kelly_fraction"):
            SettingsUpdate(kelly_fraction=0.0)

    def test_kelly_fraction_gt_1_raises(self) -> None:
        from pydantic import ValidationError

        from api.schemas import SettingsUpdate
        with pytest.raises(ValidationError, match="kelly_fraction"):
            SettingsUpdate(kelly_fraction=1.01)

    def test_kelly_fraction_1_valid(self) -> None:
        from api.schemas import SettingsUpdate
        su = SettingsUpdate(kelly_fraction=1.0)
        assert su.kelly_fraction == 1.0

    def test_max_stake_zero_raises(self) -> None:
        from pydantic import ValidationError

        from api.schemas import SettingsUpdate
        with pytest.raises(ValidationError, match="max_stake_per_race_pct"):
            SettingsUpdate(max_stake_per_race_pct=0.0)

    def test_max_stake_gt_1_raises(self) -> None:
        from pydantic import ValidationError

        from api.schemas import SettingsUpdate
        with pytest.raises(ValidationError, match="max_stake_per_race_pct"):
            SettingsUpdate(max_stake_per_race_pct=1.5)

    def test_enabled_bet_types_invalid_raises(self) -> None:
        from pydantic import ValidationError

        from api.schemas import SettingsUpdate
        with pytest.raises(ValidationError, match="Unknown bet types"):
            SettingsUpdate(enabled_bet_types=["単勝", "invalid_type"])

    def test_enabled_bet_types_all_valid(self) -> None:
        from api.schemas import SettingsUpdate
        su = SettingsUpdate(enabled_bet_types=["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "三連複", "三連単"])
        assert len(su.enabled_bet_types) == 8

    def test_none_values_accepted(self) -> None:
        from api.schemas import SettingsUpdate
        su = SettingsUpdate(bankroll=None, kelly_fraction=None)
        assert su.bankroll is None
        assert su.kelly_fraction is None


class TestSettingsApiEndpoints:
    def test_get_settings_includes_new_fields(self, api_client) -> None:
        resp = api_client.get("/api/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert "bankroll" in data
        assert "kelly_fraction" in data
        assert "max_stake_per_race_pct" in data
        assert "enabled_bet_types" in data
        assert data["bankroll"] == 100_000
        assert data["kelly_fraction"] == 0.25
        assert data["max_stake_per_race_pct"] == 0.05
        assert data["enabled_bet_types"] == ["単勝", "複勝", "ワイド", "馬連"]

    def test_put_settings_updates_bankroll(self, api_client) -> None:
        resp = api_client.put("/api/settings", json={"bankroll": 200_000})
        assert resp.status_code == 200
        assert resp.json()["bankroll"] == 200_000

    def test_put_settings_updates_kelly_fraction(self, api_client) -> None:
        resp = api_client.put("/api/settings", json={"kelly_fraction": 0.5})
        assert resp.status_code == 200
        assert resp.json()["kelly_fraction"] == 0.5

    def test_put_settings_updates_enabled_bet_types(self, api_client) -> None:
        resp = api_client.put("/api/settings", json={"enabled_bet_types": ["馬単", "三連単"]})
        assert resp.status_code == 200
        assert resp.json()["enabled_bet_types"] == ["馬単", "三連単"]

    def test_put_settings_invalid_kelly_fraction(self, api_client) -> None:
        resp = api_client.put("/api/settings", json={"kelly_fraction": 0.0})
        assert resp.status_code == 422

    def test_put_settings_invalid_bankroll(self, api_client) -> None:
        resp = api_client.put("/api/settings", json={"bankroll": 50})
        assert resp.status_code == 422

    def test_put_settings_invalid_bet_type(self, api_client) -> None:
        resp = api_client.put("/api/settings", json={"enabled_bet_types": ["unknown"]})
        assert resp.status_code == 422

    def test_put_settings_persistence(self, api_client) -> None:
        api_client.put("/api/settings", json={"bankroll": 999_000})
        resp = api_client.get("/api/settings")
        assert resp.json()["bankroll"] == 999_000
