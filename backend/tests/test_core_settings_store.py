"""Tests for core/settings_store.py — 賭け金設定 (定額) と旧 Kelly 設定の移行。

設定に残る賭け金の項目は「1 レースに使う上限 (race_budget)」だけ。1 点の額は
定数 (`ai.betting.strategy.STAKE_UNIT`)、点数は確信度が決めるので、どちらも
設定ではない。資金比率の Kelly も廃止したが、既存の settings.json には
bankroll / kelly_fraction / max_stake_per_race_pct が残っているので、
読み込み時に金額感を引き継いで読み替える。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from api.schemas import SettingsUpdate
from core.settings_store import _DEFAULTS, SettingsStore, resolve_betting_settings


class TestSettingsStoreDefaults:
    def test_race_budget_default(self, tmp_path: Path) -> None:
        store = SettingsStore(tmp_path / "settings.json")
        data = store.load()
        assert data["race_budget"] == 5_000

    def test_kelly_keys_are_gone(self, tmp_path: Path) -> None:
        store = SettingsStore(tmp_path / "settings.json")
        data = store.load()
        for key in ("bankroll", "kelly_fraction", "max_stake_per_race_pct"):
            assert key not in data

    def test_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        store = SettingsStore(tmp_path / "nope.json")
        data = store.load()
        assert data["race_budget"] == 5_000
        assert data["combo_min_hit_prob"] == _DEFAULTS["combo_min_hit_prob"]

    def test_legacy_settings_json_gets_new_fields_filled(self, tmp_path: Path) -> None:
        """新しいキーが無い古い settings.json でも既定値で埋まる。"""
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"user_agent": "Old/1.0"}), encoding="utf-8")

        data = SettingsStore(path).load()

        assert data["user_agent"] == "Old/1.0"
        assert data["race_budget"] == 5_000
        assert data["place_min_hit_prob"] == 0.60


class TestLegacyKellyMigration:
    def test_race_budget_is_derived_from_bankroll_and_pct(self, tmp_path: Path) -> None:
        """旧設定の「1 レースに使っていた上限額」を引き継ぐ。"""
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps(
                {
                    "bankroll": 200_000,
                    "kelly_fraction": 0.25,
                    "max_stake_per_race_pct": 0.05,
                }
            ),
            encoding="utf-8",
        )

        data = SettingsStore(path).load()

        # 200,000 × 0.05 = 10,000 円がこのレースの上限だった
        assert data["race_budget"] == 10_000
        assert "bankroll" not in data
        assert "kelly_fraction" not in data
        assert "max_stake_per_race_pct" not in data

    def test_explicit_race_budget_wins_over_legacy(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps(
                {"race_budget": 3_000, "bankroll": 200_000, "max_stake_per_race_pct": 0.05}
            ),
            encoding="utf-8",
        )

        assert SettingsStore(path).load()["race_budget"] == 3_000

    def test_tiny_legacy_budget_falls_back_to_default(self, tmp_path: Path) -> None:
        """100 円未満になる設定は 1 点も買えないので既定値に戻す。"""
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"bankroll": 1_000, "max_stake_per_race_pct": 0.01}),
            encoding="utf-8",
        )

        assert SettingsStore(path).load()["race_budget"] == 5_000


class TestSettingsStoreReadWrite:
    def test_write_and_read_race_budget(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        store = SettingsStore(path)
        data = store.load()
        data["race_budget"] = 20_000
        store.save(data)

        assert SettingsStore(path).load()["race_budget"] == 20_000

    def test_partial_update_preserves_other_keys(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        store = SettingsStore(path)

        data = store.load()
        data["race_budget"] = 8_000
        store.save(data)

        data2 = SettingsStore(path).load()
        data2["win_min_odds"] = 1.5
        SettingsStore(path).save(data2)

        final = SettingsStore(path).load()
        assert final["race_budget"] == 8_000
        assert final["win_min_odds"] == 1.5

    def test_corrupt_json_falls_back_to_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "settings.json"
        path.write_text("{ this is not json", encoding="utf-8")

        data = SettingsStore(path).load()
        assert data["race_budget"] == _DEFAULTS["race_budget"]


class TestSettingsApiValidation:
    def test_race_budget_ge_100_valid(self) -> None:
        assert SettingsUpdate(race_budget=100).race_budget == 100

    def test_race_budget_lt_100_raises(self) -> None:
        # 100 円未満だと 1 点も買えない
        with pytest.raises(ValidationError, match="race_budget"):
            SettingsUpdate(race_budget=99)

    def test_none_values_are_allowed(self) -> None:
        """未指定 (None) は「変更しない」を意味するので通す。"""
        su = SettingsUpdate()
        assert su.race_budget is None
        assert su.place_min_hit_prob is None


class TestRetiredSettings:
    """**設定から消したものは復活させない。**

    厚みは「1 点いくら」ではなく **何点買うか** (確信度が決める)、どの券種を
    買うかも確信度 (的中確率の下限) が決めるので、どちらも設定項目ではない。
    """

    def test_stake_and_bet_type_settings_are_gone(self, tmp_path: Path) -> None:
        data = SettingsStore(tmp_path / "settings.json").load()
        for key in (
            "stake_unit",
            "stake_units",
            "enabled_bet_types",
            "max_points_per_bet_type",
        ):
            assert key not in data

    def test_update_model_rejects_retired_keys(self) -> None:
        """廃止したキーを送っても黙って効いたことにしない (無視される)。"""
        su = SettingsUpdate(race_budget=3_000)
        assert not hasattr(su, "stake_unit")
        assert not hasattr(su, "enabled_bet_types")


class TestResolveBettingSettings:
    """買い方のパラメータを 1 箇所で解決すること。

    RACE 画面・シミュレーション・モデル評価・backtest CLI が同じ買い方を再現
    する必要がある。以前は各経路が同じ数行を書き写していて、シミュレーション
    だけ win_min_odds を渡し忘れ、Settings を変えても追従しなかった。
    """

    def test_defaults_come_from_the_settings_defaults(self) -> None:
        """既定値をリテラルで書き写さない (旧 0.30 のまま腐った実績がある)。"""
        resolved = resolve_betting_settings({})
        assert resolved.place_min_confidence == _DEFAULTS["place_min_hit_prob"]
        assert resolved.win_min_odds == _DEFAULTS["win_min_odds"]
        assert resolved.combo_min_hit_prob == _DEFAULTS["combo_min_hit_prob"]
        assert resolved.probability_model_path is None

    def test_reads_the_current_key_name_not_the_retired_one(self) -> None:
        """複勝の下限は place_min_hit_prob。旧 place_min_confidence は見ない。"""
        resolved = resolve_betting_settings(
            {"place_min_hit_prob": 0.72, "place_min_confidence": 0.30}
        )
        assert resolved.place_min_confidence == 0.72

    def test_combo_floors_fall_back_when_the_value_is_not_a_dict(self) -> None:
        resolved = resolve_betting_settings({"combo_min_hit_prob": None})
        assert resolved.combo_min_hit_prob == _DEFAULTS["combo_min_hit_prob"]
