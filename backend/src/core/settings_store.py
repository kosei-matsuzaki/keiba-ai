"""User-facing settings persisted as JSON in the data directory."""

from __future__ import annotations

import json
from pathlib import Path

from core.bet_types import DEFAULT_ENABLED_BET_TYPES
from core.paths import data_dir

_DEFAULTS: dict = {
    "user_agent": (
        "Mozilla/5.0 (compatible; keiba-ai-research/0.1; personal research only; "
        "contact: your-email@example.com)"
    ),
    "rate_min_seconds": 3.0,
    "rate_max_seconds": 6.0,
    "night_min_seconds": 5.0,
    "win_ev_threshold": 1.1,
    # 単勝は EV 条件ではなく「モデル 1 位を買う」ルール。これはそのオッズ下限。
    # (較正済み確率のもとでは EV フィルタが回収率を 0.931 → 0.698 まで落とすため)
    "win_min_odds": 1.1,
    "scraper_stopped": False,
    # 賭け金の設定 (定額)。1 レースに使う上限と 1 点あたりの額だけで決まる。
    "race_budget": 5_000,
    "stake_unit": 100,
    # 券種ごとの 1 点あたり金額。回収率の推定が確かな単複を厚く、信頼区間が
    # 0.01〜2.6 と測定不能な連系を薄く。総合回収率は券種別回収率の賭け金加重平均
    # なので、配分はモデルを変えずに効く (docs/ai-model.md「推奨ベットルール」)。
    "stake_units": {
        "単勝": 500,
        "複勝": 500,
        "馬連": 100,
        "ワイド": 100,
        "馬単": 100,
        "三連複": 100,
        "三連単": 100,
    },
    "enabled_bet_types": list(DEFAULT_ENABLED_BET_TYPES),
}

# 旧 Kelly 設定 (bankroll × max_stake_per_race_pct) から race_budget を復元する。
_LEGACY_KELLY_KEYS = ("bankroll", "kelly_fraction", "max_stake_per_race_pct")


def _migrate_legacy(data: dict) -> dict:
    """保存済みの旧 Kelly 設定を新しい定額設定に読み替える。

    race_budget が無く bankroll と max_stake_per_race_pct がある場合、
    「1 レースに使っていた上限額」= bankroll × 割合 を引き継ぐ。
    設定していた金額感を失わないための移行処理。
    """
    out = dict(data)
    if "race_budget" not in out:
        bankroll = out.get("bankroll")
        pct = out.get("max_stake_per_race_pct")
        if isinstance(bankroll, (int, float)) and isinstance(pct, (int, float)):
            budget = int(round(float(bankroll) * float(pct)))
            if budget >= 100:
                out["race_budget"] = budget
    for key in _LEGACY_KELLY_KEYS:
        out.pop(key, None)
    return out


class SettingsStore:
    """Load and persist user-editable settings as JSON."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (data_dir() / "settings.json")

    def load(self) -> dict:
        if not self._path.exists():
            return dict(_DEFAULTS)
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            # Fill any missing keys from defaults (forward-compatibility)
            merged = dict(_DEFAULTS)
            merged.update(_migrate_legacy(data))
            return merged
        except (json.JSONDecodeError, OSError):
            return dict(_DEFAULTS)

    def save(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
