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
    # 買い方に EV 条件は無い (2026-08-28 に連系の EV 閾値も廃止し、全券種で
    # 「確率の高い買い目から予算の限り買う」に統一)。これは単勝のオッズ下限。
    # (較正済み確率のもとでは EV フィルタが回収率を 0.931 → 0.698 まで落とすため)
    "win_min_odds": 1.1,
    "scraper_stopped": False,
    # 賭け金の設定 (定額)。1 レースに使う上限と 1 点あたりの額だけで決まる。
    "race_budget": 5_000,
    "stake_unit": 100,
    # 券種ごとの 1 点あたり金額。実測の回収率が高い単複 (0.931 / 0.885) を厚く、
    # 低い連系 (ワイド 0.880 〜 三連単 0.797) を薄く。総合回収率は券種別回収率の賭け金加重平均
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
    # 複勝の確信度フィルタ。proper scoring rule で学習した「確率専用モデル」の
    # ディレクトリを指定すると、AI の本命に対するそのモデルの **3 着内率** が
    # place_min_hit_prob 未満のレースでは複勝を買わなくなる。
    # 未設定 (None) なら従来どおり全レースで複勝を買う。
    #
    # **キー名を place_min_confidence から変えたのは、しきい値の目盛りが変わったため。**
    # 旧実装は 1 着確率で判定していて 0.30 = 買う割合 22% だったが、3 着内率では
    # 0.30 はほぼ全レースが通ってしまい、**設定を引き継ぐと黙ってフィルタが消える**。
    # 同じ選別度は 3 着内率 0.60 前後 (OOF 14,619 レースで買う割合 25% は 0.655)。
    # 成績は同等 (25% 選別で 0.904 → 0.907)、意味は「複勝が当たる確率」で通る。
    "probability_model_path": None,
    "place_min_hit_prob": 0.60,
    # 連系を 1 券種あたり何点まで買うか。**予算は上限であって使い切る目標ではない。**
    # 買い目は的中確率の高い順なので、深く買うほど当たりにくい買い目に金を足すことに
    # なる (実測 5,404 レース: 1 点目の的中率 0.155 → 10 点目 0.037)。上位 2 点で
    # 止めると、連系に使う金が 44% に減って回収率は 0.877 → 0.898。
    # 0 / None で無制限 (従来どおり予算まで買う)。
    "max_points_per_bet_type": 2,
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


def resolve_model_path(value: str | None) -> Path | None:
    """設定に入っているモデルパスを解決する。

    絶対パスならそのまま、相対パスなら ``data_dir()`` 基準。相対で保存できると
    設定ファイルが環境非依存になる (この repo は Windows と WSL の両方から
    使われるので、絶対パスを持たせると片方で壊れる)。
    """
    if not value:
        return None
    p = Path(value)
    return p if p.is_absolute() else (data_dir() / p)


def is_probability_model(model_path: str | None, settings: dict) -> bool:
    """``model_path`` が確率モデルとして設定されているか。

    保存済みのパスは環境によって表記が違う (Windows / WSL、絶対 / data_dir 相対)
    ので、`ai.model.registry` と同じく **basename で比べる**。
    """
    configured = settings.get("probability_model_path")
    if not configured or not model_path:
        return False
    return Path(configured).name == Path(model_path).name


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
            # **廃止したキーは落とす。** 設定を削除しても保存済み JSON には残り続け、
            # 「まだ効いていそうに見えるが実際は無視される」死に設定になる
            # (win_ev_threshold / place_ev_threshold で実際に起きた)。
            return {k: v for k, v in merged.items() if k in _DEFAULTS}
        except (json.JSONDecodeError, OSError):
            return dict(_DEFAULTS)

    def save(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
