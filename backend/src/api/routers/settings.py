"""Settings endpoints: GET /api/settings, PUT /api/settings."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from api.deps import get_settings_store
from api.schemas import SettingsResponse, SettingsUpdate
from core.bet_types import DEFAULT_ENABLED_BET_TYPES, supported_bet_types
from core.settings_store import SettingsStore

router = APIRouter()


def _dict_to_response(data: dict) -> SettingsResponse:
    return SettingsResponse(
        user_agent=data.get("user_agent", ""),
        rate_min_seconds=float(data.get("rate_min_seconds", 3.0)),
        rate_max_seconds=float(data.get("rate_max_seconds", 6.0)),
        night_min_seconds=float(data.get("night_min_seconds", 5.0)),
        win_min_odds=float(data.get("win_min_odds", 1.1)),
        scraper_stopped=bool(data.get("scraper_stopped", False)),
        race_budget=int(data.get("race_budget", 5_000)),
        stake_unit=int(data.get("stake_unit", 100)),
        stake_units={k: int(v) for k, v in (data.get("stake_units") or {}).items()},
        # 予測できない券種 (枠連) が設定に残っていても落とす。選べるのに何も
        # 起きない選択肢を残さないため (core.bet_types.supported_bet_types)。
        enabled_bet_types=supported_bet_types(
            data.get("enabled_bet_types", DEFAULT_ENABLED_BET_TYPES)
        ),
        probability_model_path=data.get("probability_model_path"),
        place_min_confidence=float(data.get("place_min_confidence", 0.30)),
    )


@router.get("/settings", response_model=SettingsResponse)
def get_settings(
    store: Annotated[SettingsStore, Depends(get_settings_store)],
) -> SettingsResponse:
    return _dict_to_response(store.load())


@router.put("/settings", response_model=SettingsResponse)
def put_settings(
    body: SettingsUpdate,
    store: Annotated[SettingsStore, Depends(get_settings_store)],
) -> SettingsResponse:
    data = store.load()
    update = body.model_dump(exclude_none=True)
    data.update(update)
    store.save(data)
    return _dict_to_response(data)
