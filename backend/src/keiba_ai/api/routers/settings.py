"""Settings endpoints: GET /api/settings, PUT /api/settings."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from keiba_ai.api.deps import get_settings_store
from keiba_ai.api.schemas import SettingsResponse, SettingsUpdate
from keiba_ai.core.settings_store import SettingsStore

router = APIRouter()


def _dict_to_response(data: dict) -> SettingsResponse:
    return SettingsResponse(
        user_agent=data.get("user_agent", ""),
        rate_min_seconds=float(data.get("rate_min_seconds", 3.0)),
        rate_max_seconds=float(data.get("rate_max_seconds", 6.0)),
        night_min_seconds=float(data.get("night_min_seconds", 5.0)),
        win_ev_threshold=float(data.get("win_ev_threshold", 1.1)),
        place_ev_threshold=float(data.get("place_ev_threshold", 1.05)),
        scraper_stopped=bool(data.get("scraper_stopped", False)),
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
