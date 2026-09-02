"""Settings endpoints: GET /api/settings, PUT /api/settings."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from api.deps import get_settings_store
from api.schemas import SettingsResponse, SettingsUpdate
from core.bet_types import DEFAULT_COMBO_MIN_HIT_PROB
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
        probability_model_path=data.get("probability_model_path"),
        # **キー名を間違えると黙って既定値が返る。** 以前ここが
        # place_min_confidence= (旧名) のままで、保存した値が GET / PUT の応答に
        # 出ず、画面が保存のたびに既定へ戻って見えていた。
        place_min_hit_prob=float(data.get("place_min_hit_prob", 0.60)),
        combo_min_hit_prob={
            k: float(v)
            for k, v in (
                data.get("combo_min_hit_prob") or DEFAULT_COMBO_MIN_HIT_PROB
            ).items()
        },
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
    # **exclude_unset**。exclude_none だと「明示的に null を送った」と「そのキーを
    # 送らなかった」が区別できず、値を **null に戻せない**。実際に
    # probability_model_path（確率モデルの割り当て解除）がこれで効かなかった。
    update = body.model_dump(exclude_unset=True)
    data.update(update)
    store.save(data)
    return _dict_to_response(data)
