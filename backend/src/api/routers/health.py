"""GET /api/health — liveness check."""

from __future__ import annotations

from fastapi import APIRouter, Request

from api.schemas import HealthResponse
from core.paths import db_path

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def get_health(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version="0.1.0",
        db_path=str(db_path()),
    )
