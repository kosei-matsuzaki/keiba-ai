"""Metrics endpoints: summary and timeseries."""

from __future__ import annotations

import contextlib
import json
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from keiba_ai.api.deps import get_session
from keiba_ai.api.schemas import MetricsSummary, MetricsTimeseries, TimeseriesPoint
from keiba_ai.db.models.model_run import ModelRun

router = APIRouter()


@router.get("/metrics/summary", response_model=MetricsSummary)
def get_metrics_summary(
    session: Annotated[Session, Depends(get_session)],
    range: str = "30d",  # noqa: A002 — parameter name matches API contract
) -> MetricsSummary:
    # Return metrics from the currently active model run
    active_run = session.scalars(
        select(ModelRun).where(ModelRun.is_active == 1).limit(1)
    ).first()
    if active_run is None:
        # Fall back to the most recent run
        active_run = session.scalars(
            select(ModelRun).order_by(ModelRun.created_at.desc()).limit(1)
        ).first()
    if active_run is None:
        return MetricsSummary(
            ndcg1=None,
            ndcg3=None,
            top1_hit=None,
            place_hit=None,
            payback_win=None,
            n_races=None,
            model_id=None,
        )

    metrics: dict = {}
    if active_run.metrics_json:
        with contextlib.suppress(json.JSONDecodeError):
            metrics = json.loads(active_run.metrics_json)

    return MetricsSummary(
        ndcg1=metrics.get("valid_ndcg1"),
        ndcg3=metrics.get("valid_ndcg3"),
        top1_hit=metrics.get("top1_hit"),
        place_hit=metrics.get("place_hit"),
        payback_win=metrics.get("payback_win"),
        n_races=metrics.get("n_races"),
        model_id=active_run.id,
    )


@router.get("/metrics/timeseries", response_model=MetricsTimeseries)
def get_metrics_timeseries(
    session: Annotated[Session, Depends(get_session)],
    metric: str = "ndcg3",
    range: str = "180d",  # noqa: A002
) -> MetricsTimeseries:
    # Map public metric name to the key in metrics_json
    _key_map = {
        "ndcg1": "valid_ndcg1",
        "ndcg3": "valid_ndcg3",
        "top1_hit": "top1_hit",
        "place_hit": "place_hit",
        "payback_win": "payback_win",
    }
    json_key = _key_map.get(metric, metric)

    runs = session.scalars(
        select(ModelRun).order_by(ModelRun.created_at).limit(100)
    ).all()

    points: list[TimeseriesPoint] = []
    for run in runs:
        value: float | None = None
        if run.metrics_json:
            with contextlib.suppress(json.JSONDecodeError, ValueError):
                m = json.loads(run.metrics_json)
                raw = m.get(json_key)
                if raw is not None:
                    value = float(raw)
        # Use created_at date portion as the x-axis label
        date_str = run.created_at[:10] if run.created_at else ""
        points.append(TimeseriesPoint(date=date_str, value=value))

    return MetricsTimeseries(metric=metric, points=points)
