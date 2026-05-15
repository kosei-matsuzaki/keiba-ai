"""Metrics endpoints: summary and timeseries."""

from __future__ import annotations

import contextlib
import json
import math
from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.deps import get_session
from api.schemas import MetricsSummary, MetricsTimeseries, TimeseriesPoint
from db.models.model_run import ModelRun

router = APIRouter()


def _pick_metric(metrics: dict, *keys: str) -> float | None:
    """Return the first non-null / non-NaN float found across the given keys.

    Order matters: callers list keys most-specific-first (e.g. valid_ then test_).
    Used so the dashboard can fall back to test_* metrics when valid is empty
    (Phase 1 で `--valid-months 0` 指定されたケース等) and to flatten model/
    baseline persisted layouts.
    """
    for key in keys:
        v = metrics.get(key)
        if v is None:
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            f = float(v)
            if not math.isnan(f):
                return f
    return None


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

    n_races_raw = metrics.get("n_races")
    n_races = int(n_races_raw) if isinstance(n_races_raw, (int, float)) else None

    return MetricsSummary(
        # valid_* が NaN になりやすい (--valid-months 0 で学習した場合) ので
        # test_* に fallback。evaluate.py --persist が走っていれば top-level の
        # ndcg* も入る可能性があるためそれも候補に含める。
        ndcg1=_pick_metric(metrics, "valid_ndcg1", "test_ndcg1", "ndcg1"),
        ndcg3=_pick_metric(metrics, "valid_ndcg3", "test_ndcg3", "ndcg3"),
        top1_hit=_pick_metric(metrics, "top1_hit"),
        place_hit=_pick_metric(metrics, "place_hit"),
        payback_win=_pick_metric(metrics, "payback_win"),
        n_races=n_races,
        model_id=active_run.id,
    )


@router.get("/metrics/timeseries", response_model=MetricsTimeseries)
def get_metrics_timeseries(
    session: Annotated[Session, Depends(get_session)],
    metric: str = "ndcg3",
    range: str = "180d",  # noqa: A002
) -> MetricsTimeseries:
    # Map public metric name → ordered fallback keys (valid → test → bare)
    _key_chain: dict[str, tuple[str, ...]] = {
        "ndcg1": ("valid_ndcg1", "test_ndcg1", "ndcg1"),
        "ndcg3": ("valid_ndcg3", "test_ndcg3", "ndcg3"),
        "top1_hit": ("top1_hit",),
        "place_hit": ("place_hit",),
        "payback_win": ("payback_win",),
        "payback_place": ("payback_place",),
    }
    keys = _key_chain.get(metric, (metric,))

    runs = session.scalars(
        select(ModelRun).order_by(ModelRun.created_at).limit(100)
    ).all()

    points: list[TimeseriesPoint] = []
    for run in runs:
        value: float | None = None
        if run.metrics_json:
            with contextlib.suppress(json.JSONDecodeError, ValueError):
                m = json.loads(run.metrics_json)
                value = _pick_metric(m, *keys)
        # Use created_at date portion as the x-axis label
        date_str = run.created_at[:10] if run.created_at else ""
        points.append(TimeseriesPoint(date=date_str, value=value))

    return MetricsTimeseries(metric=metric, points=points)
