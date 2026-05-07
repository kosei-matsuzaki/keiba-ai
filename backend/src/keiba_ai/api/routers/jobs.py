"""GET /api/jobs/{job_id} and GET /api/jobs — job status endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from keiba_ai.api.deps import get_job_registry
from keiba_ai.api.jobs import JobRegistry
from keiba_ai.api.schemas import JobInfoSchema

router = APIRouter()


@router.get("/jobs/{job_id}", response_model=JobInfoSchema)
def get_job(
    job_id: str,
    registry: Annotated[JobRegistry, Depends(get_job_registry)],
) -> JobInfoSchema:
    info = registry.get(job_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobInfoSchema(
        job_id=info.job_id,
        type=info.type,
        status=info.status,
        started_at=info.started_at,
        finished_at=info.finished_at,
        error=info.error,
        result=info.result,
    )


@router.get("/jobs", response_model=list[JobInfoSchema])
def list_jobs(
    registry: Annotated[JobRegistry, Depends(get_job_registry)],
) -> list[JobInfoSchema]:
    return [
        JobInfoSchema(
            job_id=info.job_id,
            type=info.type,
            status=info.status,
            started_at=info.started_at,
            finished_at=info.finished_at,
            error=info.error,
            result=info.result,
        )
        for info in registry.list()
    ]
