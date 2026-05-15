"""In-memory job registry for background tasks (ingest / train)."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass
class JobInfo:
    job_id: str
    type: str  # "ingest" | "train" | "simulation" など
    status: str  # "pending" | "running" | "completed" | "failed"
    started_at: str
    finished_at: str | None = None
    error: str | None = None
    # coro_factory が dict を返した場合に格納される結果 payload。
    # 例: simulation 完了時に {"run_id": 42} を入れて UI に渡す。
    result: dict[str, Any] | None = None


class JobRegistry:
    """Thread-safe in-memory registry for background jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobInfo] = {}

    def start(
        self,
        job_type: str,
        coro_factory: Callable[[], Coroutine[Any, Any, Any]],
    ) -> JobInfo:
        """Register and launch a background coroutine. Returns JobInfo immediately.

        coro_factory が dict (or None) を返すと、JobInfo.result に格納される。
        """
        job_id = str(uuid.uuid4())
        started_at = datetime.now(UTC).isoformat()
        info = JobInfo(
            job_id=job_id,
            type=job_type,
            status="running",
            started_at=started_at,
        )
        self._jobs[job_id] = info
        asyncio.create_task(self._run(info, coro_factory))
        return info

    async def _run(
        self,
        info: JobInfo,
        coro_factory: Callable[[], Coroutine[Any, Any, Any]],
    ) -> None:
        try:
            ret = await coro_factory()
            if isinstance(ret, dict):
                info.result = ret
            info.status = "completed"
        except Exception as exc:  # noqa: BLE001
            info.status = "failed"
            info.error = str(exc)
        finally:
            info.finished_at = datetime.now(UTC).isoformat()

    def get(self, job_id: str) -> JobInfo | None:
        return self._jobs.get(job_id)

    def list(self) -> list[JobInfo]:
        return list(self._jobs.values())

    # Convenience: return the latest running ingest job id, or None
    def current_ingest_job_id(self) -> str | None:
        for info in reversed(self.list()):
            if info.type == "ingest" and info.status == "running":
                return info.job_id
        return None
