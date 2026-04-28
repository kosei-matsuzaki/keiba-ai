"""POST /api/internal/shutdown — Tauri-initiated graceful shutdown."""

from __future__ import annotations

import os
import signal

from fastapi import APIRouter, BackgroundTasks

router = APIRouter()


def _shutdown() -> None:
    """Send SIGTERM to our own process; fall back to SIGINT or os._exit on Windows."""
    pid = os.getpid()
    try:
        os.kill(pid, signal.SIGTERM)
    except (AttributeError, OSError):
        # SIGTERM is not available on Windows in all contexts
        try:
            os.kill(pid, signal.SIGINT)
        except (AttributeError, OSError):
            os._exit(0)


@router.post("/internal/shutdown", status_code=200)
def shutdown(background_tasks: BackgroundTasks) -> dict:
    """Return 200 immediately, then kill the process in the background."""
    background_tasks.add_task(_shutdown)
    return {"ok": True}
