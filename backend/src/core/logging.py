"""Logging configuration for keiba-ai."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S"


def configure_logging(level: int = logging.INFO, *, log_name: str | None = None) -> None:
    """Configure root logging to stdout, plus an optional file handler.

    File logging is opt-in via the ``KEIBA_LOG_DIR`` env var. When set, a
    timestamped log file is written at ``<KEIBA_LOG_DIR>/<name>-<ts>.log`` so
    long-running CLI jobs (ingest 系) leave a persistent record under one
    directory instead of relying on manual shell redirection. ``name`` defaults
    to the invoked script's basename (e.g. ``ingest_odds``).
    """
    # Windows コンソールはデフォルト cp932 (Shift_JIS 系) で stdout を扱うため
    # em-dash (U+2014) 等の非 ASCII でログ書き込みが UnicodeEncodeError になる。
    # py3.7+ の TextIOWrapper.reconfigure で utf-8 + replace に切り替える。
    # stdout が非 TextIOWrapper の場合（リダイレクト等）は静かにスキップ。
    if hasattr(sys.stdout, "reconfigure"):
        with contextlib.suppress(AttributeError, OSError, ValueError):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format=_LOG_FORMAT,
        datefmt=_LOG_DATEFMT,
    )

    log_dir = os.getenv("KEIBA_LOG_DIR", "").strip()
    if log_dir:
        name = log_name or Path(sys.argv[0] or "keiba").stem or "keiba"
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        d = Path(log_dir)
        d.mkdir(parents=True, exist_ok=True)
        target = d / f"{name}-{ts}.log"
        root = logging.getLogger()
        root.setLevel(level)
        # 同一プロセスで二重 configure されても同じファイルへ重複追加しない。
        already = any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", None) == str(target.resolve())
            for h in root.handlers
        )
        if not already:
            handler = logging.FileHandler(target, encoding="utf-8")
            handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
            root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
