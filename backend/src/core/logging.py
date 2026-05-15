"""Logging configuration for keiba-ai."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    # Windows コンソールはデフォルト cp932 (Shift_JIS 系) で stdout を扱うため
    # em-dash (U+2014) 等の非 ASCII でログ書き込みが UnicodeEncodeError になる。
    # py3.7+ の TextIOWrapper.reconfigure で utf-8 + replace に切り替える。
    # stdout が非 TextIOWrapper の場合（リダイレクト等）は静かにスキップ。
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass

    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
