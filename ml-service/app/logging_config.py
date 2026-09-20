"""
app/logging_config.py

Structured logging configuration for the RealTimeGuard inference service.

Call ``configure_logging()`` once during application startup (in the
lifespan handler) before any other module emits log messages.

Design decisions:
- Uses Python's standard ``logging`` module (no third-party dependency).
- Format mirrors the Phase 1 training pipeline format for consistency.
- Respects the LOG_LEVEL setting so DEBUG output can be toggled via env var.
- Forces UTF-8 encoding on Windows where stdout may default to cp1252,
  which cannot encode box-drawing characters (─ →) used in messages.
"""

from __future__ import annotations

import io
import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """
    Configure the root logger for the inference service.

    Should be called exactly once, at application startup.

    Parameters
    ----------
    level : str
        Logging level string (e.g. "INFO", "DEBUG", "WARNING").
        Validated upstream by ``Settings.validate_log_level``.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # ------------------------------------------------------------------
    # Force UTF-8 on Windows so Unicode chars in log messages don't
    # raise UnicodeEncodeError on cp1252 terminals.
    # ------------------------------------------------------------------
    try:
        utf8_stream = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", line_buffering=True
        )
    except AttributeError:
        # .buffer is unavailable in some environments (pytest capture)
        utf8_stream = sys.stdout

    handler = logging.StreamHandler(utf8_stream)
    handler.setLevel(numeric_level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root_logger = logging.getLogger()
    # Remove any handlers already attached (e.g. from uvicorn's default setup)
    # before adding ours to avoid duplicate output.
    root_logger.handlers.clear()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(handler)

    # Suppress noisy third-party loggers at WARNING unless DEBUG is requested
    if numeric_level > logging.DEBUG:
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("lightgbm").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
