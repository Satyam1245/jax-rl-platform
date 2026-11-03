"""Controlled logging setup for the platform."""

from __future__ import annotations

import logging
import sys
from typing import Final

_LOGGER_NAME: Final = "rl_platform"
_LEVEL_NAMES: Final = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


def configure_logging(level: str | int = "INFO") -> logging.Logger:
    """Configure the platform logger without replacing application root handlers."""

    numeric_level = _resolve_level(level)
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(numeric_level)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(numeric_level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    return logger


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    """Return a named logger without configuring global logging implicitly."""

    return logging.getLogger(name)


def _resolve_level(level: str | int) -> int:
    if isinstance(level, bool):
        raise ValueError("Logging level must be a level name or integer")
    if isinstance(level, int):
        if level < 0:
            raise ValueError("Logging level integer must be non-negative")
        return level
    if not isinstance(level, str):
        raise ValueError("Logging level must be a level name or integer")

    name = level.strip().upper()
    if name not in _LEVEL_NAMES:
        supported = ", ".join(sorted(_LEVEL_NAMES))
        raise ValueError(f"Unsupported logging level {name!r}; use: {supported}")
    return logging.getLevelName(name)
