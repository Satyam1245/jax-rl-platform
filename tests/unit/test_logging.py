"""Tests for logging initialization."""

import logging

import pytest

from rl_platform.utils.logging import configure_logging, get_logger


def test_configure_logging_initializes_platform_logger() -> None:
    logger = configure_logging("DEBUG")

    assert logger is get_logger("rl_platform")
    assert logger.level == logging.DEBUG
    assert logger.propagate is False
    assert logger.handlers
    assert logger.handlers[0].level == logging.DEBUG


def test_configure_logging_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match="Unsupported logging level"):
        configure_logging("TRACE")
