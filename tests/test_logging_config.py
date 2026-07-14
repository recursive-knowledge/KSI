"""Tests for the centralized logging configuration and flag/env precedence."""

from __future__ import annotations

import logging

from ksi.logging_config import configure_logging


def test_default_level_is_info(monkeypatch):
    monkeypatch.delenv("KSI_LOG_LEVEL", raising=False)
    logger = configure_logging(level=None)
    assert logger.level == logging.INFO


def test_env_sets_level(monkeypatch):
    monkeypatch.setenv("KSI_LOG_LEVEL", "WARNING")
    logger = configure_logging(level=None)
    assert logger.level == logging.WARNING


def test_explicit_flag_overrides_env(monkeypatch):
    """An explicit level (the --log-level flag) beats the env var."""
    monkeypatch.setenv("KSI_LOG_LEVEL", "WARNING")
    logger = configure_logging(level="DEBUG")
    assert logger.level == logging.DEBUG


def test_lowercase_and_numeric_levels(monkeypatch):
    monkeypatch.delenv("KSI_LOG_LEVEL", raising=False)
    assert configure_logging(level="debug").level == logging.DEBUG
    assert configure_logging(level="10").level == logging.DEBUG
    assert configure_logging(level=logging.ERROR).level == logging.ERROR


def test_invalid_level_falls_back_to_info(monkeypatch):
    monkeypatch.delenv("KSI_LOG_LEVEL", raising=False)
    assert configure_logging(level="NOT_A_LEVEL").level == logging.INFO
