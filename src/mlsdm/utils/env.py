"""Environment variable parsing helpers."""

from __future__ import annotations

import logging
import os

_logger = logging.getLogger(__name__)


def get_env_float(*keys: str) -> float | None:
    """Parse the first available float environment variable.

    Ignores invalid or non-positive values, logging a warning for invalid input.
    """
    for key in keys:
        value = os.environ.get(key)
        if value is None:
            continue
        try:
            parsed = float(value)
        except ValueError:
            _logger.warning("Invalid float for %s=%r; ignoring.", key, value)
            continue
        if parsed <= 0:
            _logger.warning("Non-positive value for %s=%r; ignoring.", key, value)
            continue
        return parsed
    return None


def get_env_int(*keys: str) -> int | None:
    """Parse the first available integer environment variable.

    Ignores invalid or negative values, logging a warning for invalid input.
    """
    for key in keys:
        value = os.environ.get(key)
        if value is None:
            continue
        try:
            parsed = int(value)
        except ValueError:
            _logger.warning("Invalid int for %s=%r; ignoring.", key, value)
            continue
        if parsed < 0:
            _logger.warning("Negative value for %s=%r; ignoring.", key, value)
            continue
        return parsed
    return None
