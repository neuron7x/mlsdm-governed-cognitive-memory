"""Unified NPZ loading utilities with optional legacy pickle fallback."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.lib.npyio import NpzFile

logger = logging.getLogger(__name__)


def load_npz_arrays_safe(
    filepath: str,
    *,
    allow_legacy_pickle: bool,
    error_type: type[Exception] = ValueError,
    legacy_error_message: str | None = None,
    warning_message: str | None = None,
) -> "NpzFile":
    """Load NPZ arrays with optional legacy pickle fallback.

    Args:
        filepath: Path to the .npz file.
        allow_legacy_pickle: Allow pickle-based object arrays if detected.
        error_type: Exception type to raise when legacy pickle is refused.
        legacy_error_message: Custom error message when legacy pickle is blocked.
        warning_message: Custom warning message for legacy pickle fallback.

    Returns:
        Loaded NpzFile object.

    Raises:
        error_type: If legacy pickle detected and allow_legacy_pickle=False.
        ValueError: If file format is invalid.
    """
    default_error = (
        "Legacy pickle-based NPZ payload detected. "
        "Refusing to load without allow_legacy_pickle=True."
    )
    default_warning = (
        "Loading legacy pickle-based NPZ payload from %s. "
        "Consider re-saving to migrate to the safer format."
    )

    try:
        return np.load(filepath, allow_pickle=False)
    except ValueError as exc:
        if "Object arrays cannot be loaded when allow_pickle=False" not in str(exc):
            raise
        if not allow_legacy_pickle:
            message = legacy_error_message or default_error
            raise error_type(message) from exc

        message = warning_message or default_warning
        logger.warning(message, filepath)
        return np.load(filepath, allow_pickle=True)
