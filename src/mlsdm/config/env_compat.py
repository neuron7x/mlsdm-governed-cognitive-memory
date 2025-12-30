"""
Environment Variable Compatibility Layer.

Provides backward compatibility for legacy environment variables by mapping them
to the canonical MLSDM_* namespace defined in runtime.py.

Legacy → Canonical Mapping:
- CONFIG_PATH → MLSDM_CONFIG_PATH (then used as CONFIG_PATH in runtime)
- LLM_BACKEND → MLSDM_LLM_BACKEND (then used as LLM_BACKEND in runtime)
- DISABLE_RATE_LIMIT → MLSDM_RATE_LIMIT_ENABLED (inverted)

This module ensures a single source of truth while maintaining backward compatibility.
"""

from __future__ import annotations

import os
import warnings


def apply_env_compat() -> None:
    """Apply environment variable compatibility mapping.

    Maps legacy environment variables to their canonical equivalents.
    Does NOT overwrite canonical variables if already set (canonical takes precedence).

    This function should be called early in the application startup, before
    loading runtime configuration.

    Side effects:
        Sets environment variables to ensure compatibility.

    Example:
        >>> import os
        >>> os.environ["CONFIG_PATH"] = "config/custom.yaml"
        >>> apply_env_compat()
        >>> # CONFIG_PATH is preserved for runtime.py consumption
        >>> os.environ["CONFIG_PATH"]
        'config/custom.yaml'
    """
    # Map CONFIG_PATH to canonical (it's already canonical in runtime.py)
    # Legacy: CONFIG_PATH → stays as CONFIG_PATH (already used in runtime.py)
    # This is already canonical, no mapping needed

    # Map LLM_BACKEND to canonical (it's already canonical in runtime.py)
    # Legacy: LLM_BACKEND → stays as LLM_BACKEND (already used in runtime.py)
    # This is already canonical, no mapping needed

    # Map DISABLE_RATE_LIMIT to canonical rate limit enabled
    # Legacy: DISABLE_RATE_LIMIT=1 → means rate limiting is disabled
    # Canonical: MLSDM_RATE_LIMIT_ENABLED=0 → means rate limiting is disabled
    if "DISABLE_RATE_LIMIT" in os.environ and "MLSDM_RATE_LIMIT_ENABLED" not in os.environ:
        disable_rate_limit = os.environ.get("DISABLE_RATE_LIMIT", "0")
        # If DISABLE_RATE_LIMIT is truthy, set MLSDM_RATE_LIMIT_ENABLED to 0
        if disable_rate_limit.lower() in ("1", "true", "yes", "on"):
            os.environ["MLSDM_RATE_LIMIT_ENABLED"] = "0"
        else:
            os.environ["MLSDM_RATE_LIMIT_ENABLED"] = "1"


def warn_if_legacy_vars_used() -> list[str]:
    """Check for legacy environment variables and warn if found.

    Returns:
        List of legacy variable names that are currently set.

    Example:
        >>> import os
        >>> os.environ["DISABLE_RATE_LIMIT"] = "1"
        >>> legacy = warn_if_legacy_vars_used()
        >>> "DISABLE_RATE_LIMIT" in legacy
        True
    """
    legacy_vars: list[str] = []

    # Check for legacy vars that have canonical alternatives
    if "DISABLE_RATE_LIMIT" in os.environ:
        legacy_vars.append("DISABLE_RATE_LIMIT")
        if "MLSDM_RATE_LIMIT_ENABLED" not in os.environ:
            warnings.warn(
                "DISABLE_RATE_LIMIT is deprecated. Use MLSDM_RATE_LIMIT_ENABLED=0 instead.",
                DeprecationWarning,
                stacklevel=2,
            )

    # Note: CONFIG_PATH and LLM_BACKEND are actually still canonical
    # They're read directly by runtime.py, so they don't need warnings

    return legacy_vars


def get_env_compat_info() -> dict[str, dict[str, str | None]]:
    """Get information about environment variable compatibility.

    Returns:
        Dictionary mapping legacy variable names to their status and canonical equivalents.

    Example:
        >>> info = get_env_compat_info()
        >>> "DISABLE_RATE_LIMIT" in info
        True
    """
    return {
        "DISABLE_RATE_LIMIT": {
            "canonical": "MLSDM_RATE_LIMIT_ENABLED",
            "current_value": os.environ.get("DISABLE_RATE_LIMIT"),
            "canonical_value": os.environ.get("MLSDM_RATE_LIMIT_ENABLED"),
            "note": "Set MLSDM_RATE_LIMIT_ENABLED=0 to disable rate limiting (replaces DISABLE_RATE_LIMIT=1)",
        },
        "CONFIG_PATH": {
            "canonical": "CONFIG_PATH",
            "current_value": os.environ.get("CONFIG_PATH"),
            "canonical_value": os.environ.get("CONFIG_PATH"),
            "note": "CONFIG_PATH is already canonical (used directly by runtime.py)",
        },
        "LLM_BACKEND": {
            "canonical": "LLM_BACKEND",
            "current_value": os.environ.get("LLM_BACKEND"),
            "canonical_value": os.environ.get("LLM_BACKEND"),
            "note": "LLM_BACKEND is already canonical (used directly by runtime.py)",
        },
    }
