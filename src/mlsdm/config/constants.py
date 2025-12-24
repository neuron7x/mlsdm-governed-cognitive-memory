"""Shared configuration constants."""

from __future__ import annotations

from mlsdm.config.runtime import RuntimeMode

DEFAULT_CONFIG_PATH = "config/default_config.yaml"

# Runtime modes that must fail-fast when config files are missing
STRICT_CONFIG_MODES: set[RuntimeMode] = {
    RuntimeMode.CLOUD_PROD,
    RuntimeMode.AGENT_API,
}

__all__ = ["DEFAULT_CONFIG_PATH", "STRICT_CONFIG_MODES"]
