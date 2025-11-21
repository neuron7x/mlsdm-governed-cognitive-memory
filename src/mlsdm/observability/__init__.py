"""Observability module for MLSDM Governed Cognitive Memory.

This module provides structured logging and monitoring capabilities
for the cognitive architecture system.
"""

from .logger import (
    EventType,
    ObservabilityLogger,
    get_observability_logger,
)

__all__ = [
    "EventType",
    "ObservabilityLogger",
    "get_observability_logger",
]
