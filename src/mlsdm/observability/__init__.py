"""Observability module for MLSDM Governed Cognitive Memory.

This module provides structured logging and monitoring capabilities
for the cognitive architecture system.
"""

from .aphasia_logging import (
    LOGGER_NAME as APHASIA_LOGGER_NAME,
    AphasiaLogEvent,
    get_logger as get_aphasia_logger,
    log_aphasia_event,
)
from .aphasia_metrics import (
    AphasiaMetricsExporter,
    get_aphasia_metrics_exporter,
    reset_aphasia_metrics_exporter,
)
from .logger import (
    EventType,
    ObservabilityLogger,
    get_observability_logger,
)
from .metrics import (
    MetricsExporter,
    PhaseType,
    get_metrics_exporter,
)

__all__ = [
    # General observability
    "EventType",
    "ObservabilityLogger",
    "get_observability_logger",
    "MetricsExporter",
    "PhaseType",
    "get_metrics_exporter",
    # Aphasia-specific observability
    "APHASIA_LOGGER_NAME",
    "AphasiaLogEvent",
    "get_aphasia_logger",
    "log_aphasia_event",
    "AphasiaMetricsExporter",
    "get_aphasia_metrics_exporter",
    "reset_aphasia_metrics_exporter",
]
