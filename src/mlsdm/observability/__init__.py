"""Observability module for MLSDM Governed Cognitive Memory.

This module provides structured logging and monitoring capabilities
for the cognitive architecture system.
"""

from .aphasia_logging import (
    LOGGER_NAME as APHASIA_LOGGER_NAME,
)
from .aphasia_logging import (
    AphasiaLogEvent,
    log_aphasia_event,
)
from .aphasia_logging import (
    get_logger as get_aphasia_logger,
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
from .tracing import (
    TracerManager,
    TracingConfig,
    get_tracer,
    get_tracer_manager,
    initialize_tracing,
    shutdown_tracing,
    trace_aphasia_detection,
    trace_emergency_shutdown,
    trace_full_pipeline,
    trace_generate,
    trace_memory_retrieval,
    trace_moral_filter,
    trace_phase_transition,
    trace_process_event,
    traced,
    traced_async,
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
    # Tracing
    "TracerManager",
    "TracingConfig",
    "get_tracer",
    "get_tracer_manager",
    "initialize_tracing",
    "shutdown_tracing",
    "traced",
    "traced_async",
    "trace_generate",
    "trace_process_event",
    "trace_memory_retrieval",
    "trace_moral_filter",
    "trace_aphasia_detection",
    "trace_emergency_shutdown",
    "trace_phase_transition",
    "trace_full_pipeline",
]
