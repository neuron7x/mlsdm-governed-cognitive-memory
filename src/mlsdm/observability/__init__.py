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
    RejectionReason,
    get_observability_logger,
    payload_scrubber,
    scrub_for_log,
)
from .metrics import (
    MetricsExporter,
    PhaseType,
    get_metrics_exporter,
)
from .tracing import (
    TracerManager,
    TracingConfig,
    add_span_attributes,
    get_tracer,
    get_tracer_manager,
    initialize_tracing,
    shutdown_tracing,
    trace_aphasia_detection,
    trace_aphasia_repair,
    trace_emergency_shutdown,
    trace_full_pipeline,
    trace_generate,
    trace_llm_call,
    trace_memory_retrieval,
    trace_moral_filter,
    trace_phase_transition,
    trace_process_event,
    trace_request,
    trace_speech_governance,
    traced,
    traced_async,
)

__all__ = [
    # Aphasia-specific observability
    "APHASIA_LOGGER_NAME",
    "AphasiaLogEvent",
    "AphasiaMetricsExporter",
    # General observability
    "EventType",
    "MetricsExporter",
    "ObservabilityLogger",
    "PhaseType",
    "RejectionReason",
    "get_aphasia_logger",
    "get_aphasia_metrics_exporter",
    "get_metrics_exporter",
    "get_observability_logger",
    "log_aphasia_event",
    "payload_scrubber",
    "reset_aphasia_metrics_exporter",
    "scrub_for_log",
    # Tracing
    "TracerManager",
    "TracingConfig",
    "add_span_attributes",
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
    "trace_aphasia_repair",
    "trace_emergency_shutdown",
    "trace_phase_transition",
    "trace_full_pipeline",
    "trace_llm_call",
    "trace_request",
    "trace_speech_governance",
]
