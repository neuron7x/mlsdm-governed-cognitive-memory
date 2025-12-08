"""Test that observability modules work correctly without OpenTelemetry installed.

This test suite validates that the observability stack gracefully degrades
when OpenTelemetry is not available, ensuring that `import mlsdm` never fails
due to missing OTEL packages.
"""

import sys
from unittest.mock import patch


def test_import_without_otel():
    """Test that mlsdm can be imported when OpenTelemetry is not available.

    This simulates the scenario where OTEL packages are not installed
    by blocking the imports.
    """
    # Block opentelemetry imports by removing them from sys.modules
    # and preventing future imports
    otel_modules = [m for m in sys.modules if m.startswith("opentelemetry")]
    for module in otel_modules:
        sys.modules.pop(module, None)

    with patch.dict(sys.modules, {"opentelemetry": None, "opentelemetry.sdk": None}):
        # Force reload of observability modules
        import importlib

        import mlsdm.observability.logger
        import mlsdm.observability.tracing

        importlib.reload(mlsdm.observability.tracing)
        importlib.reload(mlsdm.observability.logger)

        # Verify OTEL_AVAILABLE is False
        assert not mlsdm.observability.tracing.OTEL_AVAILABLE
        assert not mlsdm.observability.logger.OTEL_AVAILABLE


def test_logger_works_without_otel():
    """Test that ObservabilityLogger works without OpenTelemetry."""
    from mlsdm.observability import EventType, get_observability_logger

    logger = get_observability_logger(logger_name="test_no_otel", console_output=False)

    # Should not raise any errors
    logger.info(EventType.SYSTEM_STARTUP, "Test message")
    logger.warn(EventType.SYSTEM_WARNING, "Warning message")
    logger.error(EventType.SYSTEM_ERROR, "Error message")


def test_tracer_no_op_without_otel():
    """Test that tracer provides no-op implementation without OpenTelemetry."""
    from mlsdm.observability import get_tracer, span

    tracer = get_tracer()

    # Should return a no-op tracer when OTEL not available
    # The type name should be _NoOpTracer when OTEL is not available
    # When OTEL is available, it should be Tracer
    assert tracer is not None

    # No-op span should work without errors
    with span("test_span", attr="value") as s:
        s.set_attribute("key", "value")
        s.add_event("test_event")


def test_trace_context_empty_without_otel():
    """Test that trace context returns empty strings without OpenTelemetry."""
    # Mock OTEL as unavailable
    import mlsdm.observability.logger as logger_mod
    from mlsdm.observability.logger import get_current_trace_context

    original_available = logger_mod.OTEL_AVAILABLE

    try:
        logger_mod.OTEL_AVAILABLE = False
        ctx = get_current_trace_context()

        # Should return empty strings
        assert ctx["trace_id"] == ""
        assert ctx["span_id"] == ""
    finally:
        logger_mod.OTEL_AVAILABLE = original_available


def test_tracer_manager_initialization_without_otel():
    """Test that TracerManager initializes correctly without OpenTelemetry."""
    # Mock OTEL as unavailable
    import mlsdm.observability.tracing as tracing_mod
    from mlsdm.observability import TracingConfig, get_tracer_manager

    original_available = tracing_mod.OTEL_AVAILABLE

    try:
        tracing_mod.OTEL_AVAILABLE = False

        # Reset the singleton to test fresh initialization
        tracing_mod.TracerManager.reset_instance()

        config = TracingConfig(enabled=True)
        manager = get_tracer_manager(config)
        manager.initialize()

        # Should succeed without errors
        assert manager._initialized

        # Tracer should be no-op
        tracer = manager.tracer
        assert tracer is not None
    finally:
        tracing_mod.OTEL_AVAILABLE = original_available
        tracing_mod.TracerManager.reset_instance()


def test_span_context_manager_without_otel():
    """Test that span context manager works without OpenTelemetry."""
    # Mock OTEL as unavailable
    import mlsdm.observability.tracing as tracing_mod
    from mlsdm.observability import get_tracer_manager

    original_available = tracing_mod.OTEL_AVAILABLE

    try:
        tracing_mod.OTEL_AVAILABLE = False
        tracing_mod.TracerManager.reset_instance()

        manager = get_tracer_manager()

        # Should not raise errors
        with manager.start_span("test_operation") as span:
            span.set_attribute("test", "value")
    finally:
        tracing_mod.OTEL_AVAILABLE = original_available
        tracing_mod.TracerManager.reset_instance()


def test_trace_context_filter_without_otel():
    """Test that TraceContextFilter works without OpenTelemetry."""
    import logging

    # Mock OTEL as unavailable
    import mlsdm.observability.logger as logger_mod
    from mlsdm.observability.logger import TraceContextFilter

    original_available = logger_mod.OTEL_AVAILABLE

    try:
        logger_mod.OTEL_AVAILABLE = False

        # Create a test logger and record
        test_logger = logging.getLogger("test_filter")
        test_filter = TraceContextFilter()

        # Create a log record
        record = test_logger.makeRecord(
            "test_filter",
            logging.INFO,
            __file__,
            1,
            "Test message",
            (),
            None,
        )

        # Filter should add empty trace context
        assert test_filter.filter(record)
        assert hasattr(record, "trace_id")
        assert hasattr(record, "span_id")
        assert record.trace_id == ""
        assert record.span_id == ""
    finally:
        logger_mod.OTEL_AVAILABLE = original_available
