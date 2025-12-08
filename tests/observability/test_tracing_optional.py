"""Tests for optional OpenTelemetry tracing.

This module tests that the tracing functionality works correctly both when
OpenTelemetry is available and when it's not installed.
"""

import importlib
import os
import sys
from unittest import mock

import pytest


class TestOTELAvailability:
    """Test OTEL availability detection."""

    def test_is_otel_available_when_installed(self) -> None:
        """Test that is_otel_available returns True when OTEL is installed."""
        from mlsdm.observability.tracing import is_otel_available

        # In test environment, OTEL should be available (in requirements-dev.txt)
        assert is_otel_available() is True

    def test_is_otel_enabled_default(self) -> None:
        """Test that is_otel_enabled returns correct default value."""
        from mlsdm.observability.tracing import is_otel_enabled

        # Clear any environment variables that might affect this
        with mock.patch.dict(os.environ, {}, clear=True):
            # Default should be enabled when OTEL is available
            result = is_otel_enabled()
            # Should be True if OTEL_AVAILABLE, False otherwise
            assert isinstance(result, bool)

    def test_is_otel_enabled_with_mlsdm_flag_true(self) -> None:
        """Test that MLSDM_ENABLE_OTEL=true enables tracing."""
        from mlsdm.observability.tracing import is_otel_enabled

        with mock.patch.dict(os.environ, {"MLSDM_ENABLE_OTEL": "true"}):
            result = is_otel_enabled()
            # Should be True if OTEL is available
            from mlsdm.observability.tracing import OTEL_AVAILABLE

            assert result == OTEL_AVAILABLE

    def test_is_otel_enabled_with_mlsdm_flag_false(self) -> None:
        """Test that MLSDM_ENABLE_OTEL=false disables tracing."""
        from mlsdm.observability.tracing import is_otel_enabled

        with mock.patch.dict(os.environ, {"MLSDM_ENABLE_OTEL": "false"}):
            assert is_otel_enabled() is False

    def test_is_otel_enabled_with_otel_sdk_disabled_true(self) -> None:
        """Test that OTEL_SDK_DISABLED=true disables tracing."""
        from mlsdm.observability.tracing import is_otel_enabled

        with mock.patch.dict(os.environ, {"OTEL_SDK_DISABLED": "true"}):
            assert is_otel_enabled() is False

    def test_mlsdm_flag_takes_precedence_over_otel_flag(self) -> None:
        """Test that MLSDM_ENABLE_OTEL takes precedence over OTEL_SDK_DISABLED."""
        from mlsdm.observability.tracing import is_otel_enabled

        # MLSDM_ENABLE_OTEL=true should override OTEL_SDK_DISABLED=true
        with mock.patch.dict(os.environ, {"MLSDM_ENABLE_OTEL": "true", "OTEL_SDK_DISABLED": "true"}):
            result = is_otel_enabled()
            from mlsdm.observability.tracing import OTEL_AVAILABLE

            assert result == OTEL_AVAILABLE


class TestNoOpImplementations:
    """Test NoOp implementations when OTEL is unavailable."""

    def test_noop_span_set_attribute(self) -> None:
        """Test that NoOpSpan.set_attribute doesn't raise errors."""
        from mlsdm.observability.tracing import NoOpSpan

        span = NoOpSpan()
        # Should not raise any exception
        span.set_attribute("key", "value")
        span.set_attribute("num", 123)
        span.set_attribute("bool", True)

    def test_noop_span_set_attributes(self) -> None:
        """Test that NoOpSpan.set_attributes doesn't raise errors."""
        from mlsdm.observability.tracing import NoOpSpan

        span = NoOpSpan()
        # Should not raise any exception
        span.set_attributes({"key1": "value1", "key2": 42})

    def test_noop_span_record_exception(self) -> None:
        """Test that NoOpSpan.record_exception doesn't raise errors."""
        from mlsdm.observability.tracing import NoOpSpan

        span = NoOpSpan()
        # Should not raise any exception
        span.record_exception(ValueError("test error"))

    def test_noop_span_set_status(self) -> None:
        """Test that NoOpSpan.set_status doesn't raise errors."""
        from mlsdm.observability.tracing import NoOpSpan

        span = NoOpSpan()
        # Should not raise any exception
        span.set_status("error")

    def test_noop_span_context_manager(self) -> None:
        """Test that NoOpSpan works as a context manager."""
        from mlsdm.observability.tracing import NoOpSpan

        span = NoOpSpan()
        with span:
            pass  # Should not raise any exception

    def test_noop_tracer_start_as_current_span(self) -> None:
        """Test that NoOpTracer.start_as_current_span yields NoOpSpan."""
        from mlsdm.observability.tracing import NoOpSpan, NoOpTracer

        tracer = NoOpTracer()
        with tracer.start_as_current_span("test_span") as span:
            assert isinstance(span, NoOpSpan)
            # Can call methods on the span without errors
            span.set_attribute("key", "value")


class TestTracingWithOTELAvailable:
    """Test tracing functionality when OTEL is available."""

    def test_span_context_manager_basic(self) -> None:
        """Test basic span creation with context manager."""
        from mlsdm.observability.tracing import span

        # Should work without errors whether OTEL is available or not
        with span("test_operation") as s:
            s.set_attribute("test_key", "test_value")

    def test_span_with_attributes(self) -> None:
        """Test span creation with initial attributes."""
        from mlsdm.observability.tracing import span

        with span("test_operation", key1="value1", key2=42) as s:
            # Add more attributes
            s.set_attribute("key3", "value3")

    def test_traced_decorator(self) -> None:
        """Test traced decorator on a function."""
        from mlsdm.observability.tracing import traced

        @traced("test_function")
        def test_func(x: int, y: int) -> int:
            return x + y

        result = test_func(2, 3)
        assert result == 5

    def test_traced_decorator_with_args_recording(self) -> None:
        """Test traced decorator with argument recording."""
        from mlsdm.observability.tracing import traced

        @traced("test_function", record_args=True)
        def test_func(x: int, y: int) -> int:
            return x + y

        result = test_func(2, 3)
        assert result == 5

    def test_traced_decorator_with_result_recording(self) -> None:
        """Test traced decorator with result recording."""
        from mlsdm.observability.tracing import traced

        @traced("test_function", record_result=True)
        def test_func(x: int) -> int:
            return x * 2

        result = test_func(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_traced_async_decorator(self) -> None:
        """Test traced_async decorator on an async function."""
        from mlsdm.observability.tracing import traced_async

        @traced_async("test_async_function")
        async def test_async_func(x: int) -> int:
            return x * 2

        result = await test_async_func(5)
        assert result == 10


class TestTracingConfigWithOTEL:
    """Test tracing configuration when OTEL is available."""

    def test_tracing_config_default_values(self) -> None:
        """Test TracingConfig with default values."""
        from mlsdm.observability.tracing import TracingConfig

        with mock.patch.dict(os.environ, {}, clear=True):
            config = TracingConfig()
            assert config.service_name == "mlsdm"
            assert isinstance(config.enabled, bool)

    def test_tracing_config_custom_service_name(self) -> None:
        """Test TracingConfig with custom service name."""
        from mlsdm.observability.tracing import TracingConfig

        config = TracingConfig(service_name="test-service")
        assert config.service_name == "test-service"

    def test_tracing_config_from_env(self) -> None:
        """Test TracingConfig reads from environment variables."""
        from mlsdm.observability.tracing import TracingConfig

        with mock.patch.dict(
            os.environ,
            {"OTEL_SERVICE_NAME": "env-service", "MLSDM_ENABLE_OTEL": "true"},
        ):
            config = TracingConfig()
            assert config.service_name == "env-service"
            assert config.enabled is True


class TestTracerManager:
    """Test TracerManager functionality."""

    def test_get_tracer_manager_singleton(self) -> None:
        """Test that get_tracer_manager returns a singleton."""
        from mlsdm.observability.tracing import TracerManager, get_tracer_manager

        # Reset instance first
        TracerManager.reset_instance()

        manager1 = get_tracer_manager()
        manager2 = get_tracer_manager()
        assert manager1 is manager2

    def test_tracer_manager_reset_instance(self) -> None:
        """Test that reset_instance creates a new instance."""
        from mlsdm.observability.tracing import TracerManager, get_tracer_manager

        manager1 = get_tracer_manager()
        TracerManager.reset_instance()
        manager2 = get_tracer_manager()
        # After reset, should be different instances
        assert manager1 is not manager2

    def test_tracer_manager_tracer_property(self) -> None:
        """Test that tracer property returns a tracer."""
        from mlsdm.observability.tracing import TracerManager

        TracerManager.reset_instance()
        manager = TracerManager()
        tracer = manager.tracer
        # Should return either a real Tracer or NoOpTracer
        assert tracer is not None


class TestTracingInitializationErrors:
    """Test error handling during tracing initialization."""

    def test_initialization_error_when_enabled_but_unavailable(self) -> None:
        """Test that initialization raises error when enabled but OTEL unavailable."""
        from mlsdm.observability.tracing import OTEL_AVAILABLE, TracerManager, TracingConfig

        if OTEL_AVAILABLE:
            pytest.skip("OTEL is available, cannot test unavailable scenario")

        config = TracingConfig(enabled=True)
        manager = TracerManager(config)

        with pytest.raises(RuntimeError, match="opentelemetry-sdk.*not installed"):
            manager.initialize()


class TestUtilityFunctions:
    """Test utility functions for tracing."""

    def test_add_span_attributes_with_various_types(self) -> None:
        """Test add_span_attributes with different value types."""
        from mlsdm.observability.tracing import NoOpSpan, add_span_attributes

        span = NoOpSpan()
        # Should not raise errors
        add_span_attributes(
            span,
            str_attr="string",
            int_attr=42,
            float_attr=3.14,
            bool_attr=True,
            list_attr=[1, 2, 3],
            none_attr=None,
        )

    def test_mlsdm_specific_trace_functions(self) -> None:
        """Test MLSDM-specific trace context managers."""
        from mlsdm.observability.tracing import (
            trace_aphasia_detection,
            trace_generate,
            trace_moral_filter,
            trace_process_event,
        )

        # All should work without errors
        with trace_generate("test prompt", 0.5, 100):
            pass

        with trace_process_event("cognitive", 0.5):
            pass

        with trace_moral_filter(0.5, 0.8):
            pass

        with trace_aphasia_detection(True, False, 0.7):
            pass
