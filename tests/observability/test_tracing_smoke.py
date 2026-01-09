"""
Smoke tests for OpenTelemetry tracing.

These tests verify that the tracing infrastructure is properly configured
and creates spans for key operations.

NOTE: These tests require the OpenTelemetry SDK to be installed.
Install with: pip install "mlsdm[observability]"
"""

import importlib.util

import pytest

from mlsdm.observability.tracing import (
    TracerManager,
    TracingConfig,
    get_tracer_manager,
    trace_aphasia_detection,
    trace_aphasia_repair,
    trace_full_pipeline,
    trace_generate,
    trace_llm_call,
    trace_memory_retrieval,
    trace_moral_filter,
    trace_request,
    trace_speech_governance,
)

# Check if OpenTelemetry is available
OTEL_AVAILABLE = importlib.util.find_spec("opentelemetry") is not None

# Skip all tests in this module if OpenTelemetry is not available
pytestmark = pytest.mark.skipif(
    not OTEL_AVAILABLE,
    reason="OpenTelemetry SDK not installed. Install with: pip install 'mlsdm[observability]' (issue: https://github.com/neuron7x/mlsdm/issues/1000)",
)


@pytest.fixture
def fresh_tracer():
    """Create a fresh tracer manager for isolation."""
    TracerManager.reset_instance()
    config = TracingConfig(enabled=True, exporter_type="none")
    manager = TracerManager(config)
    manager.initialize()
    yield manager
    TracerManager.reset_instance()


class TestTracingSmoke:
    """Smoke tests for tracing functionality."""

    def test_tracer_manager_initializes(self, fresh_tracer):
        """Test that tracer manager initializes without errors."""
        assert fresh_tracer.enabled
        assert fresh_tracer.tracer is not None

    def test_span_creation_basic(self, fresh_tracer):
        """Test basic span creation."""
        with fresh_tracer.start_span("test_span") as span:
            assert span is not None
            span.set_attribute("test.key", "test_value")

    def test_nested_spans(self, fresh_tracer):
        """Test that nested spans work correctly."""
        with fresh_tracer.start_span("parent") as parent:
            assert parent is not None
            with fresh_tracer.start_span("child") as child:
                assert child is not None
                with fresh_tracer.start_span("grandchild") as grandchild:
                    assert grandchild is not None

    def test_span_with_attributes(self, fresh_tracer):
        """Test span creation with initial attributes."""
        with fresh_tracer.start_span(
            "test_span",
            attributes={
                "key1": "value1",
                "key2": 123,
                "key3": True,
            },
        ) as span:
            assert span is not None


class TestTracingHelperFunctions:
    """Tests for MLSDM-specific tracing helper functions."""

    def test_trace_generate(self, fresh_tracer):
        """Test trace_generate creates span with correct attributes."""
        with trace_generate(
            prompt="Hello world",
            moral_value=0.8,
            max_tokens=128,
        ) as span:
            assert span is not None
            # Span should have been created with attributes

    def test_trace_moral_filter(self, fresh_tracer):
        """Test trace_moral_filter creates span."""
        with trace_moral_filter(threshold=0.5, score=0.8) as span:
            assert span is not None

    def test_trace_memory_retrieval(self, fresh_tracer):
        """Test trace_memory_retrieval creates span."""
        with trace_memory_retrieval(query_type="semantic", top_k=5) as span:
            assert span is not None

    def test_trace_aphasia_detection(self, fresh_tracer):
        """Test trace_aphasia_detection creates span."""
        with trace_aphasia_detection(
            detect_enabled=True,
            repair_enabled=True,
            severity_threshold=0.5,
        ) as span:
            assert span is not None

    def test_trace_aphasia_repair(self, fresh_tracer):
        """Test trace_aphasia_repair creates span."""
        with trace_aphasia_repair(
            detected=True,
            severity=0.6,
            repair_enabled=True,
        ) as span:
            assert span is not None

    def test_trace_llm_call(self, fresh_tracer):
        """Test trace_llm_call creates span."""
        with trace_llm_call(
            prompt_length=100,
            max_tokens=256,
            provider_id="test_provider",
        ) as span:
            assert span is not None

    def test_trace_speech_governance(self, fresh_tracer):
        """Test trace_speech_governance creates span."""
        with trace_speech_governance(
            governance_enabled=True,
            aphasia_mode=True,
        ) as span:
            assert span is not None

    def test_trace_full_pipeline(self, fresh_tracer):
        """Test trace_full_pipeline creates span."""
        with trace_full_pipeline(
            prompt_length=100,
            moral_value=0.7,
            phase="wake",
        ) as span:
            assert span is not None

    def test_trace_request(self, fresh_tracer):
        """Test trace_request creates span."""
        with trace_request(
            request_id="test-123",
            endpoint="/generate",
            method="POST",
        ) as span:
            assert span is not None


class TestTracingConfiguration:
    """Tests for tracing configuration options."""

    def test_tracing_can_be_disabled(self):
        """Test that tracing can be disabled via configuration."""
        TracerManager.reset_instance()
        config = TracingConfig(enabled=False)
        manager = TracerManager(config)
        manager.initialize()

        assert not manager.enabled
        TracerManager.reset_instance()

    def test_mlsdm_otel_enabled_env_var(self, monkeypatch):
        """Test MLSDM_OTEL_ENABLED environment variable."""
        TracerManager.reset_instance()
        monkeypatch.setenv("MLSDM_OTEL_ENABLED", "true")

        config = TracingConfig()
        assert config.enabled is True

        TracerManager.reset_instance()

    def test_mlsdm_otel_enabled_false(self, monkeypatch):
        """Test MLSDM_OTEL_ENABLED=false disables tracing."""
        TracerManager.reset_instance()
        monkeypatch.setenv("MLSDM_OTEL_ENABLED", "false")

        config = TracingConfig()
        assert config.enabled is False

        TracerManager.reset_instance()

    def test_mlsdm_otel_endpoint_env_var(self, monkeypatch):
        """Test MLSDM_OTEL_ENDPOINT environment variable."""
        TracerManager.reset_instance()
        test_endpoint = "http://custom-endpoint:4318"
        monkeypatch.setenv("MLSDM_OTEL_ENDPOINT", test_endpoint)

        config = TracingConfig()
        assert config.otlp_endpoint == test_endpoint

        TracerManager.reset_instance()

    def test_exporter_type_none(self):
        """Test lightweight mode with no exporter."""
        TracerManager.reset_instance()
        config = TracingConfig(enabled=True, exporter_type="none")
        manager = TracerManager(config)
        manager.initialize()

        # Should still create spans, just no export
        with manager.start_span("test") as span:
            assert span is not None

        TracerManager.reset_instance()


class TestSpanChaining:
    """Tests for verifying span chains are created correctly."""

    def test_pipeline_span_chain(self, fresh_tracer):
        """Test that a full pipeline creates proper span chain."""
        spans_created = []

        with trace_full_pipeline(100, 0.7, "wake") as pipeline_span:
            spans_created.append(pipeline_span)

            with trace_moral_filter(0.5, 0.8) as moral_span:
                spans_created.append(moral_span)

            with trace_aphasia_detection(True, True, 0.5) as aphasia_span:
                spans_created.append(aphasia_span)

            with trace_memory_retrieval("semantic", 5) as memory_span:
                spans_created.append(memory_span)

            with trace_llm_call(100, 256) as llm_span:
                spans_created.append(llm_span)

            with trace_speech_governance(True) as speech_span:
                spans_created.append(speech_span)

        # All spans should have been created
        assert len(spans_created) == 6
        for span in spans_created:
            assert span is not None

    def test_exception_handling_in_span(self, fresh_tracer):
        """Test that exceptions are properly recorded on spans."""
        tracer_manager = get_tracer_manager()

        with pytest.raises(ValueError), tracer_manager.start_span("error_span") as span:
            tracer_manager.record_exception(span, ValueError("Test error"))
            raise ValueError("Test error")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
