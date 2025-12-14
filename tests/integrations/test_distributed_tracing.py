"""
Tests for distributed tracing integration.
"""

from mlsdm.integrations import DistributedTracer


class TestDistributedTracer:
    """Test distributed tracing integration."""

    def test_initialization(self) -> None:
        """Test tracer initialization."""
        tracer = DistributedTracer(
            service_name="mlsdm-test",
            exporter_endpoint="http://localhost:4317",
            enable_console=True,
        )

        assert tracer.service_name == "mlsdm-test"
        assert tracer.exporter_endpoint == "http://localhost:4317"
        assert tracer.enable_console is True

    def test_start_span_without_otel(self) -> None:
        """Test span creation when OpenTelemetry is not available."""
        tracer = DistributedTracer(service_name="test")

        # Should return dummy span that doesn't error
        with tracer.start_span("test_operation") as span:
            span.set_attribute("test_key", "test_value")
            span.add_event("test_event")

        # Should complete without errors

    def test_start_span_with_attributes(self) -> None:
        """Test span creation with attributes."""
        tracer = DistributedTracer(service_name="test")

        attributes = {"user_id": "123", "request_type": "generate"}

        # Should work without errors even if OTEL not available
        with tracer.start_span("api_request", attributes=attributes) as span:
            span.set_attribute("additional", "value")

        # Test should complete without errors

    def test_add_event_without_otel(self) -> None:
        """Test add_event when OpenTelemetry is not available."""
        tracer = DistributedTracer(service_name="test")

        # Should not error even without OTEL
        tracer.add_event("test_event", {"key": "value"})

    def test_set_attribute_without_otel(self) -> None:
        """Test set_attribute when OpenTelemetry is not available."""
        tracer = DistributedTracer(service_name="test")

        # Should not error even without OTEL
        tracer.set_attribute("test_key", "test_value")
