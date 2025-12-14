"""
Distributed Tracing Integration

Complete OpenTelemetry integration across all MLSDM components with
automatic span creation and context propagation.
"""

import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


class DistributedTracer:
    """
    Distributed tracing manager with OpenTelemetry.

    Provides span creation, context propagation, and automatic
    instrumentation for MLSDM components.

    Example:
        >>> tracer = DistributedTracer(
        ...     service_name="mlsdm-engine",
        ...     exporter_endpoint="http://localhost:4317"
        ... )
        >>> with tracer.start_span("generate_request") as span:
        ...     span.set_attribute("prompt_length", 100)
        ...     result = engine.generate(prompt)
    """

    def __init__(
        self,
        service_name: str = "mlsdm",
        exporter_endpoint: Optional[str] = None,
        enable_console: bool = False,
    ) -> None:
        """
        Initialize distributed tracer.

        Args:
            service_name: Service name for traces
            exporter_endpoint: OTLP exporter endpoint (e.g., http://localhost:4317)
            enable_console: Enable console exporter for debugging
        """
        self.service_name = service_name
        self.exporter_endpoint = exporter_endpoint
        self.enable_console = enable_console
        self.logger = logging.getLogger(__name__)

        self._tracer: Any = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize OpenTelemetry tracer."""
        if not OTEL_AVAILABLE:
            self.logger.warning(
                "OpenTelemetry not available. Install with: "
                "pip install opentelemetry-api opentelemetry-sdk"
            )
            return

        try:
            # Create tracer provider
            provider = TracerProvider()

            # Add OTLP exporter if endpoint configured
            if self.exporter_endpoint:
                otlp_exporter = OTLPSpanExporter(endpoint=self.exporter_endpoint)
                provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
                self.logger.info(f"OTLP exporter configured: {self.exporter_endpoint}")

            # Add console exporter for debugging
            if self.enable_console:
                console_exporter = ConsoleSpanExporter()
                provider.add_span_processor(BatchSpanProcessor(console_exporter))

            # Set global tracer provider
            trace.set_tracer_provider(provider)

            # Get tracer for this service
            self._tracer = trace.get_tracer(self.service_name)

            self.logger.info(f"Distributed tracing initialized for {self.service_name}")

        except Exception as e:
            self.logger.error(f"Failed to initialize tracing: {e}")
            self._tracer = None

    @contextmanager
    def start_span(
        self, name: str, attributes: Optional[Dict[str, Any]] = None
    ) -> Iterator[Any]:
        """
        Start a new span.

        Args:
            name: Span name
            attributes: Optional span attributes

        Yields:
            Span object or None if tracing disabled
        """
        if not OTEL_AVAILABLE or self._tracer is None:
            # Return dummy span that does nothing
            yield DummySpan()
            return

        span = self._tracer.start_as_current_span(name)
        span_context = span.__enter__()

        # Set attributes
        if attributes:
            for key, value in attributes.items():
                span_context.set_attribute(key, value)

        try:
            yield span_context
        finally:
            span.__exit__(None, None, None)

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """
        Add event to current span.

        Args:
            name: Event name
            attributes: Optional event attributes
        """
        if not OTEL_AVAILABLE:
            return

        try:
            current_span = trace.get_current_span()
            if current_span:
                current_span.add_event(name, attributes or {})
        except Exception as e:
            self.logger.debug(f"Failed to add event: {e}")

    def set_attribute(self, key: str, value: Any) -> None:
        """
        Set attribute on current span.

        Args:
            key: Attribute key
            value: Attribute value
        """
        if not OTEL_AVAILABLE:
            return

        try:
            current_span = trace.get_current_span()
            if current_span:
                current_span.set_attribute(key, value)
        except Exception as e:
            self.logger.debug(f"Failed to set attribute: {e}")


class DummySpan:
    """Dummy span when tracing is disabled."""

    def set_attribute(self, key: str, value: Any) -> None:
        """No-op set attribute."""
        pass

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """No-op add event."""
        pass
