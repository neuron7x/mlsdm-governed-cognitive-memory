"""OpenTelemetry distributed tracing for MLSDM.

This module provides distributed tracing capabilities using OpenTelemetry,
enabling observability across the entire cognitive pipeline.

**IMPORTANT**: OpenTelemetry is an optional dependency. When not installed,
this module provides no-op implementations that allow the code to run without
tracing functionality. This ensures that `import mlsdm` never fails due to
missing OTEL packages.

Note: mypy type checking is relaxed for this module due to conditional OTEL imports.

Features:
- Span creation for key operations (API handlers, generate, process_event)
- Trace context propagation
- Export to configurable backends (Jaeger/OTLP)
- Integration with FastAPI middleware

Configuration:
- OTEL_SERVICE_NAME: Service name (default: mlsdm)
- OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint URL
- OTEL_EXPORTER_OTLP_PROTOCOL: Protocol (http/protobuf, grpc)
- OTEL_TRACES_SAMPLER: Sampling strategy (always_on, always_off, traceidratio)
- OTEL_TRACES_SAMPLER_ARG: Sampler argument (e.g., 0.1 for 10% sampling)
"""
# Relax mypy checking for conditional imports
# mypy: disable-error-code="assignment,misc,no-redef,unused-ignore"

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from functools import wraps
from threading import Lock
from typing import TYPE_CHECKING, Any, Literal

# Conditional OpenTelemetry imports - OTEL is optional
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import SpanKind, Status, StatusCode

    OTEL_AVAILABLE = True
except ImportError:
    # When OTEL is not installed, provide no-op stubs
    # Use descriptive stub class for better debugging
    class _NoOpOtelClass:
        """Stub class for OpenTelemetry components when OTEL is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    trace = None  # type: ignore
    Resource = None  # type: ignore
    SpanProcessor = _NoOpOtelClass  # type: ignore
    TracerProvider = _NoOpOtelClass  # type: ignore
    BatchSpanProcessor = _NoOpOtelClass  # type: ignore
    ConsoleSpanExporter = _NoOpOtelClass  # type: ignore

    # Create minimal stub classes for SpanKind, Status, StatusCode
    class SpanKind:  # type: ignore[no-redef]
        INTERNAL = 0
        SERVER = 1
        CLIENT = 2
        PRODUCER = 3
        CONSUMER = 4

    class StatusCode:  # type: ignore[no-redef]
        UNSET = 0
        OK = 1
        ERROR = 2

    class Status:  # type: ignore[no-redef]
        def __init__(self, status_code: int, description: str = "") -> None:
            self.status_code = status_code
            self.description = description

    OTEL_AVAILABLE = False

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    # Type checking imports - only used by mypy, not at runtime
    # When OTEL is not installed during type checking, these will be Any
    try:
        from opentelemetry.context import Context  # noqa: F401
        from opentelemetry.trace import Span, Tracer
    except ImportError:
        # Fallback types when OTEL not available during type checking
        Span = Any  # type: ignore
        Tracer = Any  # type: ignore
        Context = Any  # type: ignore

logger = logging.getLogger(__name__)

# Version constant for consistency across the codebase
MLSDM_VERSION = "1.0.0"

# Span attribute prefix constants
SPAN_ATTR_PREFIX_MLSDM = "mlsdm."
SPAN_ATTR_PREFIX_HTTP = "http."


# ---------------------------------------------------------------------------
# No-op implementations for when OpenTelemetry is not available
# ---------------------------------------------------------------------------


class _NoOpSpan:
    """No-op span implementation when OpenTelemetry is not available."""

    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        pass

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        pass

    def record_exception(
        self, exception: Exception, attributes: dict[str, Any] | None = None
    ) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def get_span_context(self) -> Any:
        return _NoOpSpanContext()


class _NoOpSpanContext:
    """No-op span context when OpenTelemetry is not available."""

    trace_id = 0
    span_id = 0


class _NoOpTracer:
    """No-op tracer implementation when OpenTelemetry is not available."""

    def start_as_current_span(
        self,
        name: str,
        kind: Any = None,
        attributes: dict[str, Any] | None = None,
        context: Any = None,
    ) -> _NoOpSpan:
        return _NoOpSpan()

    def start_span(
        self,
        name: str,
        kind: Any = None,
        attributes: dict[str, Any] | None = None,
        context: Any = None,
    ) -> _NoOpSpan:
        return _NoOpSpan()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TracingConfig:
    """Configuration for OpenTelemetry tracing.

    Supports both standard OpenTelemetry environment variables and
    MLSDM-specific variables for convenience:

    Standard OTEL Variables:
    - OTEL_SERVICE_NAME: Service name (default: mlsdm)
    - OTEL_SDK_DISABLED: Disable tracing (default: false)
    - OTEL_EXPORTER_TYPE: Exporter type (console, otlp, jaeger, none)
    - OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint URL
    - OTEL_EXPORTER_OTLP_PROTOCOL: Protocol (http/protobuf, grpc)
    - OTEL_TRACES_SAMPLER_ARG: Sampling rate (0.0 to 1.0)

    MLSDM-specific Variables (override OTEL equivalents):
    - MLSDM_OTEL_ENABLED: Enable tracing (takes precedence over OTEL_SDK_DISABLED)
    - MLSDM_OTEL_ENDPOINT: OTLP endpoint (takes precedence over OTEL_EXPORTER_OTLP_ENDPOINT)

    Attributes:
        service_name: Name of the service for tracing
        enabled: Whether tracing is enabled
        exporter_type: Type of exporter (console, otlp, jaeger)
        otlp_endpoint: OTLP exporter endpoint
        otlp_protocol: OTLP protocol (http/protobuf, grpc)
        sample_rate: Sampling rate (0.0 to 1.0)
        batch_max_queue_size: Maximum queue size for batch processor
        batch_max_export_batch_size: Maximum batch size for export
        batch_schedule_delay_millis: Delay between exports in milliseconds
    """

    def __init__(
        self,
        service_name: str | None = None,
        enabled: bool | None = None,
        exporter_type: Literal["console", "otlp", "jaeger", "none"] | None = None,
        otlp_endpoint: str | None = None,
        otlp_protocol: Literal["http/protobuf", "grpc"] | None = None,
        sample_rate: float | None = None,
        batch_max_queue_size: int = 2048,
        batch_max_export_batch_size: int = 512,
        batch_schedule_delay_millis: int = 5000,
    ) -> None:
        """Initialize tracing configuration from environment or parameters."""
        self.service_name: str = service_name or os.getenv("OTEL_SERVICE_NAME", "mlsdm") or "mlsdm"

        # Check for MLSDM-specific enable flag first, then fall back to OTEL standard
        mlsdm_enabled = os.getenv("MLSDM_OTEL_ENABLED")
        if enabled is not None:
            self.enabled = enabled
        elif mlsdm_enabled is not None:
            self.enabled = mlsdm_enabled.lower() == "true"
        else:
            self.enabled = os.getenv("OTEL_SDK_DISABLED", "false").lower() != "true"

        self.exporter_type = exporter_type or os.getenv(
            "OTEL_EXPORTER_TYPE", "console"
        )

        # Check for MLSDM-specific endpoint first, then fall back to OTEL standard
        mlsdm_endpoint = os.getenv("MLSDM_OTEL_ENDPOINT")
        self.otlp_endpoint = (
            otlp_endpoint
            or mlsdm_endpoint
            or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        )

        self.otlp_protocol = otlp_protocol or os.getenv(
            "OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf"
        )

        # Parse sample rate
        sample_rate_str = os.getenv("OTEL_TRACES_SAMPLER_ARG", "1.0")
        try:
            self.sample_rate = (
                sample_rate if sample_rate is not None else float(sample_rate_str)
            )
        except ValueError:
            self.sample_rate = 1.0

        self.batch_max_queue_size = batch_max_queue_size
        self.batch_max_export_batch_size = batch_max_export_batch_size
        self.batch_schedule_delay_millis = batch_schedule_delay_millis


# ---------------------------------------------------------------------------
# Tracer Management
# ---------------------------------------------------------------------------


class TracerManager:
    """Manages OpenTelemetry tracer lifecycle and provides tracing utilities.

    This class implements the singleton pattern for tracer management,
    ensuring consistent tracing across the application.

    Example:
        >>> manager = get_tracer_manager()
        >>> with manager.start_span("my_operation") as span:
        ...     span.set_attribute("key", "value")
        ...     # do work
    """

    _instance: TracerManager | None = None
    _lock: Lock = Lock()

    def __init__(self, config: TracingConfig | None = None) -> None:
        """Initialize tracer manager with configuration.

        Args:
            config: Tracing configuration. If None, uses defaults from environment.
        """
        self._config = config or TracingConfig()
        self._initialized = False
        self._tracer: Tracer | None = None
        self._provider: TracerProvider | None = None
        self._processor: SpanProcessor | None = None

    @classmethod
    def get_instance(cls, config: TracingConfig | None = None) -> TracerManager:
        """Get or create the singleton TracerManager instance.

        Note:
            The config parameter is only used when creating the instance.
            Subsequent calls with different config will be ignored.

        Args:
            config: Tracing configuration (only used on first call)

        Returns:
            TracerManager singleton instance
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.shutdown()
                cls._instance = None

    def initialize(self) -> None:
        """Initialize the OpenTelemetry tracer provider and exporter.

        This method sets up the tracer provider with the configured exporter
        and registers it as the global tracer provider.

        If OpenTelemetry is not available, this method does nothing and
        the tracer manager will provide no-op implementations.
        """
        if self._initialized:
            return

        if not OTEL_AVAILABLE:
            logger.info("OpenTelemetry not available, using no-op tracer")
            self._initialized = True
            return

        if not self._config.enabled:
            logger.info("OpenTelemetry tracing is disabled")
            return

        try:
            # Create resource with service information
            resource = Resource.create(
                {
                    "service.name": self._config.service_name,
                    "service.version": MLSDM_VERSION,
                    "deployment.environment": os.getenv("DEPLOYMENT_ENV", "development"),
                }
            )

            # Create tracer provider
            self._provider = TracerProvider(resource=resource)

            # Create exporter based on configuration
            exporter = self._create_exporter()
            if exporter is not None:
                self._processor = BatchSpanProcessor(
                    exporter,
                    max_queue_size=self._config.batch_max_queue_size,
                    max_export_batch_size=self._config.batch_max_export_batch_size,
                    schedule_delay_millis=self._config.batch_schedule_delay_millis,
                )
                self._provider.add_span_processor(self._processor)

            # Register as global provider
            trace.set_tracer_provider(self._provider)

            # Get tracer
            self._tracer = trace.get_tracer(
                self._config.service_name,
                MLSDM_VERSION,
            )

            self._initialized = True
            logger.info(
                "OpenTelemetry tracing initialized",
                extra={
                    "service_name": self._config.service_name,
                    "exporter_type": self._config.exporter_type,
                },
            )

        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry tracing: {e}")
            self._initialized = False

    def _create_exporter(self) -> Any | None:
        """Create the appropriate span exporter based on configuration.

        Returns:
            Configured span exporter or None if disabled
        """
        if not OTEL_AVAILABLE:
            return None

        if self._config.exporter_type == "none":
            return None

        if self._config.exporter_type == "console":
            return ConsoleSpanExporter()

        if self._config.exporter_type == "otlp":
            try:
                if self._config.otlp_protocol == "grpc":
                    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                        OTLPSpanExporter,
                    )

                    return OTLPSpanExporter(endpoint=self._config.otlp_endpoint)
                else:
                    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                        OTLPSpanExporter,
                    )

                    return OTLPSpanExporter(endpoint=self._config.otlp_endpoint)
            except ImportError:
                logger.warning(
                    "OTLP exporter not available, falling back to console exporter"
                )
                return ConsoleSpanExporter()

        if self._config.exporter_type == "jaeger":
            try:
                from opentelemetry.exporter.jaeger.thrift import JaegerExporter

                return JaegerExporter()
            except ImportError:
                logger.warning(
                    "Jaeger exporter not available, falling back to console exporter"
                )
                return ConsoleSpanExporter()

        return ConsoleSpanExporter()

    def shutdown(self) -> None:
        """Shutdown the tracer provider and flush pending spans."""
        if self._provider is not None:
            try:
                self._provider.shutdown()
            except Exception as e:
                logger.error(f"Error during tracer shutdown: {e}")
            finally:
                self._provider = None
                self._tracer = None
                self._initialized = False

    @property
    def tracer(self) -> Any:
        """Get the tracer instance, initializing if necessary.

        Returns:
            OpenTelemetry Tracer instance or no-op tracer if OTEL not available
        """
        if not self._initialized:
            self.initialize()

        if not OTEL_AVAILABLE or self._tracer is None:
            # Return a no-op tracer if OTEL not available or not initialized
            return _NoOpTracer()

        return self._tracer

    @property
    def enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self._config.enabled and self._initialized

    @contextmanager
    def start_span(
        self,
        name: str,
        kind: Any = None,
        attributes: dict[str, Any] | None = None,
        context: Any = None,
    ) -> Iterator[Any]:
        """Start a new span as a context manager.

        Args:
            name: Name of the span
            kind: Kind of span (INTERNAL, SERVER, CLIENT, PRODUCER, CONSUMER).
                  When OTEL is not available, this is ignored by the no-op tracer.
            attributes: Initial span attributes
            context: Parent context (optional)

        Yields:
            The created span (or no-op span if OTEL not available)

        Example:
            >>> with tracer_manager.start_span("process_event") as span:
            ...     span.set_attribute("event_type", "cognitive")
            ...     process_event()
        """
        # Use INTERNAL as default kind. When OTEL not available, SpanKind.INTERNAL
        # is 0 from our stub, but the no-op tracer ignores it anyway.
        if kind is None:
            kind = SpanKind.INTERNAL

        with self.tracer.start_as_current_span(
            name,
            kind=kind,
            attributes=attributes,
            context=context,
        ) as span:
            yield span

    def record_exception(self, span: Any, exception: Exception) -> None:
        """Record an exception on a span.

        Args:
            span: The span to record the exception on
            exception: The exception to record
        """
        if not OTEL_AVAILABLE:
            return
        span.record_exception(exception)
        span.set_status(Status(StatusCode.ERROR, str(exception)))


# ---------------------------------------------------------------------------
# Global accessor functions
# ---------------------------------------------------------------------------

_manager: TracerManager | None = None
_manager_lock = Lock()


def get_tracer_manager(config: TracingConfig | None = None) -> TracerManager:
    """Get or create the global TracerManager instance.

    Args:
        config: Tracing configuration (only used on first call)

    Returns:
        TracerManager singleton instance
    """
    return TracerManager.get_instance(config)


def get_tracer() -> Any:
    """Get the global tracer instance.

    Returns:
        OpenTelemetry Tracer instance or no-op tracer if OTEL not available
    """
    return get_tracer_manager().tracer


def initialize_tracing(config: TracingConfig | None = None) -> None:
    """Initialize OpenTelemetry tracing with the given configuration.

    This should be called at application startup before any tracing operations.

    Args:
        config: Tracing configuration
    """
    manager = get_tracer_manager(config)
    manager.initialize()


def shutdown_tracing() -> None:
    """Shutdown OpenTelemetry tracing and flush pending spans.

    This should be called at application shutdown.
    """
    TracerManager.reset_instance()


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Any]:
    """Simple context manager for creating spans with attributes.

    This is a convenience wrapper around TracerManager.start_span() that
    provides a cleaner API for common use cases.

    The function is safe to use even when tracing is disabled or when
    OpenTelemetry is not installed - it will create a no-op span that
    accepts attribute calls without error.

    Args:
        name: Name of the span (e.g., "mlsdm.generate", "mlsdm.memory.query")
        **attrs: Span attributes to set (will be prefixed with 'mlsdm.' if not already)

    Yields:
        The created span (or no-op span if OTEL not available)

    Example:
        >>> with span("mlsdm.generate", phase="wake", stateless_mode=False):
        ...     # do work
        ...     pass

        >>> with span("mlsdm.cognitive_controller.step", step=1):
        ...     result = controller.process_event(vector, moral)
    """
    manager = get_tracer_manager()

    # Normalize attribute keys to use mlsdm prefix
    normalized_attrs: dict[str, Any] = {}
    for key, value in attrs.items():
        if not key.startswith(SPAN_ATTR_PREFIX_MLSDM) and not key.startswith(SPAN_ATTR_PREFIX_HTTP):
            normalized_attrs[f"{SPAN_ATTR_PREFIX_MLSDM}{key}"] = value
        else:
            normalized_attrs[key] = value

    with manager.start_span(name, attributes=normalized_attrs) as s:
        yield s


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


def traced(
    name: str | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    record_args: bool = False,
    record_result: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to trace a function.

    Args:
        name: Span name (defaults to function name)
        kind: Span kind
        record_args: Whether to record function arguments as attributes
        record_result: Whether to record function result as attribute

    Returns:
        Decorated function

    Example:
        >>> @traced("my_operation", record_args=True)
        ... def my_function(x: int, y: int) -> int:
        ...     return x + y
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        span_name = name or func.__name__

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            manager = get_tracer_manager()

            with manager.start_span(span_name, kind=kind) as span:
                # Record arguments if requested
                if record_args:
                    for i, arg in enumerate(args):
                        span.set_attribute(f"arg.{i}", str(arg)[:1024])
                    for key, value in kwargs.items():
                        span.set_attribute(f"kwarg.{key}", str(value)[:1024])

                try:
                    result = func(*args, **kwargs)

                    # Record result if requested
                    if record_result:
                        span.set_attribute("result", str(result)[:1024])

                    return result

                except Exception as e:
                    manager.record_exception(span, e)
                    raise

        return wrapper

    return decorator


def traced_async(
    name: str | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    record_args: bool = False,
    record_result: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to trace an async function.

    Args:
        name: Span name (defaults to function name)
        kind: Span kind
        record_args: Whether to record function arguments as attributes
        record_result: Whether to record function result as attribute

    Returns:
        Decorated async function
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        span_name = name or func.__name__

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            manager = get_tracer_manager()

            with manager.start_span(span_name, kind=kind) as span:
                # Record arguments if requested
                if record_args:
                    for i, arg in enumerate(args):
                        span.set_attribute(f"arg.{i}", str(arg)[:1024])
                    for key, value in kwargs.items():
                        span.set_attribute(f"kwarg.{key}", str(value)[:1024])

                try:
                    result = await func(*args, **kwargs)

                    # Record result if requested
                    if record_result:
                        span.set_attribute("result", str(result)[:1024])

                    return result

                except Exception as e:
                    manager.record_exception(span, e)
                    raise

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# MLSDM-specific tracing utilities
# ---------------------------------------------------------------------------


def trace_generate(
    prompt: str,
    moral_value: float,
    max_tokens: int,
) -> Any:
    """Create a context manager for tracing generate operations.

    Args:
        prompt: The input prompt
        moral_value: The moral threshold value
        max_tokens: Maximum tokens to generate

    Returns:
        Context manager yielding a span
    """
    manager = get_tracer_manager()
    return manager.start_span(
        "mlsdm.generate",
        kind=SpanKind.INTERNAL,
        attributes={
            "mlsdm.prompt_length": len(prompt),
            "mlsdm.moral_value": moral_value,
            "mlsdm.max_tokens": max_tokens,
        },
    )


def trace_process_event(
    event_type: str,
    moral_value: float,
) -> Any:
    """Create a context manager for tracing process_event operations.

    Args:
        event_type: Type of event being processed
        moral_value: The moral threshold value

    Returns:
        Context manager yielding a span
    """
    manager = get_tracer_manager()
    return manager.start_span(
        "mlsdm.process_event",
        kind=SpanKind.INTERNAL,
        attributes={
            "mlsdm.event_type": event_type,
            "mlsdm.moral_value": moral_value,
        },
    )


def trace_memory_retrieval(
    query_type: str,
    top_k: int,
) -> Any:
    """Create a context manager for tracing memory retrieval operations.

    Args:
        query_type: Type of query being executed
        top_k: Number of results requested

    Returns:
        Context manager yielding a span
    """
    manager = get_tracer_manager()
    return manager.start_span(
        "mlsdm.memory_retrieval",
        kind=SpanKind.INTERNAL,
        attributes={
            "mlsdm.query_type": query_type,
            "mlsdm.top_k": top_k,
        },
    )


def trace_moral_filter(
    threshold: float,
    score: float | None = None,
) -> Any:
    """Create a context manager for tracing moral filter operations.

    Args:
        threshold: Current moral threshold
        score: Computed moral score (optional)

    Returns:
        Context manager yielding a span
    """
    manager = get_tracer_manager()
    attributes: dict[str, Any] = {"mlsdm.moral_threshold": threshold}
    if score is not None:
        attributes["mlsdm.moral_score"] = score

    return manager.start_span(
        "mlsdm.moral_filter",
        kind=SpanKind.INTERNAL,
        attributes=attributes,
    )


def trace_aphasia_detection(
    detect_enabled: bool,
    repair_enabled: bool,
    severity_threshold: float,
) -> Any:
    """Create a context manager for tracing aphasia detection operations.

    Args:
        detect_enabled: Whether aphasia detection is enabled
        repair_enabled: Whether aphasia repair is enabled
        severity_threshold: Threshold for triggering repair

    Returns:
        Context manager yielding a span
    """
    manager = get_tracer_manager()
    return manager.start_span(
        "mlsdm.aphasia_detection",
        kind=SpanKind.INTERNAL,
        attributes={
            "mlsdm.aphasia.detect_enabled": detect_enabled,
            "mlsdm.aphasia.repair_enabled": repair_enabled,
            "mlsdm.aphasia.severity_threshold": severity_threshold,
        },
    )


def trace_emergency_shutdown(
    reason: str,
    memory_mb: float | None = None,
) -> Any:
    """Create a context manager for tracing emergency shutdown events.

    Args:
        reason: Reason for the emergency shutdown
        memory_mb: Current memory usage in MB (if applicable)

    Returns:
        Context manager yielding a span
    """
    manager = get_tracer_manager()
    attributes: dict[str, Any] = {
        "mlsdm.emergency.reason": reason,
        "mlsdm.emergency.shutdown": True,
    }
    if memory_mb is not None:
        attributes["mlsdm.emergency.memory_mb"] = memory_mb

    return manager.start_span(
        "mlsdm.emergency_shutdown",
        kind=SpanKind.INTERNAL,
        attributes=attributes,
    )


def trace_phase_transition(
    from_phase: str,
    to_phase: str,
) -> Any:
    """Create a context manager for tracing phase transitions.

    Args:
        from_phase: The phase transitioning from
        to_phase: The phase transitioning to

    Returns:
        Context manager yielding a span
    """
    manager = get_tracer_manager()
    return manager.start_span(
        "mlsdm.phase_transition",
        kind=SpanKind.INTERNAL,
        attributes={
            "mlsdm.phase.from": from_phase,
            "mlsdm.phase.to": to_phase,
        },
    )


def trace_full_pipeline(
    prompt_length: int,
    moral_value: float,
    phase: str,
) -> Any:
    """Create a context manager for tracing the full cognitive pipeline.

    This is a root span that encompasses the entire request path:
    input -> validation -> controller -> memory -> moral -> aphasia -> output

    Args:
        prompt_length: Length of the input prompt (NOT the prompt itself, to avoid PII)
        moral_value: The moral threshold value
        phase: Current cognitive phase

    Returns:
        Context manager yielding a span
    """
    manager = get_tracer_manager()
    return manager.start_span(
        "mlsdm.full_pipeline",
        kind=SpanKind.SERVER,
        attributes={
            "mlsdm.prompt_length": prompt_length,
            "mlsdm.moral_value": moral_value,
            "mlsdm.phase": phase,
        },
    )


def add_span_attributes(span: Span, **attributes: Any) -> None:
    """Add attributes to a span, handling type conversion.

    Args:
        span: The span to add attributes to
        **attributes: Key-value pairs to add as attributes
    """
    for key, value in attributes.items():
        if value is None:
            continue

        # Convert to supported types
        if isinstance(value, (str, int, float, bool)):
            span.set_attribute(key, value)
        elif isinstance(value, (list, tuple)):
            # Convert lists to comma-separated strings
            span.set_attribute(key, ",".join(str(v) for v in value))
        else:
            span.set_attribute(key, str(value)[:1024])


# ---------------------------------------------------------------------------
# Additional MLSDM-specific tracing utilities (Phase 7)
# ---------------------------------------------------------------------------


def trace_aphasia_repair(
    detected: bool,
    severity: float,
    repair_enabled: bool,
) -> Any:
    """Create a context manager for tracing aphasia repair operations.

    Args:
        detected: Whether aphasia was detected
        severity: Severity score (0.0 to 1.0)
        repair_enabled: Whether repair is enabled

    Returns:
        Context manager yielding a span
    """
    manager = get_tracer_manager()
    return manager.start_span(
        "mlsdm.aphasia_repair",
        kind=SpanKind.INTERNAL,
        attributes={
            "mlsdm.aphasia.detected": detected,
            "mlsdm.aphasia.severity": severity,
            "mlsdm.aphasia.repair_enabled": repair_enabled,
        },
    )


def trace_llm_call(
    prompt_length: int,
    max_tokens: int,
    provider_id: str | None = None,
) -> Any:
    """Create a context manager for tracing LLM call operations.

    Args:
        prompt_length: Length of prompt (NOT the prompt itself)
        max_tokens: Maximum tokens to generate
        provider_id: Optional provider identifier

    Returns:
        Context manager yielding a span
    """
    manager = get_tracer_manager()
    attributes: dict[str, Any] = {
        "mlsdm.llm.prompt_length": prompt_length,
        "mlsdm.llm.max_tokens": max_tokens,
    }
    if provider_id is not None:
        attributes["mlsdm.llm.provider_id"] = provider_id

    return manager.start_span(
        "mlsdm.llm_call",
        kind=SpanKind.CLIENT,
        attributes=attributes,
    )


def trace_speech_governance(
    governance_enabled: bool,
    aphasia_mode: bool = False,
) -> Any:
    """Create a context manager for tracing speech governance operations.

    Args:
        governance_enabled: Whether speech governance is enabled
        aphasia_mode: Whether aphasia mode is enabled

    Returns:
        Context manager yielding a span
    """
    manager = get_tracer_manager()
    return manager.start_span(
        "mlsdm.speech_governance",
        kind=SpanKind.INTERNAL,
        attributes={
            "mlsdm.speech.governance_enabled": governance_enabled,
            "mlsdm.speech.aphasia_mode": aphasia_mode,
        },
    )


def trace_request(
    request_id: str,
    endpoint: str,
    method: str = "POST",
) -> Any:
    """Create a context manager for tracing HTTP requests.

    This is a high-level span for the entire API request.

    Args:
        request_id: Unique request identifier
        endpoint: API endpoint path
        method: HTTP method

    Returns:
        Context manager yielding a span
    """
    manager = get_tracer_manager()
    return manager.start_span(
        f"api.{endpoint.lstrip('/').replace('/', '_')}",
        kind=SpanKind.SERVER,
        attributes={
            "http.method": method,
            "http.route": endpoint,
            "mlsdm.request_id": request_id,
        },
    )
