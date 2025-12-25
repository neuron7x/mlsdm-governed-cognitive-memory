import asyncio
import hashlib
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import psutil
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field

# Try to import OpenTelemetry, but allow graceful degradation
try:
    from opentelemetry.trace import SpanKind

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    if not TYPE_CHECKING:
        SpanKind = None

from mlsdm.api import health
from mlsdm.api.health import _cpu_background_sampler
from mlsdm.api.lifecycle import cleanup_memory_manager, get_lifecycle_manager
from mlsdm.api.middleware import (
    BulkheadMiddleware,
    PriorityMiddleware,
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
    TimeoutMiddleware,
)
from mlsdm.config.constants import DEFAULT_CONFIG_PATH, STRICT_CONFIG_MODES
from mlsdm.config.runtime import RuntimeMode, get_runtime_mode
from mlsdm.contracts import AphasiaMetadata
from mlsdm.core.memory_manager import MemoryManager
from mlsdm.engine import NeuroEngineConfig, build_neuro_engine_from_env
from mlsdm.observability.tracing import (
    TracingConfig,
    add_span_attributes,
    get_tracer_manager,
    initialize_tracing,
    shutdown_tracing,
)
from mlsdm.utils.config_loader import ConfigLoader
from mlsdm.utils.input_validator import InputValidator
from mlsdm.utils.rate_limiter import RateLimiter
from mlsdm.utils.security_logger import SecurityEventType, get_security_logger

logger = logging.getLogger(__name__)
security_logger = get_security_logger()

# Global reference to CPU background sampler task
_cpu_background_task: asyncio.Task[None] | None = None

_app_instance: FastAPI | None = None
_config_data: dict[str, Any] | None = None
_config_path: str | None = None
_runtime_mode: RuntimeMode | None = None
_manager: MemoryManager | None = None
_neuro_engine: Any | None = None
_rate_limiter: RateLimiter | None = None
_validator: InputValidator | None = None
_secure_mode_enabled: bool = False
_rate_limiting_enabled: bool = False
_rate_limit_requests: int = 5
_rate_limit_window: int = 1
_RUNTIME_ENV_KEYS = {
    "MLSDM_RUNTIME_MODE",
    "MLSDM_WORKERS",
    "MLSDM_RELOAD",
    "MLSDM_LOG_LEVEL",
    "MLSDM_TIMEOUT_KEEP_ALIVE",
    "MLSDM_RATE_LIMIT_ENABLED",
    "MLSDM_SECURE_MODE",
    "MLSDM_ENGINE_ENABLE_METRICS",
    "MLSDM_DEBUG",
}


def _get_env_bool(key: str, default: bool = False) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _get_env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


# Initialize OpenTelemetry tracing
# Can be disabled via OTEL_SDK_DISABLED=true or OTEL_EXPORTER_TYPE=none
_exporter_type_env = os.getenv("OTEL_EXPORTER_TYPE", "none")
# Validate exporter type
_valid_exporter_types = ("console", "otlp", "jaeger", "none")
_exporter_type = _exporter_type_env if _exporter_type_env in _valid_exporter_types else "none"
_tracing_config = TracingConfig(
    exporter_type=_exporter_type,  # type: ignore[arg-type]  # Validated above
)
initialize_tracing(_tracing_config)

def _resolve_runtime_mode(runtime_mode: RuntimeMode | str | None) -> RuntimeMode:
    if runtime_mode is None:
        return get_runtime_mode()
    if isinstance(runtime_mode, RuntimeMode):
        return runtime_mode
    try:
        return RuntimeMode(runtime_mode)
    except ValueError:
        logger.warning("Unknown runtime mode '%s', falling back to dev", runtime_mode)
        return RuntimeMode.DEV


def _load_config_with_runtime_policy(
    config_path: str, runtime_mode: RuntimeMode | str
) -> tuple[dict[str, Any], str, RuntimeMode]:
    mode_enum = _resolve_runtime_mode(runtime_mode)
    if not isinstance(config_path, str) or not config_path:
        raise ValueError("CONFIG_PATH must be a non-empty string")

    removed_env: dict[str, str] = {}
    for key in _RUNTIME_ENV_KEYS:
        if key in os.environ:
            removed_env[key] = os.environ.pop(key)

    try:
        if os.path.exists(config_path):
            return ConfigLoader.load_config(config_path), config_path, mode_enum

        if mode_enum in STRICT_CONFIG_MODES:
            raise FileNotFoundError(
                f"Configuration file not found: {config_path} (mode={mode_enum.value})"
            )

        logger.warning(
            "Configuration file not found at %s; falling back to default config for mode %s",
            config_path,
            mode_enum.value,
        )
        return ConfigLoader.load_config(DEFAULT_CONFIG_PATH), DEFAULT_CONFIG_PATH, mode_enum
    finally:
        os.environ.update(removed_env)


def _initialize_runtime_components(config_data: dict[str, Any]) -> None:
    global _manager, _neuro_engine, _rate_limiter, _validator
    global _rate_limiting_enabled, _rate_limit_requests, _rate_limit_window, _secure_mode_enabled

    _manager = MemoryManager(config_data)

    _secure_mode_enabled = _get_env_bool("MLSDM_SECURE_MODE", False)
    _rate_limiting_enabled = _get_env_bool(
        "MLSDM_RATE_LIMIT_ENABLED", True
    ) and not _get_env_bool("DISABLE_RATE_LIMIT", False)
    _rate_limit_requests = _get_env_int("RATE_LIMIT_REQUESTS", 5)
    _rate_limit_window = _get_env_int("RATE_LIMIT_WINDOW", 1)

    if _rate_limit_requests <= 0 or _rate_limit_window <= 0:
        security_logger.log_security_config_error(
            "invalid_rate_limit",
            "RATE_LIMIT_REQUESTS and RATE_LIMIT_WINDOW must be positive integers; "
            "falling back to 5 requests per second.",
        )
        _rate_limit_requests = 5
        _rate_limit_window = 1

    rate = _rate_limit_requests / _rate_limit_window
    capacity = max(1, _rate_limit_requests)
    _rate_limiter = RateLimiter(rate=rate, capacity=capacity)
    _validator = InputValidator()

    engine_config = NeuroEngineConfig(
        dim=_manager.dimension,
        enable_fslgs=False,  # FSLGS is optional
        enable_metrics=True,
    )
    _neuro_engine = build_neuro_engine_from_env(config=engine_config)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Modern lifespan context manager for FastAPI startup/shutdown events."""
    global _cpu_background_task

    # STARTUP
    lifecycle = get_lifecycle_manager()
    manager = _require_manager()
    await lifecycle.startup()

    # Register cleanup tasks
    lifecycle.register_cleanup(lambda: cleanup_memory_manager(manager))

    # Start CPU background sampler for non-blocking health checks
    _cpu_background_task = asyncio.create_task(_cpu_background_sampler())
    logger.info("Started CPU background sampler for health checks")

    # Legacy warmup still useful as immediate fallback
    try:
        psutil.cpu_percent(interval=0.1)
    except Exception as e:
        logger.warning(f"Failed to initialize CPU monitoring: {e}")

    # Log system startup
    security_logger.log_system_event(
        SecurityEventType.STARTUP,
        "MLSDM Governed Cognitive Memory API started",
        additional_data={"version": "1.0.0", "dimension": manager.dimension},
    )

    yield

    # SHUTDOWN
    # Cancel CPU background sampler gracefully
    if _cpu_background_task and not _cpu_background_task.done():
        _cpu_background_task.cancel()
        try:
            await _cpu_background_task
        except asyncio.CancelledError:
            logger.debug("CPU background sampler cancelled during shutdown")
        logger.info("Stopped CPU background sampler")

    # Log system shutdown
    security_logger.log_system_event(
        SecurityEventType.SHUTDOWN, "MLSDM Governed Cognitive Memory API shutting down"
    )

    shutdown_tracing()
    await lifecycle.shutdown()


def create_app(
    *,
    config_path: str | None = None,
    runtime_mode: RuntimeMode | str | None = None,
) -> FastAPI:
    """Create a FastAPI application instance with runtime-aware config loading."""
    global _config_data, _config_path, _runtime_mode, app, _app_instance

    if _app_instance is not None:
        return _app_instance

    path_candidate = config_path or os.getenv("CONFIG_PATH") or DEFAULT_CONFIG_PATH
    if not isinstance(path_candidate, str) or not path_candidate:
        raise ValueError("CONFIG_PATH must be a non-empty string")
    config, resolved_path, resolved_mode = _load_config_with_runtime_policy(
        path_candidate, runtime_mode or get_runtime_mode()
    )

    _config_data = config
    _config_path = resolved_path
    _runtime_mode = resolved_mode
    _initialize_runtime_components(config)

    fastapi_app = FastAPI(
        title="mlsdm-governed-cognitive-memory",
        version="1.0.0",
        description="Production-ready neurobiologically-grounded cognitive architecture",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    fastapi_app.add_middleware(SecurityHeadersMiddleware)
    fastapi_app.add_middleware(RequestIDMiddleware)
    fastapi_app.add_middleware(TimeoutMiddleware)
    fastapi_app.add_middleware(PriorityMiddleware)
    fastapi_app.add_middleware(BulkheadMiddleware)

    fastapi_app.include_router(health.router)

    if _manager is not None:
        health.set_memory_manager(_manager)
    if _neuro_engine is not None:
        health.set_neuro_engine(_neuro_engine)

    app = fastapi_app
    _app_instance = fastapi_app
    return fastapi_app


# Initialize FastAPI with production-ready settings (canonical singleton for import string)
app = create_app()


def _require_manager() -> MemoryManager:
    if _manager is None:
        raise RuntimeError("Application not initialized; call create_app() first.")
    return _manager


def _require_rate_limiter() -> RateLimiter:
    if _rate_limiter is None:
        raise RuntimeError("Rate limiter not initialized; call create_app() first.")
    return _rate_limiter


def _require_validator() -> InputValidator:
    if _validator is None:
        raise RuntimeError("Validator not initialized; call create_app() first.")
    return _validator


def _require_engine() -> Any:
    if _neuro_engine is None:
        raise RuntimeError("Neuro engine not initialized; call create_app() first.")
    return _neuro_engine


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


def _rate_limit_message() -> str:
    if _rate_limit_window <= 1:
        return f"Rate limit exceeded. Maximum {_rate_limit_requests} requests per second."
    return (
        "Rate limit exceeded. Maximum "
        f"{_rate_limit_requests} requests per {_rate_limit_window} seconds."
    )


def _get_client_id(request: Request) -> str:
    """Get pseudonymized client identifier from request.

    Uses SHA256 hash of IP + User-Agent to create a unique but
    non-PII identifier for rate limiting and audit logging.
    """
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "")

    # Create hash for pseudonymization (no PII stored)
    identifier = f"{client_ip}:{user_agent}"
    hashed = hashlib.sha256(identifier.encode()).hexdigest()[:16]
    return hashed


async def get_current_user(token: str | None = Depends(oauth2_scheme)) -> str | None:
    """Authenticate user with enhanced security logging."""
    api_key = os.getenv("API_KEY")

    if _secure_mode_enabled and not api_key:
        security_logger.log_security_config_error(
            "missing_api_key",
            "MLSDM_SECURE_MODE enabled but API_KEY is not configured.",
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured.",
        )

    if api_key and token != api_key:
        reason = "Missing token" if not token else "Invalid token"
        security_logger.log_auth_failure(client_id="unknown", reason=reason)
        raise HTTPException(status_code=401, detail="Invalid authentication")

    if token:
        security_logger.log_auth_success(client_id="unknown")
    return token


class EventInput(BaseModel):
    event_vector: list[float]
    moral_value: float


class StateResponse(BaseModel):
    L1_norm: float
    L2_norm: float
    L3_norm: float
    current_phase: str
    latent_events_count: int
    accepted_events_count: int
    total_events_processed: int
    moral_filter_threshold: float


# Request/Response models for /generate endpoint
class GenerateRequest(BaseModel):
    """Request model for generate endpoint."""

    prompt: str = Field(..., min_length=1, description="Input text prompt to process")
    max_tokens: int | None = Field(
        None, ge=1, le=4096, description="Maximum number of tokens to generate"
    )
    moral_value: float | None = Field(None, ge=0.0, le=1.0, description="Moral threshold value")


# Request/Response models for /infer endpoint (extended API)
class InferRequest(BaseModel):
    """Request model for infer endpoint with extended governance options."""

    prompt: str = Field(..., min_length=1, description="Input text prompt to process")
    moral_value: float | None = Field(
        None, ge=0.0, le=1.0, description="Moral threshold value (default: 0.5)"
    )
    max_tokens: int | None = Field(
        None, ge=1, le=4096, description="Maximum number of tokens to generate"
    )
    secure_mode: bool = Field(
        default=False, description="Enable enhanced security filtering for sensitive contexts"
    )
    aphasia_mode: bool = Field(
        default=False, description="Enable aphasia detection and repair for output quality"
    )
    rag_enabled: bool = Field(
        default=True, description="Enable RAG-based context retrieval from memory"
    )
    context_top_k: int | None = Field(
        None, ge=1, le=100, description="Number of context items for RAG (default: 5)"
    )
    user_intent: str | None = Field(
        None, description="User intent category (e.g., 'conversational', 'analytical')"
    )


class InferResponse(BaseModel):
    """Response model for infer endpoint with detailed metadata."""

    response: str = Field(description="Generated response text")
    accepted: bool = Field(description="Whether the request was accepted")
    phase: str = Field(description="Current cognitive phase ('wake' or 'sleep')")
    moral_metadata: dict[str, Any] | None = Field(
        default=None, description="Moral filtering metadata"
    )
    aphasia_metadata: AphasiaMetadata | None = Field(
        default=None, description="Aphasia detection/repair metadata (if aphasia_mode enabled)"
    )
    rag_metadata: dict[str, Any] | None = Field(
        default=None, description="RAG retrieval metadata (context items, relevance)"
    )
    timing: dict[str, float] | None = Field(
        default=None, description="Performance timing in milliseconds"
    )
    governance: dict[str, Any] | None = Field(
        default=None, description="Full governance state information"
    )


class CognitiveStateDTO(BaseModel):
    """Cognitive state snapshot for API responses.

    CONTRACT: These fields are part of the stable API contract.
    Do not modify without a major version bump.
    """

    phase: str = Field(description="Current cognitive phase (wake/sleep)")
    stateless_mode: bool = Field(description="Whether running in stateless/degraded mode")
    emergency_shutdown: bool = Field(description="Whether emergency shutdown is active")
    memory_used_mb: float | None = Field(default=None, description="Aggregated memory usage in MB")
    moral_threshold: float | None = Field(
        default=None, description="Current moral filter threshold (0.0-1.0)"
    )


class GenerateResponse(BaseModel):
    """Response model for generate endpoint.

    CONTRACT: Core fields (always present, part of stable API contract):
    - response: Generated text
    - accepted: Whether the request was morally accepted
    - phase: Current cognitive phase
    - moral_score: Moral score used for this request
    - aphasia_flags: Aphasia detection flags (if available)
    - emergency_shutdown: Whether system is in emergency shutdown state
    - cognitive_state: Aggregated cognitive state snapshot

    Optional metrics/diagnostics:
    - metrics: Performance and timing information
    - safety_flags: Safety-related validation results
    - memory_stats: Memory state statistics
    """

    # Core contract fields (always present)
    response: str = Field(description="Generated response text")
    accepted: bool = Field(description="Whether the request was morally accepted")
    phase: str = Field(description="Current cognitive phase")
    moral_score: float | None = Field(default=None, description="Moral score used for this request")
    aphasia_flags: dict[str, Any] | None = Field(
        default=None, description="Aphasia detection flags (if available)"
    )
    emergency_shutdown: bool = Field(
        default=False, description="Whether system is in emergency shutdown state"
    )
    cognitive_state: CognitiveStateDTO | None = Field(
        default=None, description="Aggregated cognitive state snapshot (stable fields only)"
    )

    # Optional diagnostic fields
    metrics: dict[str, Any] | None = Field(default=None, description="Performance timing metrics")
    safety_flags: dict[str, Any] | None = Field(
        default=None, description="Safety validation results"
    )
    memory_stats: dict[str, Any] | None = Field(default=None, description="Memory state statistics")


class ErrorDetail(BaseModel):
    """Structured error detail."""

    error_type: str = Field(description="Type of error")
    message: str = Field(description="Human-readable error message")
    details: dict[str, Any] | None = Field(default=None, description="Additional error details")


class ErrorResponse(BaseModel):
    """Structured error response."""

    error: ErrorDetail


@app.post("/v1/process_event/", response_model=StateResponse)
async def process_event(
    event: EventInput, request: Request, user: str = Depends(get_current_user)
) -> StateResponse:
    """Process event with comprehensive security validation.

    Implements rate limiting, input validation, and audit logging
    as specified in SECURITY_POLICY.md.
    """
    client_id = _get_client_id(request)
    rate_limiter = _require_rate_limiter()
    validator = _require_validator()
    manager = _require_manager()

    # Rate limiting check (can be disabled for testing)
    if _rate_limiting_enabled and not rate_limiter.is_allowed(client_id):
        security_logger.log_rate_limit_exceeded(client_id=client_id)
        raise HTTPException(status_code=429, detail=_rate_limit_message())

    # Validate moral value
    try:
        moral_value = validator.validate_moral_value(event.moral_value)
    except ValueError as e:
        security_logger.log_invalid_input(client_id=client_id, error_message=str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Validate and convert vector
    try:
        vec = validator.validate_vector(event.event_vector, expected_dim=manager.dimension, normalize=False)
    except ValueError as e:
        security_logger.log_invalid_input(client_id=client_id, error_message=str(e))
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Process the event
    await manager.process_event(vec, moral_value)

    return await get_state(request, user)


@app.get("/v1/state/", response_model=StateResponse)
async def get_state(request: Request, user: str = Depends(get_current_user)) -> StateResponse:
    """Get system state with rate limiting."""
    client_id = _get_client_id(request)
    rate_limiter = _require_rate_limiter()
    manager = _require_manager()

    # Rate limiting check (can be disabled for testing)
    if _rate_limiting_enabled and not rate_limiter.is_allowed(client_id):
        security_logger.log_rate_limit_exceeded(client_id=client_id)
        raise HTTPException(status_code=429, detail=_rate_limit_message())

    L1, L2, L3 = manager.memory.get_state()
    metrics = manager.metrics_collector.get_metrics()
    return StateResponse(
        L1_norm=float(np.linalg.norm(L1)),
        L2_norm=float(np.linalg.norm(L2)),
        L3_norm=float(np.linalg.norm(L3)),
        current_phase=manager.rhythm.get_current_phase(),
        latent_events_count=int(metrics["latent_events_count"]),
        accepted_events_count=int(metrics["accepted_events_count"]),
        total_events_processed=int(metrics["total_events_processed"]),
        moral_filter_threshold=float(manager.filter.threshold),
    )


@app.post(
    "/generate",
    response_model=GenerateResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    tags=["Generation"],
)
async def generate(
    request_body: GenerateRequest,
    request: Request,
    _user: str | None = Depends(get_current_user),
) -> GenerateResponse | JSONResponse:
    """Generate a response using the NeuroCognitiveEngine.

    This endpoint processes the input prompt through the complete cognitive pipeline,
    including moral filtering, memory retrieval, and rhythm management.

    Args:
        request_body: Generation request parameters.
        request: FastAPI request object.

    Returns:
        Generated response with core fields (response, phase, accepted)
        plus optional metrics, safety_flags, and memory_stats.

    Raises:
        HTTPException: 400 for invalid input, 429 for rate limit, 500 for internal error.
    """
    client_id = _get_client_id(request)
    request_id = getattr(request.state, "request_id", None)
    rate_limiter = _require_rate_limiter()
    engine = _require_engine()

    # Start root span for the generate endpoint
    tracer_manager = get_tracer_manager()
    span_kind = SpanKind.SERVER if OTEL_AVAILABLE and SpanKind is not None else None
    with tracer_manager.start_span(
        "api.generate",
        kind=span_kind,
        attributes={
            "http.method": "POST",
            "http.route": "/generate",
            "mlsdm.prompt_length": len(request_body.prompt) if request_body.prompt else 0,
            "mlsdm.max_tokens": request_body.max_tokens or 512,
            "mlsdm.moral_value": request_body.moral_value or 0.5,
        },
    ) as span:
        # Add request_id to span for correlation
        if request_id:
            span.set_attribute("mlsdm.request_id", request_id)

        # Rate limiting check (can be disabled for testing)
        if _rate_limiting_enabled and not rate_limiter.is_allowed(client_id):
            security_logger.log_rate_limit_exceeded(client_id=client_id)
            span.set_attribute("mlsdm.rate_limited", True)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": {
                        "error_type": "rate_limit_exceeded",
                        "message": _rate_limit_message(),
                        "details": None,
                    }
                },
            )

        # Validate prompt
        if not request_body.prompt or not request_body.prompt.strip():
            security_logger.log_invalid_input(
                client_id=client_id, error_message="Prompt cannot be empty"
            )
            span.set_attribute("mlsdm.validation_error", "empty_prompt")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": {
                        "error_type": "validation_error",
                        "message": "Prompt cannot be empty",
                        "details": {"field": "prompt"},
                    }
                },
            )

        try:
            # Build kwargs for engine
            kwargs: dict[str, Any] = {"prompt": request_body.prompt}
            if request_body.max_tokens is not None:
                kwargs["max_tokens"] = request_body.max_tokens
            if request_body.moral_value is not None:
                kwargs["moral_value"] = request_body.moral_value

            # Generate response (engine has its own child spans)
            result: dict[str, Any] = engine.generate(**kwargs)

            # Extract phase from mlsdm state if available
            mlsdm_state = result.get("mlsdm", {})
            phase = mlsdm_state.get("phase", "unknown")

            # Determine if request was accepted (no rejection and has response)
            rejected_at = result.get("rejected_at")
            error_info = result.get("error")
            accepted = rejected_at is None and error_info is None and bool(result.get("response"))

            # Add result attributes to span
            add_span_attributes(
                span,
                **{
                    "mlsdm.phase": phase,
                    "mlsdm.accepted": accepted,
                    "mlsdm.rejected_at": rejected_at or "",
                    "mlsdm.response_length": len(result.get("response", "")),
                },
            )

            # Build safety flags from validation steps
            safety_flags = None
            validation_steps = result.get("validation_steps", [])
            if validation_steps:
                safety_flags = {
                    "validation_steps": validation_steps,
                    "rejected_at": rejected_at,
                }

            # Build metrics from timing info
            metrics = None
            timing = result.get("timing")
            if timing:
                metrics = {"timing": timing}
                # Add timing to span
                if "total" in timing:
                    span.set_attribute("mlsdm.latency_ms", timing["total"])

            # Build memory stats from mlsdm state
            memory_stats = None
            if mlsdm_state:
                memory_stats = {
                    "step": mlsdm_state.get("step"),
                    "moral_threshold": mlsdm_state.get("moral_threshold"),
                    "context_items": mlsdm_state.get("context_items"),
                }

            # Extract moral_score from request or mlsdm state
            moral_score = request_body.moral_value
            if moral_score is None:
                moral_score = mlsdm_state.get("moral_threshold")

            # Extract aphasia_flags from speech_governance if available
            aphasia_flags = None
            speech_gov = mlsdm_state.get("speech_governance")
            if speech_gov and "metadata" in speech_gov:
                aphasia_report = speech_gov["metadata"].get("aphasia_report")
                if aphasia_report:
                    aphasia_flags = {
                        "is_aphasic": aphasia_report.get("is_aphasic", False),
                        "severity": aphasia_report.get("severity", 0.0),
                    }

            # Build cognitive_state snapshot (stable, safe fields only)
            # CONTRACT: These fields are part of the stable API contract
            cognitive_state = CognitiveStateDTO(
                phase=phase,
                stateless_mode=mlsdm_state.get("stateless_mode", False),
                emergency_shutdown=False,  # Engine doesn't have controller-level shutdown
                memory_used_mb=mlsdm_state.get("memory_used_mb"),
                moral_threshold=mlsdm_state.get("moral_threshold"),
            )

            return GenerateResponse(
                response=result.get("response", ""),
                accepted=accepted,
                phase=phase,
                moral_score=moral_score,
                aphasia_flags=aphasia_flags,
                emergency_shutdown=False,  # Engine doesn't have controller-level shutdown
                cognitive_state=cognitive_state,
                metrics=metrics,
                safety_flags=safety_flags,
                memory_stats=memory_stats,
            )

        except Exception as e:
            # Log the error but don't expose stack trace in response
            logger.exception("Error in generate endpoint")
            security_logger.log_invalid_input(
                client_id=client_id, error_message=f"Internal error: {type(e).__name__}"
            )
            # Record exception on span
            tracer_manager.record_exception(span, e)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": {
                        "error_type": "internal_error",
                        "message": "An internal error occurred. Please try again later.",
                        "details": None,
                    }
                },
            )


@app.post(
    "/infer",
    response_model=InferResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    tags=["Inference"],
)
async def infer(
    request_body: InferRequest,
    request: Request,
) -> InferResponse | JSONResponse:
    """Generate a response with extended governance options.

    This endpoint provides fine-grained control over the cognitive pipeline,
    including secure mode, aphasia detection/repair, and RAG retrieval.

    Features:
    - **secure_mode**: Enhanced filtering for sensitive/security-critical contexts
    - **aphasia_mode**: Detection and repair of telegraphic/fragmented speech patterns
    - **rag_enabled**: Context retrieval from cognitive memory for coherent responses

    Args:
        request_body: Inference request parameters with governance options.
        request: FastAPI request object.

    Returns:
        InferResponse with response text and detailed governance metadata.
    """
    client_id = _get_client_id(request)
    request_id = getattr(request.state, "request_id", None)
    rate_limiter = _require_rate_limiter()
    engine = _require_engine()

    # Start root span for the infer endpoint
    tracer_manager = get_tracer_manager()
    span_kind = SpanKind.SERVER if OTEL_AVAILABLE and SpanKind is not None else None
    with tracer_manager.start_span(
        "api.infer",
        kind=span_kind,
        attributes={
            "http.method": "POST",
            "http.route": "/infer",
            "mlsdm.prompt_length": len(request_body.prompt) if request_body.prompt else 0,
            "mlsdm.secure_mode": request_body.secure_mode,
            "mlsdm.aphasia_mode": request_body.aphasia_mode,
            "mlsdm.rag_enabled": request_body.rag_enabled,
        },
    ) as span:
        # Add request_id to span for correlation
        if request_id:
            span.set_attribute("mlsdm.request_id", request_id)

        # Rate limiting check
        if _rate_limiting_enabled and not rate_limiter.is_allowed(client_id):
            security_logger.log_rate_limit_exceeded(client_id=client_id)
            span.set_attribute("mlsdm.rate_limited", True)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": {
                        "error_type": "rate_limit_exceeded",
                        "message": _rate_limit_message(),
                        "details": None,
                    }
                },
            )

        # Validate prompt
        if not request_body.prompt or not request_body.prompt.strip():
            security_logger.log_invalid_input(
                client_id=client_id, error_message="Prompt cannot be empty"
            )
            span.set_attribute("mlsdm.validation_error", "empty_prompt")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "error": {
                        "error_type": "validation_error",
                        "message": "Prompt cannot be empty",
                        "details": {"field": "prompt"},
                    }
                },
            )

        try:
            # Build kwargs for engine
            kwargs: dict[str, Any] = {"prompt": request_body.prompt}

            if request_body.max_tokens is not None:
                kwargs["max_tokens"] = request_body.max_tokens

            # Apply moral_value with secure_mode boost
            base_moral = request_body.moral_value if request_body.moral_value is not None else 0.5
            if request_body.secure_mode:
                # In secure mode, raise moral threshold by 0.2 (capped at 1.0)
                kwargs["moral_value"] = min(1.0, base_moral + 0.2)
            else:
                kwargs["moral_value"] = base_moral

            if request_body.user_intent is not None:
                kwargs["user_intent"] = request_body.user_intent

            # RAG control via context_top_k
            if request_body.rag_enabled:
                kwargs["context_top_k"] = request_body.context_top_k or 5
            else:
                # Disable RAG by setting context_top_k to 0
                kwargs["context_top_k"] = 0

            # Generate response (engine has its own child spans)
            result: dict[str, Any] = engine.generate(**kwargs)

            # Extract state
            mlsdm_state = result.get("mlsdm", {})
            phase = mlsdm_state.get("phase", "unknown")
            rejected_at = result.get("rejected_at")
            error_info = result.get("error")
            accepted = rejected_at is None and error_info is None and bool(result.get("response"))

            # Add result attributes to span
            add_span_attributes(
                span,
                **{
                    "mlsdm.phase": phase,
                    "mlsdm.accepted": accepted,
                    "mlsdm.rejected_at": rejected_at or "",
                    "mlsdm.response_length": len(result.get("response", "")),
                },
            )

            # Build moral metadata
            moral_metadata = {
                "threshold": mlsdm_state.get("moral_threshold"),
                "secure_mode": request_body.secure_mode,
                "applied_moral_value": kwargs.get("moral_value"),
            }

            # Build RAG metadata
            rag_metadata = None
            if request_body.rag_enabled:
                rag_metadata = {
                    "enabled": True,
                    "context_items_retrieved": mlsdm_state.get("context_items", 0),
                    "top_k": kwargs.get("context_top_k", 5),
                }
            else:
                rag_metadata = {
                    "enabled": False,
                    "context_items_retrieved": 0,
                    "top_k": 0,
                }

            # Build aphasia metadata (detection happens at speech governance level)
            aphasia_metadata: AphasiaMetadata | None = None
            if request_body.aphasia_mode:
                speech_gov = result.get("mlsdm", {}).get("speech_governance")
                if speech_gov and "metadata" in speech_gov:
                    aphasia_report = speech_gov["metadata"].get("aphasia_report")
                    aphasia_metadata = AphasiaMetadata(
                        enabled=True,
                        detected=aphasia_report.get("is_aphasic") if aphasia_report else False,
                        severity=aphasia_report.get("severity") if aphasia_report else 0.0,
                        repaired=speech_gov["metadata"].get("repaired", False),
                    )
                    # Add aphasia info to span
                    if aphasia_report:
                        span.set_attribute(
                            "mlsdm.aphasia.detected", aphasia_report.get("is_aphasic", False)
                        )
                        span.set_attribute(
                            "mlsdm.aphasia.severity", aphasia_report.get("severity", 0.0)
                        )
                else:
                    # Aphasia mode requested but no speech governor configured
                    aphasia_metadata = AphasiaMetadata(
                        enabled=True,
                        detected=False,
                        severity=0.0,
                        repaired=False,
                        note="aphasia_mode enabled but no speech governor configured",
                    )

            # Timing info
            timing = result.get("timing")
            if timing and "total" in timing:
                span.set_attribute("mlsdm.latency_ms", timing["total"])

            return InferResponse(
                response=result.get("response", ""),
                accepted=accepted,
                phase=phase,
                moral_metadata=moral_metadata,
                aphasia_metadata=aphasia_metadata,
                rag_metadata=rag_metadata,
                timing=timing,
                governance=result.get("governance"),
            )

        except Exception as e:
            logger.exception("Error in infer endpoint")
            security_logger.log_invalid_input(
                client_id=client_id, error_message=f"Internal error: {type(e).__name__}"
            )
            # Record exception on span
            tracer_manager.record_exception(span, e)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": {
                        "error_type": "internal_error",
                        "message": "An internal error occurred. Please try again later.",
                        "details": None,
                    }
                },
            )


@app.get("/status", tags=["Health"])
async def get_status() -> dict[str, Any]:
    """Get extended service status with system info.

    Returns system status including version, backend, memory usage,
    and configuration info.
    """
    import psutil

    manager = _require_manager()

    return {
        "status": "ok",
        "version": "1.2.0",
        "backend": os.environ.get("LLM_BACKEND", "local_stub"),
        "system": {
            "memory_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 2),
            "cpu_percent": psutil.cpu_percent(),
        },
        "config": {
            "dimension": manager.dimension,
            "rate_limiting_enabled": _rate_limiting_enabled,
            "rate_limit_requests": _rate_limit_requests,
            "rate_limit_window": _rate_limit_window,
            "secure_mode": _secure_mode_enabled,
        },
    }
