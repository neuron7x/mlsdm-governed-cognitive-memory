import hashlib
import logging
import os
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field

from mlsdm.api import health
from mlsdm.api.lifecycle import cleanup_memory_manager, get_lifecycle_manager
from mlsdm.api.middleware import RequestIDMiddleware, SecurityHeadersMiddleware
from mlsdm.core.memory_manager import MemoryManager
from mlsdm.engine import NeuroEngineConfig, build_neuro_engine_from_env
from mlsdm.utils.config_loader import ConfigLoader
from mlsdm.utils.input_validator import InputValidator
from mlsdm.utils.rate_limiter import RateLimiter
from mlsdm.utils.security_logger import SecurityEventType, get_security_logger

logger = logging.getLogger(__name__)
security_logger = get_security_logger()

# Initialize FastAPI with production-ready settings
app = FastAPI(
    title="mlsdm-governed-cognitive-memory",
    version="1.0.0",
    description="Production-ready neurobiologically-grounded cognitive architecture",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add production middleware
app.add_middleware(RequestIDMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

# Include health check router
app.include_router(health.router)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

_config_path = os.getenv("CONFIG_PATH", "config/default_config.yaml")
_manager = MemoryManager(ConfigLoader.load_config(_config_path))

# Set memory manager for health checks
health.set_memory_manager(_manager)

# Initialize rate limiter (5 RPS per client as per SECURITY_POLICY.md)
# Can be disabled in testing with DISABLE_RATE_LIMIT=1
_rate_limiting_enabled = os.getenv("DISABLE_RATE_LIMIT") != "1"
_rate_limiter = RateLimiter(rate=5.0, capacity=10)

# Initialize input validator
_validator = InputValidator()

# Initialize NeuroCognitiveEngine for /generate endpoint
_engine_config = NeuroEngineConfig(
    dim=_manager.dimension,
    enable_fslgs=False,  # FSLGS is optional
    enable_metrics=True,
)
_neuro_engine = build_neuro_engine_from_env(config=_engine_config)


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


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """Authenticate user with enhanced security logging."""
    api_key = os.getenv("API_KEY")

    if api_key and token != api_key:
        security_logger.log_auth_failure(
            client_id="unknown",
            reason="Invalid token"
        )
        raise HTTPException(status_code=401, detail="Invalid authentication")

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
    moral_value: float | None = Field(
        None, ge=0.0, le=1.0, description="Moral threshold value"
    )


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
        default=False,
        description="Enable enhanced security filtering for sensitive contexts"
    )
    aphasia_mode: bool = Field(
        default=False,
        description="Enable aphasia detection and repair for output quality"
    )
    rag_enabled: bool = Field(
        default=True,
        description="Enable RAG-based context retrieval from memory"
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
    aphasia_metadata: dict[str, Any] | None = Field(
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


class GenerateResponse(BaseModel):
    """Response model for generate endpoint.

    Core fields (always present):
    - response: Generated text
    - phase: Current cognitive phase
    - accepted: Whether the request was accepted

    Optional metrics/diagnostics:
    - metrics: Performance and timing information
    - safety_flags: Safety-related validation results
    - memory_stats: Memory state statistics
    """

    response: str = Field(description="Generated response text")
    phase: str = Field(description="Current cognitive phase")
    accepted: bool = Field(description="Whether the request was accepted")
    metrics: dict[str, Any] | None = Field(
        default=None, description="Performance timing metrics"
    )
    safety_flags: dict[str, Any] | None = Field(
        default=None, description="Safety validation results"
    )
    memory_stats: dict[str, Any] | None = Field(
        default=None, description="Memory state statistics"
    )


class ErrorDetail(BaseModel):
    """Structured error detail."""

    error_type: str = Field(description="Type of error")
    message: str = Field(description="Human-readable error message")
    details: dict[str, Any] | None = Field(
        default=None, description="Additional error details"
    )


class ErrorResponse(BaseModel):
    """Structured error response."""

    error: ErrorDetail


@app.post("/v1/process_event/", response_model=StateResponse)
async def process_event(
    event: EventInput,
    request: Request,
    user: str = Depends(get_current_user)
) -> StateResponse:
    """Process event with comprehensive security validation.

    Implements rate limiting, input validation, and audit logging
    as specified in SECURITY_POLICY.md.
    """
    client_id = _get_client_id(request)

    # Rate limiting check (can be disabled for testing)
    if _rate_limiting_enabled and not _rate_limiter.is_allowed(client_id):
        security_logger.log_rate_limit_exceeded(client_id=client_id)
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Maximum 5 requests per second."
        )

    # Validate moral value
    try:
        moral_value = _validator.validate_moral_value(event.moral_value)
    except ValueError as e:
        security_logger.log_invalid_input(
            client_id=client_id,
            error_message=str(e)
        )
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Validate and convert vector
    try:
        vec = _validator.validate_vector(
            event.event_vector,
            expected_dim=_manager.dimension,
            normalize=False
        )
    except ValueError as e:
        security_logger.log_invalid_input(
            client_id=client_id,
            error_message=str(e)
        )
        raise HTTPException(status_code=400, detail=str(e)) from e

    # Process the event
    await _manager.process_event(vec, moral_value)

    return await get_state(request, user)


@app.get("/v1/state/", response_model=StateResponse)
async def get_state(
    request: Request,
    user: str = Depends(get_current_user)
) -> StateResponse:
    """Get system state with rate limiting."""
    client_id = _get_client_id(request)

    # Rate limiting check (can be disabled for testing)
    if _rate_limiting_enabled and not _rate_limiter.is_allowed(client_id):
        security_logger.log_rate_limit_exceeded(client_id=client_id)
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Maximum 5 requests per second."
        )

    L1, L2, L3 = _manager.memory.get_state()
    metrics = _manager.metrics_collector.get_metrics()
    return StateResponse(
        L1_norm=float(np.linalg.norm(L1)),
        L2_norm=float(np.linalg.norm(L2)),
        L3_norm=float(np.linalg.norm(L3)),
        current_phase=_manager.rhythm.get_current_phase(),
        latent_events_count=int(metrics["latent_events_count"]),
        accepted_events_count=int(metrics["accepted_events_count"]),
        total_events_processed=int(metrics["total_events_processed"]),
        moral_filter_threshold=float(_manager.filter.threshold),
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

    # Rate limiting check (can be disabled for testing)
    if _rate_limiting_enabled and not _rate_limiter.is_allowed(client_id):
        security_logger.log_rate_limit_exceeded(client_id=client_id)
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": {
                    "error_type": "rate_limit_exceeded",
                    "message": "Rate limit exceeded. Maximum 5 requests per second.",
                    "details": None,
                }
            },
        )

    # Validate prompt
    if not request_body.prompt or not request_body.prompt.strip():
        security_logger.log_invalid_input(
            client_id=client_id,
            error_message="Prompt cannot be empty"
        )
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

        # Generate response
        result: dict[str, Any] = _neuro_engine.generate(**kwargs)

        # Extract phase from mlsdm state if available
        mlsdm_state = result.get("mlsdm", {})
        phase = mlsdm_state.get("phase", "unknown")

        # Determine if request was accepted (no rejection and has response)
        rejected_at = result.get("rejected_at")
        error_info = result.get("error")
        accepted = rejected_at is None and error_info is None and bool(result.get("response"))

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

        # Build memory stats from mlsdm state
        memory_stats = None
        if mlsdm_state:
            memory_stats = {
                "step": mlsdm_state.get("step"),
                "moral_threshold": mlsdm_state.get("moral_threshold"),
                "context_items": mlsdm_state.get("context_items"),
            }

        return GenerateResponse(
            response=result.get("response", ""),
            phase=phase,
            accepted=accepted,
            metrics=metrics,
            safety_flags=safety_flags,
            memory_stats=memory_stats,
        )

    except Exception as e:
        # Log the error but don't expose stack trace in response
        logger.exception("Error in generate endpoint")
        security_logger.log_invalid_input(
            client_id=client_id,
            error_message=f"Internal error: {type(e).__name__}"
        )
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


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize application on startup."""
    # Initialize lifecycle manager
    lifecycle = get_lifecycle_manager()
    await lifecycle.startup()

    # Register cleanup tasks
    lifecycle.register_cleanup(lambda: cleanup_memory_manager(_manager))

    # Log system startup
    security_logger.log_system_event(
        SecurityEventType.STARTUP,
        "MLSDM Governed Cognitive Memory API started",
        additional_data={
            "version": "1.0.0",
            "dimension": _manager.dimension
        }
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

    # Rate limiting check
    if _rate_limiting_enabled and not _rate_limiter.is_allowed(client_id):
        security_logger.log_rate_limit_exceeded(client_id=client_id)
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": {
                    "error_type": "rate_limit_exceeded",
                    "message": "Rate limit exceeded. Maximum 5 requests per second.",
                    "details": None,
                }
            },
        )

    # Validate prompt
    if not request_body.prompt or not request_body.prompt.strip():
        security_logger.log_invalid_input(
            client_id=client_id,
            error_message="Prompt cannot be empty"
        )
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

        # Generate response
        result: dict[str, Any] = _neuro_engine.generate(**kwargs)

        # Extract state
        mlsdm_state = result.get("mlsdm", {})
        phase = mlsdm_state.get("phase", "unknown")
        rejected_at = result.get("rejected_at")
        error_info = result.get("error")
        accepted = rejected_at is None and error_info is None and bool(result.get("response"))

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
        aphasia_metadata = None
        if request_body.aphasia_mode:
            speech_gov = result.get("mlsdm", {}).get("speech_governance")
            if speech_gov and "metadata" in speech_gov:
                aphasia_report = speech_gov["metadata"].get("aphasia_report")
                aphasia_metadata = {
                    "enabled": True,
                    "detected": aphasia_report.get("is_aphasic") if aphasia_report else False,
                    "severity": aphasia_report.get("severity") if aphasia_report else 0.0,
                    "repaired": speech_gov["metadata"].get("repaired", False),
                }
            else:
                # Aphasia mode requested but no speech governor configured
                aphasia_metadata = {
                    "enabled": True,
                    "detected": False,
                    "severity": 0.0,
                    "repaired": False,
                    "note": "aphasia_mode enabled but no speech governor configured",
                }

        # Timing info
        timing = result.get("timing")

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
            client_id=client_id,
            error_message=f"Internal error: {type(e).__name__}"
        )
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

    return {
        "status": "ok",
        "version": "1.2.0",
        "backend": os.environ.get("LLM_BACKEND", "local_stub"),
        "system": {
            "memory_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 2),
            "cpu_percent": psutil.cpu_percent(),
        },
        "config": {
            "dimension": _manager.dimension,
            "rate_limiting_enabled": _rate_limiting_enabled,
        },
    }


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up resources on shutdown."""
    # Log system shutdown
    security_logger.log_system_event(
        SecurityEventType.SHUTDOWN,
        "MLSDM Governed Cognitive Memory API shutting down"
    )

    # Execute graceful shutdown
    lifecycle = get_lifecycle_manager()
    await lifecycle.shutdown()
