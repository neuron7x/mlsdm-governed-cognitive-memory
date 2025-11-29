"""
Centralized Pydantic schemas for MLSDM API.

This module provides the stable API contract schemas for all HTTP endpoints.
These schemas define the request/response formats and should remain stable
across minor versions. Breaking changes require a major version bump.

API Contract Stability Policy:
------------------------------
The following fields are part of the stable API contract and will not be
removed or renamed without a major version bump:

GenerateResponse (stable):
    - response: str
    - phase: str
    - accepted: bool

ErrorResponse (stable):
    - error.error_type: str
    - error.message: str

HealthStatus (stable):
    - status: str
    - timestamp: float

ReadinessStatus (stable):
    - ready: bool
    - status: str
    - timestamp: float
    - checks: dict[str, bool]
"""

from typing import Any

from pydantic import BaseModel, Field

# ==============================================================================
# Request Schemas
# ==============================================================================


class GenerateRequest(BaseModel):
    """Request model for /generate endpoint.

    Contract Fields (stable):
        - prompt: Required input text
        - max_tokens: Optional token limit
        - moral_value: Optional moral threshold
    """

    prompt: str = Field(..., min_length=1, description="Input text prompt to process")
    max_tokens: int | None = Field(
        None, ge=1, le=4096, description="Maximum number of tokens to generate"
    )
    moral_value: float | None = Field(
        None, ge=0.0, le=1.0, description="Moral threshold value"
    )


class InferRequest(BaseModel):
    """Request model for /infer endpoint with extended governance options.

    Contract Fields (stable):
        - prompt: Required input text
        - moral_value: Optional moral threshold
        - secure_mode: Enhanced security filtering
        - rag_enabled: RAG context retrieval toggle
    """

    prompt: str = Field(..., min_length=1, description="Input text prompt to process")
    moral_value: float | None = Field(
        None, ge=0.0, le=1.0, description="Moral threshold value (default: 0.5)"
    )
    max_tokens: int | None = Field(
        None, ge=1, le=4096, description="Maximum number of tokens to generate"
    )
    secure_mode: bool = Field(
        default=False,
        description="Enable enhanced security filtering for sensitive contexts",
    )
    aphasia_mode: bool = Field(
        default=False,
        description="Enable aphasia detection and repair for output quality",
    )
    rag_enabled: bool = Field(
        default=True,
        description="Enable RAG-based context retrieval from memory",
    )
    context_top_k: int | None = Field(
        None, ge=1, le=100, description="Number of context items for RAG (default: 5)"
    )
    user_intent: str | None = Field(
        None, description="User intent category (e.g., 'conversational', 'analytical')"
    )


class EventInput(BaseModel):
    """Request model for event processing.

    Contract Fields (stable):
        - event_vector: Embedding vector
        - moral_value: Moral score for filtering
    """

    event_vector: list[float]
    moral_value: float


# ==============================================================================
# Response Schemas
# ==============================================================================


class GenerateResponse(BaseModel):
    """Response model for /generate endpoint.

    Contract Fields (stable, guaranteed across minor versions):
        - response: Generated text
        - phase: Current cognitive phase ('wake' or 'sleep')
        - accepted: Whether the request was accepted

    Optional Fields (may be extended):
        - metrics: Performance and timing information
        - safety_flags: Safety-related validation results
        - memory_stats: Memory state statistics
        - moral_score: Moral evaluation score (if available)
        - aphasia_flags: Aphasia detection flags (if available)
        - emergency_shutdown: Emergency shutdown indicator
        - latency_ms: Total processing latency in milliseconds
        - cognitive_state: Aggregated cognitive state snapshot
    """

    response: str = Field(description="Generated response text")
    phase: str = Field(description="Current cognitive phase ('wake' or 'sleep')")
    accepted: bool = Field(description="Whether the request was accepted")

    # Optional metrics and diagnostics
    metrics: dict[str, Any] | None = Field(
        default=None, description="Performance timing metrics"
    )
    safety_flags: dict[str, Any] | None = Field(
        default=None, description="Safety validation results"
    )
    memory_stats: dict[str, Any] | None = Field(
        default=None, description="Memory state statistics"
    )

    # Extended fields for CORE-07 cognitive state integration
    moral_score: float | None = Field(
        default=None, description="Moral evaluation score (0.0-1.0)"
    )
    aphasia_flags: dict[str, Any] | None = Field(
        default=None, description="Aphasia detection flags"
    )
    emergency_shutdown: bool | None = Field(
        default=None, description="Emergency shutdown indicator"
    )
    latency_ms: float | None = Field(
        default=None, description="Total processing latency in milliseconds"
    )
    cognitive_state: dict[str, Any] | None = Field(
        default=None, description="Aggregated cognitive state snapshot"
    )


class InferResponse(BaseModel):
    """Response model for /infer endpoint with detailed metadata.

    Contract Fields (stable):
        - response: Generated text
        - accepted: Whether the request was accepted
        - phase: Current cognitive phase

    Optional Fields (may be extended):
        - moral_metadata: Moral filtering metadata
        - aphasia_metadata: Aphasia detection/repair metadata
        - rag_metadata: RAG retrieval metadata
        - timing: Performance timing
        - governance: Full governance state
    """

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


class StateResponse(BaseModel):
    """Response model for /v1/state endpoint.

    Contract Fields (stable):
        - L1_norm, L2_norm, L3_norm: Memory layer norms
        - current_phase: Current cognitive phase
        - total_events_processed: Event count
    """

    L1_norm: float
    L2_norm: float
    L3_norm: float
    current_phase: str
    latent_events_count: int
    accepted_events_count: int
    total_events_processed: int
    moral_filter_threshold: float


# ==============================================================================
# Error Schemas
# ==============================================================================


class ErrorDetail(BaseModel):
    """Structured error detail.

    Contract Fields (stable):
        - error_type: Type/category of error
        - message: Human-readable error message

    Optional Fields:
        - details: Additional error context
        - debug_id: Request ID for debugging/correlation
    """

    error_type: str = Field(description="Type of error (e.g., 'validation_error', 'rate_limit_exceeded')")
    message: str = Field(description="Human-readable error message")
    details: dict[str, Any] | None = Field(
        default=None, description="Additional error details"
    )
    debug_id: str | None = Field(
        default=None, description="Request ID for debugging/correlation"
    )


class ErrorResponse(BaseModel):
    """Structured error response.

    Contract Fields (stable):
        - error: ErrorDetail with error information

    All error responses from the API follow this format.
    """

    error: ErrorDetail


# ==============================================================================
# Health Schemas
# ==============================================================================


class SimpleHealthStatus(BaseModel):
    """Simple health status response for basic health check.

    Contract Fields (stable):
        - status: Health status string ("healthy")
    """

    status: str


class HealthStatus(BaseModel):
    """Basic health status response for liveness probe.

    Contract Fields (stable):
        - status: Liveness status ("alive")
        - timestamp: Unix timestamp of response
    """

    status: str
    timestamp: float


class ReadinessStatus(BaseModel):
    """Readiness status response for readiness probe.

    Contract Fields (stable):
        - ready: Boolean indicating readiness
        - status: Status string ("ready" or "not_ready")
        - timestamp: Unix timestamp of response
        - checks: Individual check results

    Core checks include:
        - memory_manager: Whether memory manager is initialized
        - memory_available: System memory availability
        - cpu_available: CPU availability

    Extended checks (optional):
        - emergency_shutdown: Whether emergency shutdown is active
        - rhythm_ready: Whether cognitive rhythm is in ready state
        - moral_ready: Whether moral filter is ready
    """

    ready: bool
    status: str
    timestamp: float
    checks: dict[str, bool]


class DetailedHealthStatus(BaseModel):
    """Detailed health status response.

    Contract Fields (stable):
        - status: Health status ("healthy" or "unhealthy")
        - timestamp: Unix timestamp
        - uptime_seconds: Service uptime
        - system: System resource information

    Optional Fields:
        - memory_state: Memory layer norms
        - phase: Current cognitive phase
        - statistics: Processing statistics
        - cognitive_state: Aggregated cognitive state (safe for exposure)
    """

    status: str
    timestamp: float
    uptime_seconds: float
    system: dict[str, Any]
    memory_state: dict[str, Any] | None
    phase: str | None
    statistics: dict[str, Any] | None
    cognitive_state: dict[str, Any] | None = Field(
        default=None, description="Aggregated cognitive state snapshot (safe for exposure)"
    )


# ==============================================================================
# Export all schemas
# ==============================================================================


__all__ = [
    # Request schemas
    "GenerateRequest",
    "InferRequest",
    "EventInput",
    # Response schemas
    "GenerateResponse",
    "InferResponse",
    "StateResponse",
    # Error schemas
    "ErrorDetail",
    "ErrorResponse",
    # Health schemas
    "SimpleHealthStatus",
    "HealthStatus",
    "ReadinessStatus",
    "DetailedHealthStatus",
]
