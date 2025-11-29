"""
Stable Pydantic schemas for MLSDM API.

This module defines the canonical API contract for the MLSDM service.
All request/response schemas are defined here to ensure a stable,
documented interface that can be safely used in production.

## Contract Stability Guarantees

The following fields are considered **stable contract** and will not
change without a major version bump (breaking change):

### GenerateResponse (stable fields):
- response: str
- phase: str
- accepted: bool

### HealthStatus (stable fields):
- status: str
- timestamp: float

### ReadinessStatus (stable fields):
- ready: bool
- status: str
- timestamp: float
- checks: dict[str, bool]

### ErrorResponse (stable fields):
- error.error_type: str
- error.message: str
- error.details: dict | None

Optional/diagnostic fields may be added or modified in minor versions.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# =============================================================================
# Request Schemas
# =============================================================================


class GenerateRequest(BaseModel):
    """Request model for /generate endpoint.

    Attributes:
        prompt: Input text prompt to process (required, min 1 char).
        max_tokens: Maximum number of tokens to generate (1-4096).
        moral_value: Moral threshold value (0.0-1.0).
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

    Attributes:
        prompt: Input text prompt to process (required, min 1 char).
        moral_value: Moral threshold value (0.0-1.0), default 0.5.
        max_tokens: Maximum number of tokens to generate (1-4096).
        secure_mode: Enable enhanced security filtering.
        aphasia_mode: Enable aphasia detection and repair.
        rag_enabled: Enable RAG context retrieval, default True.
        context_top_k: Number of context items for RAG (1-100).
        user_intent: User intent category string.
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
    """Request model for event processing endpoint.

    Attributes:
        event_vector: Event embedding vector.
        moral_value: Moral value for filtering.
    """

    event_vector: list[float]
    moral_value: float


# =============================================================================
# Response Schemas
# =============================================================================


class GenerateResponse(BaseModel):
    """Response model for /generate endpoint.

    ## Contract Stability
    The following fields are **stable contract** (won't change without major version):
    - response: str - Generated response text
    - phase: str - Current cognitive phase
    - accepted: bool - Whether the request was accepted

    Optional diagnostic fields may change in minor versions:
    - metrics: Performance timing metrics
    - safety_flags: Safety validation results
    - memory_stats: Memory state statistics
    - moral_score: Moral evaluation score
    - aphasia_flags: Aphasia detection flags
    - emergency_shutdown: Emergency shutdown status
    - latency_ms: Request latency in milliseconds
    - cognitive_state: Cognitive state snapshot
    """

    # Stable contract fields (do not modify without major version bump)
    response: str = Field(description="Generated response text")
    phase: str = Field(description="Current cognitive phase ('wake' or 'sleep')")
    accepted: bool = Field(description="Whether the request was accepted")

    # Optional diagnostic fields (may change in minor versions)
    metrics: dict[str, Any] | None = Field(
        default=None, description="Performance timing metrics"
    )
    safety_flags: dict[str, Any] | None = Field(
        default=None, description="Safety validation results"
    )
    memory_stats: dict[str, Any] | None = Field(
        default=None, description="Memory state statistics"
    )
    moral_score: float | None = Field(
        default=None, description="Moral evaluation score (0.0-1.0)"
    )
    aphasia_flags: dict[str, Any] | None = Field(
        default=None, description="Aphasia detection flags"
    )
    emergency_shutdown: bool | None = Field(
        default=None, description="Whether emergency shutdown is active"
    )
    latency_ms: float | None = Field(
        default=None, description="Request latency in milliseconds"
    )
    cognitive_state: dict[str, Any] | None = Field(
        default=None, description="Cognitive state snapshot"
    )


class InferResponse(BaseModel):
    """Response model for /infer endpoint with detailed metadata.

    Attributes:
        response: Generated response text.
        accepted: Whether the request was accepted.
        phase: Current cognitive phase.
        moral_metadata: Moral filtering metadata.
        aphasia_metadata: Aphasia detection/repair metadata.
        rag_metadata: RAG retrieval metadata.
        timing: Performance timing in milliseconds.
        governance: Full governance state information.
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
    """Response model for system state endpoint.

    Attributes:
        L1_norm: L1 memory layer norm.
        L2_norm: L2 memory layer norm.
        L3_norm: L3 memory layer norm.
        current_phase: Current cognitive phase.
        latent_events_count: Number of latent events.
        accepted_events_count: Number of accepted events.
        total_events_processed: Total events processed.
        moral_filter_threshold: Current moral filter threshold.
    """

    L1_norm: float
    L2_norm: float
    L3_norm: float
    current_phase: str
    latent_events_count: int
    accepted_events_count: int
    total_events_processed: int
    moral_filter_threshold: float


# =============================================================================
# Health Schemas
# =============================================================================


class SimpleHealthStatus(BaseModel):
    """Simple health status response for basic health check.

    ## Contract Stability
    This is a stable contract field:
    - status: str - Health status ("healthy")
    """

    status: str


class HealthStatus(BaseModel):
    """Basic health status response.

    ## Contract Stability
    These are stable contract fields:
    - status: str - Liveness status ("alive")
    - timestamp: float - Unix timestamp
    """

    status: str
    timestamp: float


class ReadinessStatus(BaseModel):
    """Readiness status response.

    ## Contract Stability
    These are stable contract fields:
    - ready: bool - Whether service is ready
    - status: str - "ready" or "not_ready"
    - timestamp: float - Unix timestamp
    - checks: dict[str, bool] - Individual check results
    """

    ready: bool
    status: str
    timestamp: float
    checks: dict[str, bool]
    # Optional fields (may change in minor versions)
    emergency_shutdown: bool | None = Field(
        default=None, description="Whether emergency shutdown is active"
    )
    cognitive_state: dict[str, Any] | None = Field(
        default=None, description="Aggregated cognitive state (safe subset)"
    )


class DetailedHealthStatus(BaseModel):
    """Detailed health status response.

    Attributes:
        status: Health status ("healthy" or "unhealthy").
        timestamp: Unix timestamp.
        uptime_seconds: Service uptime in seconds.
        system: System resource information.
        memory_state: Memory layer norms.
        phase: Current cognitive phase.
        statistics: Processing statistics.
    """

    status: str
    timestamp: float
    uptime_seconds: float
    system: dict[str, Any]
    memory_state: dict[str, Any] | None
    phase: str | None
    statistics: dict[str, Any] | None


# =============================================================================
# Error Schemas
# =============================================================================


class ErrorDetail(BaseModel):
    """Structured error detail.

    ## Contract Stability
    These are stable contract fields:
    - error_type: str - Type of error (e.g., "validation_error", "internal_error")
    - message: str - Human-readable error message
    - details: dict | None - Additional error context
    """

    error_type: str = Field(description="Type of error")
    message: str = Field(description="Human-readable error message")
    details: dict[str, Any] | None = Field(
        default=None, description="Additional error details"
    )
    debug_id: str | None = Field(
        default=None, description="Debug identifier for error tracking"
    )


class ErrorResponse(BaseModel):
    """Structured error response.

    ## Contract Stability
    The error field structure is stable contract:
    - error.error_type: str
    - error.message: str
    - error.details: dict | None
    """

    error: ErrorDetail


# =============================================================================
# SDK Response DTOs
# =============================================================================


class GenerateResponseDTO(BaseModel):
    """Data Transfer Object matching GenerateResponse for SDK clients.

    This DTO provides a 1:1 mapping with the API's GenerateResponse schema
    for use in typed SDK clients.

    ## Contract Stability
    Same stability guarantees as GenerateResponse:
    - response, phase, accepted are stable
    - Other fields may change in minor versions
    """

    response: str = Field(description="Generated response text")
    phase: str = Field(description="Current cognitive phase")
    accepted: bool = Field(description="Whether the request was accepted")
    metrics: dict[str, Any] | None = Field(default=None)
    safety_flags: dict[str, Any] | None = Field(default=None)
    memory_stats: dict[str, Any] | None = Field(default=None)
    moral_score: float | None = Field(default=None)
    aphasia_flags: dict[str, Any] | None = Field(default=None)
    emergency_shutdown: bool | None = Field(default=None)
    latency_ms: float | None = Field(default=None)
    cognitive_state: dict[str, Any] | None = Field(default=None)

    @classmethod
    def from_api_response(cls, data: dict[str, Any]) -> GenerateResponseDTO:
        """Create DTO from API response dictionary."""
        return cls(
            response=data.get("response", ""),
            phase=data.get("phase", "unknown"),
            accepted=data.get("accepted", False),
            metrics=data.get("metrics"),
            safety_flags=data.get("safety_flags"),
            memory_stats=data.get("memory_stats"),
            moral_score=data.get("moral_score"),
            aphasia_flags=data.get("aphasia_flags"),
            emergency_shutdown=data.get("emergency_shutdown"),
            latency_ms=data.get("latency_ms"),
            cognitive_state=data.get("cognitive_state"),
        )
