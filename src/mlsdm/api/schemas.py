"""
API Schema Definitions for MLSDM.

This module provides centralized Pydantic models for all API request/response schemas.
These schemas define the stable API contract and should NOT be modified without a major
version bump.

**Contract Stability Guarantee:**
The following fields are considered part of the stable API contract:
- GenerateRequest: prompt, max_tokens, moral_value
- GenerateResponse: response, phase, accepted
- HealthStatus: status, timestamp
- ReadinessStatus: ready, status, timestamp, checks
- ErrorResponse: error (with error_type, message, details)

Modifying these fields' names, types, or removing them requires a major version change.
Adding new optional fields is allowed without a major version bump.
"""

from typing import Any

from pydantic import BaseModel, Field

# ============================================================================
# Health Schemas
# ============================================================================


class SimpleHealthStatus(BaseModel):
    """Simple health status response for basic health check.

    Contract Fields (stable):
        - status: Health status string (e.g., "healthy")
    """

    status: str = Field(description="Health status")


class HealthStatus(BaseModel):
    """Basic health status response with timestamp.

    Contract Fields (stable):
        - status: Health status string (e.g., "alive")
        - timestamp: Unix timestamp of the health check
    """

    status: str = Field(description="Health status")
    timestamp: float = Field(description="Unix timestamp of the health check")


class ReadinessStatus(BaseModel):
    """Readiness status response for Kubernetes readiness probe.

    Contract Fields (stable):
        - ready: Boolean indicating if service is ready to accept traffic
        - status: Status string ("ready" or "not_ready")
        - timestamp: Unix timestamp of the readiness check
        - checks: Dictionary of individual check results

    Optional Extension Fields:
        - cognitive_state: Aggregated cognitive state information (optional)
    """

    ready: bool = Field(description="Whether the service is ready to accept traffic")
    status: str = Field(description="Readiness status ('ready' or 'not_ready')")
    timestamp: float = Field(description="Unix timestamp")
    checks: dict[str, bool] = Field(description="Individual check results")
    cognitive_state: dict[str, Any] | None = Field(
        default=None,
        description="Aggregated cognitive state from core (optional, added in v1.2)"
    )


class DetailedHealthStatus(BaseModel):
    """Detailed health status response with comprehensive system information.

    Contract Fields (stable):
        - status: Health status string ("healthy" or "unhealthy")
        - timestamp: Unix timestamp
        - uptime_seconds: Service uptime in seconds
        - system: System resource information

    Optional Extension Fields:
        - memory_state: Memory layer norms (optional)
        - phase: Current cognitive phase (optional)
        - statistics: Processing statistics (optional)
    """

    status: str = Field(description="Health status ('healthy' or 'unhealthy')")
    timestamp: float = Field(description="Unix timestamp")
    uptime_seconds: float = Field(description="Service uptime in seconds")
    system: dict[str, Any] = Field(description="System resource information")
    memory_state: dict[str, Any] | None = Field(
        default=None, description="Memory layer norms"
    )
    phase: str | None = Field(default=None, description="Current cognitive phase")
    statistics: dict[str, Any] | None = Field(
        default=None, description="Processing statistics"
    )


# ============================================================================
# Error Schemas
# ============================================================================


class ErrorDetail(BaseModel):
    """Structured error detail.

    Contract Fields (stable):
        - error_type: Type of error (e.g., "validation_error", "rate_limit_exceeded")
        - message: Human-readable error message

    Optional Extension Fields:
        - details: Additional error details (optional)
        - debug_id: Unique identifier for debugging (optional, added in v1.2)
    """

    error_type: str = Field(description="Type of error")
    message: str = Field(description="Human-readable error message")
    details: dict[str, Any] | None = Field(
        default=None, description="Additional error details"
    )
    debug_id: str | None = Field(
        default=None,
        description="Unique identifier for debugging/correlation (optional)"
    )


class ErrorResponse(BaseModel):
    """Structured error response.

    Contract Fields (stable):
        - error: ErrorDetail object with error information

    All API errors should follow this format for 4xx and 5xx responses.
    """

    error: ErrorDetail = Field(description="Error details")


# ============================================================================
# Generation Request/Response Schemas
# ============================================================================


class GenerateRequest(BaseModel):
    """Request model for generate endpoint.

    Contract Fields (stable):
        - prompt: Input text prompt (required, min 1 character)

    Optional Extension Fields:
        - max_tokens: Maximum number of tokens to generate (1-4096)
        - moral_value: Moral threshold value (0.0-1.0)
    """

    prompt: str = Field(..., min_length=1, description="Input text prompt to process")
    max_tokens: int | None = Field(
        None, ge=1, le=4096, description="Maximum number of tokens to generate"
    )
    moral_value: float | None = Field(
        None, ge=0.0, le=1.0, description="Moral threshold value"
    )


class GenerateResponse(BaseModel):
    """Response model for generate endpoint.

    Contract Fields (stable):
        - response: Generated response text
        - phase: Current cognitive phase ("wake", "sleep", or "unknown")
        - accepted: Whether the request was morally accepted

    Optional Extension Fields:
        - moral_score: Computed moral score (optional, added in v1.2)
        - aphasia_flags: Aphasia detection flags (optional, added in v1.2)
        - emergency_shutdown: Emergency shutdown status (optional, added in v1.2)
        - latency_ms: Request latency in milliseconds (optional)
        - cognitive_state: Aggregated cognitive state (optional, added in v1.2)
        - metrics: Performance timing metrics (optional)
        - safety_flags: Safety validation results (optional)
        - memory_stats: Memory state statistics (optional)
    """

    response: str = Field(description="Generated response text")
    phase: str = Field(description="Current cognitive phase")
    accepted: bool = Field(description="Whether the request was accepted")
    moral_score: float | None = Field(
        default=None, description="Computed moral score (0.0-1.0)"
    )
    aphasia_flags: dict[str, bool] | None = Field(
        default=None, description="Aphasia detection flags"
    )
    emergency_shutdown: bool | None = Field(
        default=None, description="Whether system is in emergency shutdown state"
    )
    latency_ms: float | None = Field(
        default=None, description="Request latency in milliseconds"
    )
    cognitive_state: dict[str, Any] | None = Field(
        default=None, description="Aggregated cognitive state information"
    )
    metrics: dict[str, Any] | None = Field(
        default=None, description="Performance timing metrics"
    )
    safety_flags: dict[str, Any] | None = Field(
        default=None, description="Safety validation results"
    )
    memory_stats: dict[str, Any] | None = Field(
        default=None, description="Memory state statistics"
    )


# ============================================================================
# Infer Request/Response Schemas (Extended API)
# ============================================================================


class InferRequest(BaseModel):
    """Request model for infer endpoint with extended governance options.

    Contract Fields (stable):
        - prompt: Input text prompt (required, min 1 character)

    Optional Extension Fields:
        - moral_value: Moral threshold value (default: 0.5)
        - max_tokens: Maximum number of tokens to generate (1-4096)
        - secure_mode: Enable enhanced security filtering
        - aphasia_mode: Enable aphasia detection and repair
        - rag_enabled: Enable RAG-based context retrieval
        - context_top_k: Number of context items for RAG (1-100)
        - user_intent: User intent category
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
    """Response model for infer endpoint with detailed metadata.

    Contract Fields (stable):
        - response: Generated response text
        - accepted: Whether the request was accepted
        - phase: Current cognitive phase

    Optional Extension Fields:
        - moral_metadata: Moral filtering metadata
        - aphasia_metadata: Aphasia detection/repair metadata
        - rag_metadata: RAG retrieval metadata
        - timing: Performance timing in milliseconds
        - governance: Full governance state information
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


# ============================================================================
# State/Event Schemas
# ============================================================================


class EventInput(BaseModel):
    """Request model for event processing endpoint.

    Contract Fields (stable):
        - event_vector: Event embedding vector
        - moral_value: Moral value for filtering
    """

    event_vector: list[float] = Field(description="Event embedding vector")
    moral_value: float = Field(description="Moral value for filtering")


class StateResponse(BaseModel):
    """Response model for state queries.

    Contract Fields (stable):
        - L1_norm: L1 memory layer norm
        - L2_norm: L2 memory layer norm
        - L3_norm: L3 memory layer norm
        - current_phase: Current cognitive phase
        - latent_events_count: Count of latent events
        - accepted_events_count: Count of accepted events
        - total_events_processed: Total events processed
        - moral_filter_threshold: Current moral threshold
    """

    L1_norm: float = Field(description="L1 memory layer norm")
    L2_norm: float = Field(description="L2 memory layer norm")
    L3_norm: float = Field(description="L3 memory layer norm")
    current_phase: str = Field(description="Current cognitive phase")
    latent_events_count: int = Field(description="Count of latent events")
    accepted_events_count: int = Field(description="Count of accepted events")
    total_events_processed: int = Field(description="Total events processed")
    moral_filter_threshold: float = Field(description="Current moral threshold")


# ============================================================================
# SDK DTO Classes (for typed SDK responses)
# ============================================================================


class GenerateResponseDTO:
    """Data Transfer Object for SDK generate responses.

    This class provides a typed interface for SDK clients, matching the
    GenerateResponse schema 1:1 for type safety and IDE support.

    Attributes:
        response: Generated response text
        phase: Current cognitive phase
        accepted: Whether the request was accepted
        moral_score: Computed moral score (optional)
        aphasia_flags: Aphasia detection flags (optional)
        emergency_shutdown: Emergency shutdown status (optional)
        latency_ms: Request latency in milliseconds (optional)
        cognitive_state: Aggregated cognitive state (optional)
        metrics: Performance timing metrics (optional)
        safety_flags: Safety validation results (optional)
        memory_stats: Memory state statistics (optional)
    """

    __slots__ = (
        "response",
        "phase",
        "accepted",
        "moral_score",
        "aphasia_flags",
        "emergency_shutdown",
        "latency_ms",
        "cognitive_state",
        "metrics",
        "safety_flags",
        "memory_stats",
    )

    def __init__(
        self,
        response: str,
        phase: str,
        accepted: bool,
        moral_score: float | None = None,
        aphasia_flags: dict[str, bool] | None = None,
        emergency_shutdown: bool | None = None,
        latency_ms: float | None = None,
        cognitive_state: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        safety_flags: dict[str, Any] | None = None,
        memory_stats: dict[str, Any] | None = None,
    ) -> None:
        self.response = response
        self.phase = phase
        self.accepted = accepted
        self.moral_score = moral_score
        self.aphasia_flags = aphasia_flags
        self.emergency_shutdown = emergency_shutdown
        self.latency_ms = latency_ms
        self.cognitive_state = cognitive_state
        self.metrics = metrics
        self.safety_flags = safety_flags
        self.memory_stats = memory_stats

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerateResponseDTO":
        """Create a GenerateResponseDTO from a dictionary response.

        Args:
            data: Dictionary containing response data from API

        Returns:
            GenerateResponseDTO instance

        Raises:
            KeyError: If required fields are missing
            TypeError: If field types are incorrect
        """
        return cls(
            response=str(data["response"]),
            phase=str(data["phase"]),
            accepted=bool(data["accepted"]),
            moral_score=data.get("moral_score"),
            aphasia_flags=data.get("aphasia_flags"),
            emergency_shutdown=data.get("emergency_shutdown"),
            latency_ms=data.get("latency_ms"),
            cognitive_state=data.get("cognitive_state"),
            metrics=data.get("metrics"),
            safety_flags=data.get("safety_flags"),
            memory_stats=data.get("memory_stats"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all fields
        """
        return {
            "response": self.response,
            "phase": self.phase,
            "accepted": self.accepted,
            "moral_score": self.moral_score,
            "aphasia_flags": self.aphasia_flags,
            "emergency_shutdown": self.emergency_shutdown,
            "latency_ms": self.latency_ms,
            "cognitive_state": self.cognitive_state,
            "metrics": self.metrics,
            "safety_flags": self.safety_flags,
            "memory_stats": self.memory_stats,
        }

    def __repr__(self) -> str:
        return (
            f"GenerateResponseDTO(response={self.response[:50]!r}..., "
            f"phase={self.phase!r}, accepted={self.accepted})"
        )
