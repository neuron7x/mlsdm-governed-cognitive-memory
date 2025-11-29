"""
MLSDM SDK Data Transfer Objects (DTOs).

This module defines typed response objects for the MLSDM Python SDK.
These DTOs provide type-safe access to response data with proper
attribute access and serialization support.

API Contract Stability:
----------------------
The following fields are part of the stable SDK contract:

GenerateResponseDTO (stable):
    - response: str
    - phase: str
    - accepted: bool

These fields will not be removed or renamed without a major version bump.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerateResponseDTO:
    """Data transfer object for generate response.

    This DTO provides typed, attribute-based access to generation results.
    It corresponds to the API's GenerateResponse schema.

    Contract Fields (stable, guaranteed across minor versions):
        - response: Generated text
        - phase: Current cognitive phase ('wake' or 'sleep')
        - accepted: Whether the request was accepted

    Optional Fields (may be None):
        - metrics: Performance and timing information
        - safety_flags: Safety-related validation results
        - memory_stats: Memory state statistics
        - moral_score: Moral evaluation score (if available)
        - aphasia_flags: Aphasia detection flags (if available)
        - emergency_shutdown: Emergency shutdown indicator
        - latency_ms: Total processing latency in milliseconds
        - cognitive_state: Aggregated cognitive state snapshot
        - error: Error information (if any)
        - rejected_at: Stage at which request was rejected (if any)

    Example:
        >>> result = client.generate("Hello, world!")
        >>> print(result.response)
        >>> print(f"Phase: {result.phase}, Accepted: {result.accepted}")
        >>> if result.latency_ms:
        ...     print(f"Latency: {result.latency_ms:.2f}ms")
    """

    # Core fields (always present) - STABLE CONTRACT
    response: str = ""
    phase: str = "unknown"
    accepted: bool = False

    # Optional metrics and diagnostics
    metrics: dict[str, Any] | None = None
    safety_flags: dict[str, Any] | None = None
    memory_stats: dict[str, Any] | None = None

    # Extended fields for cognitive state
    moral_score: float | None = None
    aphasia_flags: dict[str, Any] | None = None
    emergency_shutdown: bool | None = None
    latency_ms: float | None = None
    cognitive_state: dict[str, Any] | None = None

    # Error tracking
    error: dict[str, Any] | None = None
    rejected_at: str | None = None

    # Raw response data (for backward compatibility)
    _raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerateResponseDTO:
        """Create a GenerateResponseDTO from a dictionary.

        This method maps the engine's raw response dictionary to a typed DTO.

        Args:
            data: Raw response dictionary from the engine.

        Returns:
            A typed GenerateResponseDTO instance.
        """
        # Extract mlsdm state for phase and other info
        mlsdm_state = data.get("mlsdm", {})

        # Determine phase from mlsdm state
        phase = mlsdm_state.get("phase", "unknown")

        # Determine accepted status
        rejected_at = data.get("rejected_at")
        error_info = data.get("error")
        accepted = rejected_at is None and error_info is None and bool(data.get("response"))

        # Extract latency from timing
        timing = data.get("timing")
        latency_ms = timing.get("total") if timing else None

        # Build safety flags from validation steps
        safety_flags = None
        validation_steps = data.get("validation_steps", [])
        if validation_steps:
            safety_flags = {
                "validation_steps": validation_steps,
                "rejected_at": rejected_at,
            }

        # Build metrics from timing
        metrics = None
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

        return cls(
            response=data.get("response", ""),
            phase=phase,
            accepted=accepted,
            metrics=metrics,
            safety_flags=safety_flags,
            memory_stats=memory_stats,
            moral_score=mlsdm_state.get("moral_threshold"),
            aphasia_flags=None,  # Reserved for future use
            emergency_shutdown=None,  # Would come from cognitive state
            latency_ms=latency_ms,
            cognitive_state=mlsdm_state if mlsdm_state else None,
            error=error_info,
            rejected_at=rejected_at,
            _raw=data,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert the DTO to a dictionary.

        Returns:
            Dictionary representation of the response.
        """
        return {
            "response": self.response,
            "phase": self.phase,
            "accepted": self.accepted,
            "metrics": self.metrics,
            "safety_flags": self.safety_flags,
            "memory_stats": self.memory_stats,
            "moral_score": self.moral_score,
            "aphasia_flags": self.aphasia_flags,
            "emergency_shutdown": self.emergency_shutdown,
            "latency_ms": self.latency_ms,
            "cognitive_state": self.cognitive_state,
            "error": self.error,
            "rejected_at": self.rejected_at,
        }

    @property
    def raw(self) -> dict[str, Any]:
        """Access the raw response data.

        Returns:
            The original raw dictionary from the engine.
        """
        return self._raw

    @property
    def is_success(self) -> bool:
        """Check if the request was successful.

        Returns:
            True if the request was accepted and has a response.
        """
        return self.accepted and bool(self.response)

    @property
    def is_rejected(self) -> bool:
        """Check if the request was rejected.

        Returns:
            True if the request was rejected at some stage.
        """
        return self.rejected_at is not None

    @property
    def has_error(self) -> bool:
        """Check if there was an error.

        Returns:
            True if there was an error during processing.
        """
        return self.error is not None


__all__ = [
    "GenerateResponseDTO",
]
