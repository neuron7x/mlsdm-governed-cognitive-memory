"""Performance SLO (Service Level Objective) thresholds.

This module defines conservative SLO thresholds for CI/CD and production monitoring.
Values are derived from SLO_SPEC.md with CI-appropriate safety margins.

References:
    - SLO_SPEC.md: Service Level Objectives specification
    - OBSERVABILITY_SPEC.md: Observability and metrics schema
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class LatencySLO:
    """Latency SLO thresholds in milliseconds.
    
    Based on SLO_SPEC.md targets with conservative margins for CI stability.
    """
    
    # API endpoint latencies (HTTP layer)
    api_p50_ms: float = 50.0  # Target: 30ms, CI: 50ms for safety margin
    api_p95_ms: float = 150.0  # Target: 120ms, CI: 150ms for variability
    api_p99_ms: float = 250.0  # Stretch goal, not enforced in CI
    
    # Engine latencies (NeuroCognitiveEngine)
    engine_total_p50_ms: float = 100.0  # Includes all processing
    engine_total_p95_ms: float = 600.0  # Target: 500ms, CI: 600ms
    engine_preflight_p95_ms: float = 30.0  # Target: 20ms, CI: 30ms
    
    # Generation latency (with stub backend)
    generation_p95_ms: float = 50.0  # Stub backend should be fast


@dataclass(frozen=True)
class ErrorRateSLO:
    """Error rate SLO thresholds as percentages.
    
    Lower values indicate better quality. Based on SLO_SPEC.md error budget.
    """
    
    # Overall error rate (5xx errors, system failures)
    max_error_rate_percent: float = 1.0  # Target: 0.5%, CI: 1.0% for stability
    
    # Availability (inverse of error rate)
    min_availability_percent: float = 99.0  # Target: 99.9%, CI: 99.0%
    
    # Request rejection rate (not counted as errors, but monitored)
    expected_rejection_rate_percent_min: float = 0.0
    expected_rejection_rate_percent_max: float = 30.0


@dataclass(frozen=True)
class ThroughputSLO:
    """Throughput SLO thresholds.
    
    Defines expected request processing capacity.
    """
    
    # Minimum sustained throughput (requests per second)
    min_rps: float = 50.0  # Conservative for CI, production target: 1000+ RPS
    
    # Maximum queue depth before degradation
    max_queue_depth: int = 100
    
    # Concurrent request capacity
    min_concurrent_capacity: int = 10


@dataclass(frozen=True)
class LoadProfile:
    """Standard load test profile definition."""
    
    name: Literal["light", "moderate", "spike"]
    total_requests: int
    concurrency: int
    description: str


# Standard load profiles for testing
LOAD_PROFILES: dict[str, LoadProfile] = {
    "light": LoadProfile(
        name="light",
        total_requests=50,
        concurrency=5,
        description="Light load: 50 requests, 5 concurrent"
    ),
    "moderate": LoadProfile(
        name="moderate",
        total_requests=200,
        concurrency=10,
        description="Moderate load: 200 requests, 10 concurrent"
    ),
    "spike": LoadProfile(
        name="spike",
        total_requests=100,
        concurrency=20,
        description="Spike load: 100 requests, 20 concurrent (tests circuit breaker)"
    ),
}


# Default SLO instances
DEFAULT_LATENCY_SLO = LatencySLO()
DEFAULT_ERROR_RATE_SLO = ErrorRateSLO()
DEFAULT_THROUGHPUT_SLO = ThroughputSLO()


def get_load_profile(name: Literal["light", "moderate", "spike"]) -> LoadProfile:
    """Get a standard load profile by name.
    
    Args:
        name: Profile name (light, moderate, or spike)
        
    Returns:
        LoadProfile configuration
        
    Raises:
        KeyError: If profile name is invalid
    """
    return LOAD_PROFILES[name]
