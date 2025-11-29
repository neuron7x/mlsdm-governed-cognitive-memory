"""Health check endpoints for MLSDM API.

Provides liveness, readiness, and detailed health status endpoints
with appropriate HTTP status codes based on system state.

API Contract Stability Policy:
-----------------------------
The following fields are part of the stable API contract:

/health (SimpleHealthStatus):
    - status: str (always "healthy" when service is responsive)

/health/liveness (HealthStatus):
    - status: str (always "alive")
    - timestamp: float

/health/readiness (ReadinessStatus):
    - ready: bool
    - status: str ("ready" or "not_ready")
    - timestamp: float
    - checks: dict[str, bool]

/health/detailed (DetailedHealthStatus):
    - status: str ("healthy" or "unhealthy")
    - timestamp: float
    - uptime_seconds: float
    - system: dict[str, Any]

These fields will not be removed or renamed without a major version bump.
"""

import logging
import time
from typing import Any

import numpy as np
import psutil
from fastapi import APIRouter, Response, status
from fastapi.responses import PlainTextResponse

from mlsdm.api.schemas import (
    DetailedHealthStatus,
    HealthStatus,
    ReadinessStatus,
    SimpleHealthStatus,
)
from mlsdm.observability.metrics import get_metrics_exporter

logger = logging.getLogger(__name__)

# Health check router
router = APIRouter(prefix="/health", tags=["health"])


# Track when the service started
_start_time = time.time()

# Global manager reference (to be set by the application)
_memory_manager: Any | None = None


def set_memory_manager(manager: Any) -> None:
    """Set the global memory manager reference for health checks.

    Args:
        manager: MemoryManager instance
    """
    global _memory_manager
    _memory_manager = manager


def get_memory_manager() -> Any | None:
    """Get the global memory manager reference.

    Returns:
        MemoryManager instance or None
    """
    return _memory_manager


@router.get("", response_model=SimpleHealthStatus)
async def health_check() -> SimpleHealthStatus:
    """Simple health check endpoint.

    Returns basic health status without detailed information.
    This is the primary endpoint for simple health checks.

    Returns:
        SimpleHealthStatus with status="healthy" if service is responsive.
    """
    return SimpleHealthStatus(status="healthy")


@router.get("/liveness", response_model=HealthStatus)
async def liveness() -> HealthStatus:
    """Liveness probe endpoint.

    Indicates whether the process is alive and running.
    Always returns 200 if the process is responsive.

    Returns:
        HealthStatus with 200 status code
    """
    return HealthStatus(
        status="alive",
        timestamp=time.time(),
    )


@router.get("/readiness", response_model=ReadinessStatus)
async def readiness(response: Response) -> ReadinessStatus:
    """Readiness probe endpoint.

    Indicates whether the system can accept traffic.
    Returns 200 if ready, 503 if not ready.

    Args:
        response: FastAPI response object to set status code

    Returns:
        ReadinessStatus with appropriate status code
    """
    checks: dict[str, bool] = {}
    all_ready = True

    # Check if memory manager is initialized
    manager = get_memory_manager()
    checks["memory_manager"] = manager is not None
    if not checks["memory_manager"]:
        all_ready = False

    # Check system resources
    try:
        memory = psutil.virtual_memory()
        # Consider not ready if memory usage > 95%
        checks["memory_available"] = memory.percent < 95.0
        if not checks["memory_available"]:
            all_ready = False
    except Exception as e:
        logger.warning(f"Failed to check memory availability: {e}")
        checks["memory_available"] = False
        all_ready = False

    # Check CPU
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        # Consider not ready if CPU usage > 98%
        checks["cpu_available"] = cpu_percent < 98.0
        if not checks["cpu_available"]:
            all_ready = False
    except Exception as e:
        logger.warning(f"Failed to check CPU availability: {e}")
        checks["cpu_available"] = False
        all_ready = False

    # Set response status code
    if all_ready:
        response.status_code = status.HTTP_200_OK
        status_str = "ready"
    else:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        status_str = "not_ready"

    return ReadinessStatus(
        ready=all_ready,
        status=status_str,
        timestamp=time.time(),
        checks=checks,
    )


@router.get("/detailed", response_model=DetailedHealthStatus)
async def detailed_health(response: Response) -> DetailedHealthStatus:
    """Detailed health status endpoint.

    Provides comprehensive system status including:
    - Memory state (L1, L2, L3 norms)
    - Current cognitive phase
    - System statistics
    - Resource usage

    Returns 200 if healthy, 503 if unhealthy.

    Args:
        response: FastAPI response object to set status code

    Returns:
        DetailedHealthStatus with appropriate status code
    """
    current_time = time.time()
    uptime = current_time - _start_time

    # Collect system information
    system_info: dict[str, Any] = {}
    try:
        memory = psutil.virtual_memory()
        system_info["memory_percent"] = memory.percent
        system_info["memory_available_mb"] = memory.available / (1024 * 1024)
        system_info["memory_total_mb"] = memory.total / (1024 * 1024)
    except Exception as e:
        logger.error(f"Failed to get memory info: {e}")
        system_info["memory_error"] = str(e)

    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        system_info["cpu_percent"] = cpu_percent
        system_info["cpu_count"] = psutil.cpu_count()
    except Exception as e:
        logger.error(f"Failed to get CPU info: {e}")
        system_info["cpu_error"] = str(e)

    try:
        disk = psutil.disk_usage("/")
        system_info["disk_percent"] = disk.percent
    except Exception as e:
        logger.error(f"Failed to get disk info: {e}")
        system_info["disk_error"] = str(e)

    # Get memory manager state if available
    memory_state: dict[str, Any] | None = None
    phase: str | None = None
    statistics: dict[str, Any] | None = None
    is_healthy = True

    manager = get_memory_manager()
    if manager is not None:
        try:
            # Get memory layer states
            l1, l2, l3 = manager.memory.get_state()
            memory_state = {
                "L1_norm": float(np.linalg.norm(l1)),
                "L2_norm": float(np.linalg.norm(l2)),
                "L3_norm": float(np.linalg.norm(l3)),
            }

            # Get current phase
            phase = manager.rhythm.get_current_phase()

            # Get statistics
            metrics = manager.metrics_collector.get_metrics()
            statistics = {
                "total_events_processed": int(metrics["total_events_processed"]),
                "accepted_events_count": int(metrics["accepted_events_count"]),
                "latent_events_count": int(metrics["latent_events_count"]),
                "moral_filter_threshold": float(manager.filter.threshold),
            }

            # Calculate average latency if available
            if metrics["latencies"]:
                statistics["avg_latency_seconds"] = float(
                    sum(metrics["latencies"]) / len(metrics["latencies"])
                )
                statistics["avg_latency_ms"] = statistics["avg_latency_seconds"] * 1000

        except Exception as e:
            logger.error(f"Failed to get manager state: {e}")
            is_healthy = False
            statistics = {"error": str(e)}
    else:
        is_healthy = False

    # Check overall health
    if system_info.get("memory_percent", 0) > 95:
        is_healthy = False
    if system_info.get("cpu_percent", 0) > 98:
        is_healthy = False

    # Set response status code
    if is_healthy:
        response.status_code = status.HTTP_200_OK
        health_status = "healthy"
    else:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        health_status = "unhealthy"

    return DetailedHealthStatus(
        status=health_status,
        timestamp=current_time,
        uptime_seconds=uptime,
        system=system_info,
        memory_state=memory_state,
        phase=phase,
        statistics=statistics,
    )


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    """Prometheus metrics endpoint.

    Exports metrics in Prometheus text format for scraping.

    Returns:
        Prometheus-formatted metrics as plain text
    """
    metrics_exporter = get_metrics_exporter()
    return metrics_exporter.get_metrics_text()
