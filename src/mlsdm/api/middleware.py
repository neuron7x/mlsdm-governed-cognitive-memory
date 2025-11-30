"""Middleware components for production-ready API.

Provides:
- Request ID tracking and correlation
- Security headers
- Bulkhead pattern for request isolation (REL-002)
- Request timeout handling (REL-004)
- Queue depth metrics
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bulkhead Pattern (REL-002)
# ---------------------------------------------------------------------------


@dataclass
class BulkheadMetrics:
    """Metrics for bulkhead monitoring.

    Attributes:
        total_requests: Total requests received
        accepted_requests: Requests that acquired a slot
        rejected_requests: Requests rejected due to capacity
        current_active: Currently active requests
        max_queue_depth: Maximum observed queue depth
    """

    total_requests: int = 0
    accepted_requests: int = 0
    rejected_requests: int = 0
    current_active: int = 0
    max_queue_depth: int = 0

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "accepted_requests": self.accepted_requests,
            "rejected_requests": self.rejected_requests,
            "current_active": self.current_active,
            "max_queue_depth": self.max_queue_depth,
        }


class BulkheadSemaphore:
    """Semaphore-based bulkhead for request isolation.

    Provides concurrency limiting with queue depth tracking and metrics.

    Example:
        >>> bulkhead = BulkheadSemaphore(max_concurrent=10)
        >>> async with bulkhead.acquire():
        ...     await handle_request()
    """

    def __init__(
        self,
        max_concurrent: int = 100,
        queue_timeout: float = 5.0,
    ) -> None:
        """Initialize bulkhead.

        Args:
            max_concurrent: Maximum concurrent requests
            queue_timeout: Timeout for waiting to acquire slot (seconds)
        """
        self._max_concurrent = max_concurrent
        self._queue_timeout = queue_timeout
        self._semaphore: asyncio.Semaphore | None = None
        self._metrics = BulkheadMetrics()
        self._lock = Lock()
        self._waiting_count = 0

    def _get_semaphore(self) -> asyncio.Semaphore:
        """Get or create the semaphore lazily for the current event loop.

        Returns:
            asyncio.Semaphore bound to the current event loop
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._semaphore

    @property
    def metrics(self) -> BulkheadMetrics:
        """Get current metrics."""
        with self._lock:
            return BulkheadMetrics(
                total_requests=self._metrics.total_requests,
                accepted_requests=self._metrics.accepted_requests,
                rejected_requests=self._metrics.rejected_requests,
                current_active=self._metrics.current_active,
                max_queue_depth=self._metrics.max_queue_depth,
            )

    @asynccontextmanager
    async def acquire(self) -> Any:
        """Acquire a slot from the bulkhead.

        Raises:
            asyncio.TimeoutError: If slot cannot be acquired within timeout

        Yields:
            None when slot is acquired
        """
        semaphore = self._get_semaphore()

        with self._lock:
            self._metrics.total_requests += 1
            self._waiting_count += 1
            queue_depth = self._waiting_count
            if queue_depth > self._metrics.max_queue_depth:
                self._metrics.max_queue_depth = queue_depth

        try:
            # Try to acquire with timeout
            await asyncio.wait_for(
                semaphore.acquire(),
                timeout=self._queue_timeout,
            )

            with self._lock:
                self._waiting_count -= 1
                self._metrics.accepted_requests += 1
                self._metrics.current_active += 1

            try:
                yield
            finally:
                semaphore.release()
                with self._lock:
                    self._metrics.current_active -= 1

        except asyncio.TimeoutError:
            with self._lock:
                self._waiting_count -= 1
                self._metrics.rejected_requests += 1
            raise

    @property
    def available(self) -> int:
        """Get number of available slots."""
        return self._max_concurrent - self._metrics.current_active

    @property
    def queue_depth(self) -> int:
        """Get current queue depth."""
        with self._lock:
            return self._waiting_count


class BulkheadMiddleware(BaseHTTPMiddleware):
    """Middleware implementing the bulkhead pattern for request isolation.

    Limits the number of concurrent requests to prevent resource exhaustion
    and isolate failures. Implements REL-002 from PROD_GAPS.md.

    Example:
        >>> app.add_middleware(
        ...     BulkheadMiddleware,
        ...     max_concurrent=100,
        ...     queue_timeout=5.0
        ... )
    """

    def __init__(
        self,
        app: Any,
        max_concurrent: int | None = None,
        queue_timeout: float | None = None,
    ) -> None:
        """Initialize bulkhead middleware.

        Args:
            app: FastAPI/Starlette application
            max_concurrent: Maximum concurrent requests (default: 100, env: MLSDM_MAX_CONCURRENT)
            queue_timeout: Queue timeout in seconds (default: 5.0, env: MLSDM_QUEUE_TIMEOUT)
        """
        super().__init__(app)
        self._max_concurrent = max_concurrent or int(os.getenv("MLSDM_MAX_CONCURRENT", "100"))
        self._queue_timeout = queue_timeout or float(os.getenv("MLSDM_QUEUE_TIMEOUT", "5.0"))
        self._bulkhead = BulkheadSemaphore(
            max_concurrent=self._max_concurrent,
            queue_timeout=self._queue_timeout,
        )

    @property
    def metrics(self) -> BulkheadMetrics:
        """Get bulkhead metrics."""
        return self._bulkhead.metrics

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Process request with bulkhead isolation."""
        try:
            async with self._bulkhead.acquire():
                return await call_next(request)
        except asyncio.TimeoutError:
            logger.warning(
                "Request rejected by bulkhead",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "queue_depth": self._bulkhead.queue_depth,
                    "active_requests": self._bulkhead.metrics.current_active,
                },
            )
            return Response(
                content='{"error": {"error_code": "E903", "message": "Service temporarily unavailable - too many concurrent requests"}}',
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                media_type="application/json",
                headers={"Retry-After": str(int(self._queue_timeout))},
            )


# ---------------------------------------------------------------------------
# Request Timeout (REL-004)
# ---------------------------------------------------------------------------


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce request-level timeouts.

    Implements REL-004 from PROD_GAPS.md. Returns 504 Gateway Timeout
    if request processing exceeds configured timeout.

    Example:
        >>> app.add_middleware(TimeoutMiddleware, timeout=30.0)
    """

    def __init__(
        self,
        app: Any,
        timeout: float | None = None,
        exclude_paths: list[str] | None = None,
    ) -> None:
        """Initialize timeout middleware.

        Args:
            app: FastAPI/Starlette application
            timeout: Request timeout in seconds (default: 30.0, env: MLSDM_REQUEST_TIMEOUT)
            exclude_paths: Paths to exclude from timeout (e.g., health checks)
        """
        super().__init__(app)
        self._timeout = timeout or float(os.getenv("MLSDM_REQUEST_TIMEOUT", "30.0"))
        self._exclude_paths = set(exclude_paths or ["/health", "/health/live", "/health/ready"])

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Process request with timeout enforcement."""
        # Skip timeout for excluded paths
        if request.url.path in self._exclude_paths:
            return await call_next(request)

        start_time = time.time()

        try:
            response = await asyncio.wait_for(
                call_next(request),
                timeout=self._timeout,
            )
            return response

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(
                "Request timeout",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "timeout": self._timeout,
                    "elapsed": elapsed,
                    "request_id": getattr(request.state, "request_id", "unknown"),
                },
            )
            return Response(
                content='{"error": {"error_code": "E902", "message": "Request timed out"}}',
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                media_type="application/json",
                headers={"X-Request-Timeout": str(self._timeout)},
            )


# ---------------------------------------------------------------------------
# Existing Middleware (unchanged)
# ---------------------------------------------------------------------------


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID to all requests.

    Adds a unique request ID to each request for correlation across
    distributed systems. The request ID is:
    - Generated if not provided in X-Request-ID header
    - Added to response headers
    - Made available to all downstream handlers via request.state
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Process request and add request ID."""
        # Get or generate request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Store in request state for access by handlers
        request.state.request_id = request_id

        # Log request start
        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else "unknown",
            },
        )

        # Process request and time it
        start_time = time.time()
        try:
            response = await call_next(request)
        except Exception as e:
            # Log error with request ID
            logger.error(
                f"Request failed: {str(e)}",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                },
                exc_info=True,
            )
            raise
        finally:
            # Log request completion
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "duration_ms": duration_ms,
                    "status_code": response.status_code if "response" in locals() else None,
                },
            )

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        # Add timing header
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"

        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to all responses.

    Adds standard security headers recommended by OWASP:
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: DENY
    - X-XSS-Protection: 1; mode=block
    - Strict-Transport-Security: max-age=31536000; includeSubDomains
    - Content-Security-Policy: default-src 'self'
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Process request and add security headers."""
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Only add HSTS if using HTTPS
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # CSP header - restrictive default
        response.headers["Content-Security-Policy"] = "default-src 'self'"

        return response


def get_request_id(request: Request) -> str:
    """Get request ID from request state.

    Args:
        request: FastAPI request object

    Returns:
        Request ID string, or "unknown" if not set
    """
    return getattr(request.state, "request_id", "unknown")
