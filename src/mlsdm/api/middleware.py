"""Middleware components for production-ready API.

Provides request ID tracking, timing, and correlation capabilities.
"""

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


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
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains"
            )

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
