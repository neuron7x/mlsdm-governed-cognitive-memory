"""
SDK Exceptions for MLSDM HTTP Client.

This module provides typed exceptions for the MLSDM HTTP SDK client,
allowing for precise error handling and recovery strategies.
"""

from typing import Any


class MLSDMError(Exception):
    """Base exception for all MLSDM SDK errors.

    Attributes:
        message: Human-readable error message
        debug_id: Optional debug/correlation ID from the API
    """

    def __init__(
        self,
        message: str,
        debug_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.debug_id = debug_id

    def __str__(self) -> str:
        if self.debug_id:
            return f"{self.message} (debug_id={self.debug_id})"
        return self.message


class MLSDMClientError(MLSDMError):
    """Exception for 4xx client errors from the MLSDM API.

    This indicates a problem with the request (invalid input, authentication, etc.).

    Attributes:
        message: Human-readable error message
        status_code: HTTP status code (4xx)
        error_type: Error type from API (e.g., "validation_error", "rate_limit_exceeded")
        details: Additional error details from API
        debug_id: Optional debug/correlation ID from the API
    """

    def __init__(
        self,
        message: str,
        status_code: int,
        error_type: str | None = None,
        details: dict[str, Any] | None = None,
        debug_id: str | None = None,
    ) -> None:
        super().__init__(message, debug_id)
        self.status_code = status_code
        self.error_type = error_type
        self.details = details

    def __str__(self) -> str:
        base = f"[{self.status_code}] {self.message}"
        if self.error_type:
            base = f"[{self.status_code}] {self.error_type}: {self.message}"
        if self.debug_id:
            base += f" (debug_id={self.debug_id})"
        return base


class MLSDMValidationError(MLSDMClientError):
    """Exception for validation errors (HTTP 422).

    This indicates the request body failed Pydantic validation.

    Attributes:
        validation_errors: List of validation error details from FastAPI/Pydantic
    """

    def __init__(
        self,
        message: str,
        validation_errors: list[dict[str, Any]],
        debug_id: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=422,
            error_type="validation_error",
            details={"validation_errors": validation_errors},
            debug_id=debug_id,
        )
        self.validation_errors = validation_errors


class MLSDMRateLimitError(MLSDMClientError):
    """Exception for rate limit exceeded (HTTP 429).

    This indicates the client is sending too many requests.

    Attributes:
        retry_after: Optional number of seconds to wait before retrying
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded. Maximum 5 requests per second.",
        retry_after: float | None = None,
        debug_id: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=429,
            error_type="rate_limit_exceeded",
            debug_id=debug_id,
        )
        self.retry_after = retry_after


class MLSDMAuthenticationError(MLSDMClientError):
    """Exception for authentication errors (HTTP 401).

    This indicates missing or invalid authentication credentials.
    """

    def __init__(
        self,
        message: str = "Authentication required or invalid credentials",
        debug_id: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=401,
            error_type="unauthorized",
            debug_id=debug_id,
        )


class MLSDMServerError(MLSDMError):
    """Exception for 5xx server errors from the MLSDM API.

    This indicates a problem on the server side.

    Attributes:
        message: Human-readable error message
        status_code: HTTP status code (5xx)
        error_type: Error type from API (e.g., "internal_error")
        debug_id: Optional debug/correlation ID from the API
    """

    def __init__(
        self,
        message: str,
        status_code: int,
        error_type: str | None = None,
        debug_id: str | None = None,
    ) -> None:
        super().__init__(message, debug_id)
        self.status_code = status_code
        self.error_type = error_type

    def __str__(self) -> str:
        base = f"[{self.status_code}] {self.message}"
        if self.error_type:
            base = f"[{self.status_code}] {self.error_type}: {self.message}"
        if self.debug_id:
            base += f" (debug_id={self.debug_id})"
        return base


class MLSDMTimeoutError(MLSDMError):
    """Exception for request timeout.

    This indicates the request took too long to complete.

    Attributes:
        message: Human-readable error message
        timeout_seconds: Configured timeout value in seconds
        debug_id: Optional debug/correlation ID
    """

    def __init__(
        self,
        message: str = "Request timed out",
        timeout_seconds: float | None = None,
        debug_id: str | None = None,
    ) -> None:
        super().__init__(message, debug_id)
        self.timeout_seconds = timeout_seconds

    def __str__(self) -> str:
        base = self.message
        if self.timeout_seconds:
            base += f" (timeout={self.timeout_seconds}s)"
        if self.debug_id:
            base += f" (debug_id={self.debug_id})"
        return base


class MLSDMConnectionError(MLSDMError):
    """Exception for connection errors.

    This indicates the client could not connect to the API server.

    Attributes:
        message: Human-readable error message
        url: URL that could not be reached
        original_error: Original exception that caused this error
    """

    def __init__(
        self,
        message: str = "Could not connect to MLSDM API",
        url: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.url = url
        self.original_error = original_error

    def __str__(self) -> str:
        base = self.message
        if self.url:
            base += f" (url={self.url})"
        return base
