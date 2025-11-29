"""
MLSDM SDK Exceptions.

This module defines SDK-specific exceptions for the MLSDM Python SDK.
These exceptions provide clear, typed error handling for SDK consumers.

Exception Hierarchy:
    MLSDMError (base)
    ├── MLSDMClientError (4xx-like errors)
    │   ├── MLSDMValidationError (invalid input)
    │   └── MLSDMConfigError (configuration errors)
    ├── MLSDMServerError (5xx-like errors)
    └── MLSDMTimeoutError (timeout errors)
"""

from __future__ import annotations


class MLSDMError(Exception):
    """Base exception for all MLSDM SDK errors.

    All SDK exceptions inherit from this class, allowing consumers to
    catch all SDK-related errors with a single except clause.

    Attributes:
        message: Human-readable error message.
        details: Optional dictionary with additional error context.
    """

    def __init__(self, message: str, details: dict | None = None) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message.
            details: Optional dictionary with additional error context.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __repr__(self) -> str:
        """Return a string representation of the exception."""
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details!r})"


class MLSDMClientError(MLSDMError):
    """Exception raised for client-side errors (4xx-like).

    This exception indicates that the error is likely due to invalid
    input, configuration, or other client-side issues.

    Examples:
        - Invalid prompt (empty, too long)
        - Invalid parameters (out of range values)
        - Missing required configuration
    """

    pass


class MLSDMValidationError(MLSDMClientError):
    """Exception raised for validation errors.

    This exception is raised when input validation fails, such as:
        - Empty or whitespace-only prompts
        - Parameters outside valid ranges
        - Invalid data types

    Attributes:
        field: The field that failed validation.
        value: The invalid value that was provided.
    """

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: object | None = None,
        details: dict | None = None,
    ) -> None:
        """Initialize the validation error.

        Args:
            message: Human-readable error message.
            field: The field that failed validation.
            value: The invalid value that was provided.
            details: Optional dictionary with additional error context.
        """
        super().__init__(message, details)
        self.field = field
        self.value = value


class MLSDMConfigError(MLSDMClientError):
    """Exception raised for configuration errors.

    This exception is raised when SDK configuration is invalid, such as:
        - Invalid backend name
        - Missing API keys
        - Invalid model configuration
    """

    pass


class MLSDMServerError(MLSDMError):
    """Exception raised for server-side errors (5xx-like).

    This exception indicates that the error occurred on the server side,
    such as:
        - Internal processing errors
        - Backend LLM failures
        - Memory/resource exhaustion

    Attributes:
        error_type: The type of server error (e.g., 'internal_error').
        retry_after: Optional hint for when to retry (in seconds).
    """

    def __init__(
        self,
        message: str,
        error_type: str | None = None,
        retry_after: float | None = None,
        details: dict | None = None,
    ) -> None:
        """Initialize the server error.

        Args:
            message: Human-readable error message.
            error_type: The type of server error.
            retry_after: Optional hint for when to retry (in seconds).
            details: Optional dictionary with additional error context.
        """
        super().__init__(message, details)
        self.error_type = error_type or "internal_error"
        self.retry_after = retry_after


class MLSDMTimeoutError(MLSDMError):
    """Exception raised when a request times out.

    This exception is raised when an operation exceeds the configured
    timeout, such as:
        - LLM generation taking too long
        - Backend service timeout
        - Network timeout

    Attributes:
        timeout_seconds: The timeout value that was exceeded.
        operation: The operation that timed out (e.g., 'generate').
    """

    def __init__(
        self,
        message: str,
        timeout_seconds: float | None = None,
        operation: str | None = None,
        details: dict | None = None,
    ) -> None:
        """Initialize the timeout error.

        Args:
            message: Human-readable error message.
            timeout_seconds: The timeout value that was exceeded.
            operation: The operation that timed out.
            details: Optional dictionary with additional error context.
        """
        super().__init__(message, details)
        self.timeout_seconds = timeout_seconds
        self.operation = operation or "unknown"


class MLSDMRateLimitError(MLSDMClientError):
    """Exception raised when rate limit is exceeded.

    This exception is raised when too many requests are made within
    a time window.

    Attributes:
        retry_after: Seconds to wait before retrying.
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: float | None = None,
        details: dict | None = None,
    ) -> None:
        """Initialize the rate limit error.

        Args:
            message: Human-readable error message.
            retry_after: Seconds to wait before retrying.
            details: Optional dictionary with additional error context.
        """
        super().__init__(message, details)
        self.retry_after = retry_after


__all__ = [
    "MLSDMError",
    "MLSDMClientError",
    "MLSDMValidationError",
    "MLSDMConfigError",
    "MLSDMServerError",
    "MLSDMTimeoutError",
    "MLSDMRateLimitError",
]
