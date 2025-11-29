"""
SDK Exceptions for MLSDM API Client.

This module defines typed exceptions for the MLSDM SDK HTTP client.
These exceptions provide clear, typed error handling for HTTP-based
interactions with the MLSDM API.
"""

from __future__ import annotations


class MLSDMError(Exception):
    """Base exception for all MLSDM SDK errors.

    Attributes:
        message: Human-readable error message.
        debug_id: Optional debug identifier for error tracking.
    """

    def __init__(self, message: str, debug_id: str | None = None) -> None:
        """Initialize base MLSDM error.

        Args:
            message: Human-readable error message.
            debug_id: Optional debug identifier for error tracking.
        """
        self.message = message
        self.debug_id = debug_id
        super().__init__(message)

    def __str__(self) -> str:
        """Return string representation."""
        if self.debug_id:
            return f"{self.message} (debug_id: {self.debug_id})"
        return self.message


class MLSDMClientError(MLSDMError):
    """Exception for 4xx client errors from the MLSDM API.

    Raised when the API returns a 4xx status code indicating
    a client-side error (e.g., validation error, bad request).

    Attributes:
        message: Human-readable error message.
        status_code: HTTP status code (4xx range).
        error_type: Type of error from API response.
        details: Additional error details from API.
        debug_id: Optional debug identifier.
    """

    def __init__(
        self,
        message: str,
        status_code: int,
        error_type: str | None = None,
        details: dict | None = None,
        debug_id: str | None = None,
    ) -> None:
        """Initialize client error.

        Args:
            message: Human-readable error message.
            status_code: HTTP status code (4xx range).
            error_type: Type of error from API response.
            details: Additional error details from API.
            debug_id: Optional debug identifier.
        """
        super().__init__(message, debug_id)
        self.status_code = status_code
        self.error_type = error_type
        self.details = details

    def __str__(self) -> str:
        """Return string representation."""
        base = f"[{self.status_code}] {self.message}"
        if self.error_type:
            base = f"[{self.status_code}:{self.error_type}] {self.message}"
        if self.debug_id:
            base += f" (debug_id: {self.debug_id})"
        return base


class MLSDMServerError(MLSDMError):
    """Exception for 5xx server errors from the MLSDM API.

    Raised when the API returns a 5xx status code indicating
    a server-side error (e.g., internal error, service unavailable).

    Attributes:
        message: Human-readable error message.
        status_code: HTTP status code (5xx range).
        error_type: Type of error from API response.
        debug_id: Optional debug identifier.
    """

    def __init__(
        self,
        message: str,
        status_code: int,
        error_type: str | None = None,
        debug_id: str | None = None,
    ) -> None:
        """Initialize server error.

        Args:
            message: Human-readable error message.
            status_code: HTTP status code (5xx range).
            error_type: Type of error from API response.
            debug_id: Optional debug identifier.
        """
        super().__init__(message, debug_id)
        self.status_code = status_code
        self.error_type = error_type

    def __str__(self) -> str:
        """Return string representation."""
        base = f"[{self.status_code}] {self.message}"
        if self.error_type:
            base = f"[{self.status_code}:{self.error_type}] {self.message}"
        if self.debug_id:
            base += f" (debug_id: {self.debug_id})"
        return base


class MLSDMTimeoutError(MLSDMError):
    """Exception for timeout errors when calling the MLSDM API.

    Raised when a request to the API times out.

    Attributes:
        message: Human-readable error message.
        timeout_seconds: The timeout value that was exceeded.
    """

    def __init__(
        self,
        message: str,
        timeout_seconds: float,
        debug_id: str | None = None,
    ) -> None:
        """Initialize timeout error.

        Args:
            message: Human-readable error message.
            timeout_seconds: The timeout value that was exceeded.
            debug_id: Optional debug identifier.
        """
        super().__init__(message, debug_id)
        self.timeout_seconds = timeout_seconds

    def __str__(self) -> str:
        """Return string representation."""
        base = f"{self.message} (timeout: {self.timeout_seconds}s)"
        if self.debug_id:
            base += f" (debug_id: {self.debug_id})"
        return base


class MLSDMConnectionError(MLSDMError):
    """Exception for connection errors when calling the MLSDM API.

    Raised when unable to connect to the API server.

    Attributes:
        message: Human-readable error message.
        url: The URL that failed to connect.
    """

    def __init__(
        self,
        message: str,
        url: str | None = None,
        debug_id: str | None = None,
    ) -> None:
        """Initialize connection error.

        Args:
            message: Human-readable error message.
            url: The URL that failed to connect.
            debug_id: Optional debug identifier.
        """
        super().__init__(message, debug_id)
        self.url = url

    def __str__(self) -> str:
        """Return string representation."""
        base = self.message
        if self.url:
            base += f" (url: {self.url})"
        if self.debug_id:
            base += f" (debug_id: {self.debug_id})"
        return base
