"""
MLSDM HTTP Client: Typed HTTP client for MLSDM API.

This module provides a strongly-typed HTTP client for interacting with
the MLSDM API server. It handles HTTP requests, response parsing,
error handling, and timeout management.

Example:
    >>> client = MLSDMHttpClient(base_url="http://localhost:8000")
    >>> response = client.generate("Hello, world!")
    >>> print(response.response)
    >>> print(response.phase)
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from mlsdm.api.schemas import GenerateResponseDTO
from mlsdm.sdk.exceptions import (
    MLSDMClientError,
    MLSDMConnectionError,
    MLSDMServerError,
    MLSDMTimeoutError,
)

logger = logging.getLogger(__name__)

# Default timeout for HTTP requests (30 seconds)
DEFAULT_TIMEOUT_SECONDS: float = 30.0


class MLSDMHttpClient:
    """Typed HTTP client for MLSDM API.

    This client provides strongly-typed methods for interacting with
    the MLSDM API over HTTP. It handles:
    - Request serialization and response parsing
    - HTTP error handling with typed exceptions
    - Timeout management
    - Connection error handling

    Attributes:
        base_url: Base URL of the MLSDM API server.
        timeout: Default timeout for requests in seconds.

    Example:
        >>> client = MLSDMHttpClient(base_url="http://localhost:8000")
        >>> try:
        ...     response = client.generate("What is AI?")
        ...     print(f"Response: {response.response}")
        ...     print(f"Phase: {response.phase}")
        ... except MLSDMClientError as e:
        ...     print(f"Client error: {e}")
        ... except MLSDMServerError as e:
        ...     print(f"Server error: {e}")
        ... except MLSDMTimeoutError as e:
        ...     print(f"Timeout: {e}")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize the HTTP client.

        Args:
            base_url: Base URL of the MLSDM API server.
            timeout: Default timeout for requests in seconds.
        """
        # Remove trailing slash from base_url
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        # Set default headers
        self._session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    def generate(
        self,
        prompt: str,
        moral_value: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> GenerateResponseDTO:
        """Generate a response using the MLSDM API.

        Makes a POST request to /generate endpoint and returns a typed DTO.

        Args:
            prompt: Input text prompt to process.
            moral_value: Optional moral threshold value (0.0-1.0).
            max_tokens: Optional maximum tokens to generate (1-4096).
            **kwargs: Additional parameters to pass to the API.

        Returns:
            GenerateResponseDTO with the generated response and metadata.

        Raises:
            MLSDMClientError: For 4xx HTTP errors.
            MLSDMServerError: For 5xx HTTP errors.
            MLSDMTimeoutError: If the request times out.
            MLSDMConnectionError: If unable to connect to the server.
        """
        # Build request body
        body: dict[str, Any] = {"prompt": prompt}
        if moral_value is not None:
            body["moral_value"] = moral_value
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        body.update(kwargs)

        # Make request
        url = f"{self.base_url}/generate"
        response_data = self._post(url, body)

        # Parse into DTO
        return GenerateResponseDTO.from_api_response(response_data)

    def health(self) -> dict[str, Any]:
        """Check basic health status.

        Makes a GET request to /health endpoint.

        Returns:
            Health status dictionary.

        Raises:
            MLSDMConnectionError: If unable to connect to the server.
        """
        url = f"{self.base_url}/health"
        return self._get(url)

    def ready(self) -> dict[str, Any]:
        """Check readiness status.

        Makes a GET request to /ready endpoint.

        Returns:
            Readiness status dictionary.

        Raises:
            MLSDMServerError: If service is not ready (503).
            MLSDMConnectionError: If unable to connect to the server.
        """
        url = f"{self.base_url}/ready"
        return self._get(url)

    def readiness(self) -> dict[str, Any]:
        """Check readiness status (alias for ready()).

        Makes a GET request to /health/readiness endpoint.

        Returns:
            Readiness status dictionary.

        Raises:
            MLSDMServerError: If service is not ready (503).
            MLSDMConnectionError: If unable to connect to the server.
        """
        url = f"{self.base_url}/health/readiness"
        return self._get(url)

    def _get(self, url: str) -> dict[str, Any]:
        """Make a GET request.

        Args:
            url: Full URL to request.

        Returns:
            Response JSON as dictionary.

        Raises:
            MLSDMClientError: For 4xx HTTP errors.
            MLSDMServerError: For 5xx HTTP errors.
            MLSDMTimeoutError: If the request times out.
            MLSDMConnectionError: If unable to connect to the server.
        """
        try:
            response = self._session.get(url, timeout=self.timeout)
            return self._handle_response(response)
        except requests.Timeout as e:
            logger.error(f"Timeout during GET {url}: {e}")
            raise MLSDMTimeoutError(
                message=f"Request to {url} timed out",
                timeout_seconds=self.timeout,
            ) from e
        except requests.ConnectionError as e:
            logger.error(f"Connection error during GET {url}: {e}")
            raise MLSDMConnectionError(
                message=f"Failed to connect to {url}",
                url=url,
            ) from e

    def _post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        """Make a POST request.

        Args:
            url: Full URL to request.
            body: Request body as dictionary.

        Returns:
            Response JSON as dictionary.

        Raises:
            MLSDMClientError: For 4xx HTTP errors.
            MLSDMServerError: For 5xx HTTP errors.
            MLSDMTimeoutError: If the request times out.
            MLSDMConnectionError: If unable to connect to the server.
        """
        try:
            response = self._session.post(url, json=body, timeout=self.timeout)
            return self._handle_response(response)
        except requests.Timeout as e:
            logger.error(f"Timeout during POST {url}: {e}")
            raise MLSDMTimeoutError(
                message=f"Request to {url} timed out",
                timeout_seconds=self.timeout,
            ) from e
        except requests.ConnectionError as e:
            logger.error(f"Connection error during POST {url}: {e}")
            raise MLSDMConnectionError(
                message=f"Failed to connect to {url}",
                url=url,
            ) from e

    def _handle_response(self, response: requests.Response) -> dict[str, Any]:
        """Handle HTTP response and raise appropriate exceptions.

        Args:
            response: The requests Response object.

        Returns:
            Response JSON as dictionary.

        Raises:
            MLSDMClientError: For 4xx HTTP errors.
            MLSDMServerError: For 5xx HTTP errors.
        """
        status_code = response.status_code

        # Try to parse JSON response
        try:
            data = response.json()
        except ValueError:
            # If not JSON, use text
            data = {"message": response.text}

        # Handle success
        if 200 <= status_code < 300:
            return data

        # Extract error details from standard error response format
        error_info = data.get("error", {})
        error_type = error_info.get("error_type") if isinstance(error_info, dict) else None
        error_message = error_info.get("message") if isinstance(error_info, dict) else str(data)
        error_details = error_info.get("details") if isinstance(error_info, dict) else None
        debug_id = error_info.get("debug_id") if isinstance(error_info, dict) else None

        # Handle 4xx client errors
        if 400 <= status_code < 500:
            # Handle Pydantic validation errors (422)
            if status_code == 422 and "detail" in data:
                detail = data["detail"]
                if isinstance(detail, list) and len(detail) > 0:
                    error_message = detail[0].get("msg", str(detail))
                    error_type = "validation_error"
                    error_details = {"validation_errors": detail}

            raise MLSDMClientError(
                message=error_message or f"Client error: {status_code}",
                status_code=status_code,
                error_type=error_type,
                details=error_details,
                debug_id=debug_id,
            )

        # Handle 5xx server errors
        if status_code >= 500:
            raise MLSDMServerError(
                message=error_message or f"Server error: {status_code}",
                status_code=status_code,
                error_type=error_type,
                debug_id=debug_id,
            )

        # Unexpected status code
        raise MLSDMServerError(
            message=f"Unexpected status code: {status_code}",
            status_code=status_code,
        )

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()

    def __enter__(self) -> MLSDMHttpClient:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and close session."""
        self.close()
