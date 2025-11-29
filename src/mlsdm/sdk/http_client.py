"""
NeuroEngineHTTPClient: HTTP-based Python SDK for MLSDM API.

This module provides a typed HTTP client for interacting with the MLSDM
FastAPI service. It offers a clean, type-safe interface with proper
error handling and timeout management.

Example:
    >>> from mlsdm.sdk import NeuroEngineHTTPClient
    >>> client = NeuroEngineHTTPClient(base_url="http://localhost:8000")
    >>> response = client.generate("What is consciousness?")
    >>> print(response.response)
    >>> print(response.phase)
"""

from __future__ import annotations

import logging
from typing import Any

import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout

from mlsdm.api.schemas import GenerateResponseDTO
from mlsdm.sdk.exceptions import (
    MLSDMAuthenticationError,
    MLSDMClientError,
    MLSDMConnectionError,
    MLSDMRateLimitError,
    MLSDMServerError,
    MLSDMTimeoutError,
    MLSDMValidationError,
)

logger = logging.getLogger(__name__)


class NeuroEngineHTTPClient:
    """HTTP-based client for interacting with MLSDM API.

    This client provides a strongly typed interface for the MLSDM FastAPI
    service with proper error handling, timeout management, and
    authentication support.

    Args:
        base_url: Base URL of the MLSDM API (e.g., "http://localhost:8000")
        timeout: Request timeout in seconds (default: 30.0)
        api_key: Optional API key for authenticated endpoints

    Example:
        >>> # Basic usage
        >>> client = NeuroEngineHTTPClient(base_url="http://localhost:8000")
        >>> response = client.generate("Hello, world!")
        >>> print(response.response)

        >>> # With authentication and custom timeout
        >>> client = NeuroEngineHTTPClient(
        ...     base_url="http://localhost:8000",
        ...     timeout=60.0,
        ...     api_key="your-api-key"
        ... )
        >>> response = client.generate("Complex query", max_tokens=1024)

    Raises:
        MLSDMClientError: For 4xx client errors
        MLSDMServerError: For 5xx server errors
        MLSDMTimeoutError: When request times out
        MLSDMConnectionError: When unable to connect to API
    """

    # Default timeout for requests (in seconds)
    DEFAULT_TIMEOUT = 30.0

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize the HTTP client.

        Args:
            base_url: Base URL of the MLSDM API
            timeout: Request timeout in seconds (default: 30.0)
            api_key: Optional API key for authentication
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT
        self._api_key = api_key
        self._session = requests.Session()

        # Set default headers
        self._session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

        # Add authentication header if API key provided
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

    @property
    def base_url(self) -> str:
        """Get the base URL of the API."""
        return self._base_url

    @property
    def timeout(self) -> float:
        """Get the request timeout in seconds."""
        return self._timeout

    def _handle_response_error(self, response: requests.Response) -> None:
        """Handle HTTP error responses and raise appropriate exceptions.

        Args:
            response: HTTP response object

        Raises:
            MLSDMValidationError: For 422 validation errors
            MLSDMRateLimitError: For 429 rate limit errors
            MLSDMAuthenticationError: For 401 authentication errors
            MLSDMClientError: For other 4xx errors
            MLSDMServerError: For 5xx errors
        """
        status_code = response.status_code

        # Try to parse error response
        try:
            data = response.json()
        except ValueError:
            data = {}

        # Extract debug_id from headers if present
        debug_id = response.headers.get("x-request-id")

        # Handle 4xx client errors
        if 400 <= status_code < 500:
            # Handle FastAPI/Pydantic validation errors (422)
            if status_code == 422:
                detail = data.get("detail", [])
                raise MLSDMValidationError(
                    message="Request validation failed",
                    validation_errors=detail if isinstance(detail, list) else [detail],
                    debug_id=debug_id,
                )

            # Handle rate limiting (429)
            if status_code == 429:
                error = data.get("error", {})
                retry_after = response.headers.get("Retry-After")
                retry_after_float = float(retry_after) if retry_after else None
                raise MLSDMRateLimitError(
                    message=error.get("message", "Rate limit exceeded"),
                    retry_after=retry_after_float,
                    debug_id=debug_id,
                )

            # Handle authentication errors (401)
            if status_code == 401:
                error = data.get("error", {})
                raise MLSDMAuthenticationError(
                    message=error.get("message", data.get("detail", "Authentication required")),
                    debug_id=debug_id,
                )

            # Handle other 4xx errors
            error = data.get("error", {})
            raise MLSDMClientError(
                message=error.get("message", data.get("detail", "Client error")),
                status_code=status_code,
                error_type=error.get("error_type"),
                details=error.get("details"),
                debug_id=debug_id,
            )

        # Handle 5xx server errors
        if status_code >= 500:
            error = data.get("error", {})
            raise MLSDMServerError(
                message=error.get("message", "Internal server error"),
                status_code=status_code,
                error_type=error.get("error_type", "internal_error"),
                debug_id=debug_id,
            )

    def _request(
        self,
        method: str,
        endpoint: str,
        json_data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., "/generate")
            json_data: JSON body data for POST requests
            params: Query parameters for GET requests

        Returns:
            Parsed JSON response

        Raises:
            MLSDMClientError: For 4xx client errors
            MLSDMServerError: For 5xx server errors
            MLSDMTimeoutError: When request times out
            MLSDMConnectionError: When unable to connect
        """
        url = f"{self._base_url}{endpoint}"

        try:
            response = self._session.request(
                method=method,
                url=url,
                json=json_data,
                params=params,
                timeout=self._timeout,
            )

            # Check for errors
            if response.status_code >= 400:
                self._handle_response_error(response)

            result: dict[str, Any] = response.json()
            return result

        except Timeout as e:
            raise MLSDMTimeoutError(
                message="Request timed out",
                timeout_seconds=self._timeout,
            ) from e

        except RequestsConnectionError as e:
            raise MLSDMConnectionError(
                message=f"Could not connect to MLSDM API at {url}",
                url=url,
                original_error=e,
            ) from e

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        moral_value: float | None = None,
    ) -> GenerateResponseDTO:
        """Generate a response using the NeuroCognitiveEngine.

        This method calls the /generate endpoint to process the prompt
        through the complete cognitive pipeline including moral filtering,
        memory retrieval, and rhythm management.

        Args:
            prompt: Input text prompt to process
            max_tokens: Maximum number of tokens to generate (1-4096)
            moral_value: Moral threshold value (0.0-1.0)

        Returns:
            GenerateResponseDTO with response text and metadata

        Raises:
            MLSDMClientError: For 4xx client errors (invalid input, etc.)
            MLSDMServerError: For 5xx server errors
            MLSDMTimeoutError: When request times out
            MLSDMConnectionError: When unable to connect

        Example:
            >>> client = NeuroEngineHTTPClient()
            >>> response = client.generate(
            ...     prompt="What is consciousness?",
            ...     max_tokens=256,
            ...     moral_value=0.7
            ... )
            >>> print(f"Response: {response.response}")
            >>> print(f"Phase: {response.phase}")
            >>> print(f"Accepted: {response.accepted}")
        """
        # Build request body
        body: dict[str, Any] = {"prompt": prompt}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if moral_value is not None:
            body["moral_value"] = moral_value

        # Make request
        data = self._request("POST", "/generate", json_data=body)

        # Parse response into typed DTO
        return GenerateResponseDTO.from_dict(data)

    def health(self) -> dict[str, Any]:
        """Check basic health of the API.

        Returns:
            Health status dictionary with 'status' field

        Raises:
            MLSDMServerError: If service is unhealthy
            MLSDMTimeoutError: When request times out
            MLSDMConnectionError: When unable to connect

        Example:
            >>> client = NeuroEngineHTTPClient()
            >>> health = client.health()
            >>> print(health["status"])  # "healthy"
        """
        return self._request("GET", "/health")

    def health_liveness(self) -> dict[str, Any]:
        """Check liveness of the API (Kubernetes liveness probe).

        Returns:
            Liveness status with 'status' and 'timestamp'

        Raises:
            MLSDMServerError: If service is not alive
            MLSDMTimeoutError: When request times out
            MLSDMConnectionError: When unable to connect
        """
        return self._request("GET", "/health/liveness")

    def health_readiness(self) -> dict[str, Any]:
        """Check readiness of the API (Kubernetes readiness probe).

        Returns:
            Readiness status with 'ready', 'status', 'timestamp', 'checks',
            and optionally 'cognitive_state'

        Raises:
            MLSDMServerError: If service is not ready (503)
            MLSDMTimeoutError: When request times out
            MLSDMConnectionError: When unable to connect
        """
        return self._request("GET", "/health/readiness")

    def health_detailed(self) -> dict[str, Any]:
        """Get detailed health status of the API.

        Returns:
            Detailed health status with system info, memory state,
            phase, and statistics

        Raises:
            MLSDMServerError: If service is unhealthy (503)
            MLSDMTimeoutError: When request times out
            MLSDMConnectionError: When unable to connect
        """
        return self._request("GET", "/health/detailed")

    def status(self) -> dict[str, Any]:
        """Get extended service status with system info.

        Returns:
            Service status including version, backend, system metrics,
            and configuration info

        Raises:
            MLSDMServerError: For server errors
            MLSDMTimeoutError: When request times out
            MLSDMConnectionError: When unable to connect

        Example:
            >>> client = NeuroEngineHTTPClient()
            >>> status = client.status()
            >>> print(f"Version: {status['version']}")
            >>> print(f"Backend: {status['backend']}")
        """
        return self._request("GET", "/status")

    def infer(
        self,
        prompt: str,
        moral_value: float | None = None,
        max_tokens: int | None = None,
        secure_mode: bool = False,
        aphasia_mode: bool = False,
        rag_enabled: bool = True,
        context_top_k: int | None = None,
        user_intent: str | None = None,
    ) -> dict[str, Any]:
        """Generate a response with extended governance options.

        This method calls the /infer endpoint for fine-grained control
        over the cognitive pipeline including secure mode, aphasia
        detection/repair, and RAG retrieval.

        Args:
            prompt: Input text prompt to process
            moral_value: Moral threshold value (0.0-1.0, default: 0.5)
            max_tokens: Maximum number of tokens to generate (1-4096)
            secure_mode: Enable enhanced security filtering
            aphasia_mode: Enable aphasia detection and repair
            rag_enabled: Enable RAG-based context retrieval (default: True)
            context_top_k: Number of context items for RAG (1-100)
            user_intent: User intent category

        Returns:
            InferResponse-like dictionary with response and detailed metadata

        Raises:
            MLSDMClientError: For 4xx client errors
            MLSDMServerError: For 5xx server errors
            MLSDMTimeoutError: When request times out
            MLSDMConnectionError: When unable to connect

        Example:
            >>> client = NeuroEngineHTTPClient()
            >>> response = client.infer(
            ...     prompt="Explain quantum computing",
            ...     secure_mode=True,
            ...     rag_enabled=True,
            ...     context_top_k=3
            ... )
            >>> print(response["moral_metadata"]["secure_mode"])  # True
        """
        # Build request body
        body: dict[str, Any] = {
            "prompt": prompt,
            "secure_mode": secure_mode,
            "aphasia_mode": aphasia_mode,
            "rag_enabled": rag_enabled,
        }
        if moral_value is not None:
            body["moral_value"] = moral_value
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if context_top_k is not None:
            body["context_top_k"] = context_top_k
        if user_intent is not None:
            body["user_intent"] = user_intent

        # Make request
        return self._request("POST", "/infer", json_data=body)

    def close(self) -> None:
        """Close the HTTP session.

        Call this method when you're done using the client to release
        resources. Alternatively, use the client as a context manager.
        """
        self._session.close()

    def __enter__(self) -> NeuroEngineHTTPClient:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager and close session."""
        self.close()

    def __repr__(self) -> str:
        return f"NeuroEngineHTTPClient(base_url={self._base_url!r}, timeout={self._timeout})"
