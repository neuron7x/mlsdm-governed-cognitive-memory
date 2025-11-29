"""
MLSDM SDK: Public Python SDK for NeuroCognitiveEngine.

This module provides:
- NeuroCognitiveClient: Direct engine integration for local use
- NeuroEngineHTTPClient: HTTP client for remote API access

Exceptions for HTTP client:
- MLSDMClientError: For 4xx client errors
- MLSDMServerError: For 5xx server errors
- MLSDMTimeoutError: For request timeouts
- MLSDMConnectionError: For connection failures
- MLSDMValidationError: For validation errors (422)
- MLSDMRateLimitError: For rate limit errors (429)
- MLSDMAuthenticationError: For auth errors (401)
"""

from mlsdm.sdk.exceptions import (
    MLSDMAuthenticationError,
    MLSDMClientError,
    MLSDMConnectionError,
    MLSDMError,
    MLSDMRateLimitError,
    MLSDMServerError,
    MLSDMTimeoutError,
    MLSDMValidationError,
)
from mlsdm.sdk.http_client import NeuroEngineHTTPClient
from mlsdm.sdk.neuro_engine_client import NeuroCognitiveClient

__all__ = [
    # Clients
    "NeuroCognitiveClient",
    "NeuroEngineHTTPClient",
    # Exceptions
    "MLSDMError",
    "MLSDMClientError",
    "MLSDMServerError",
    "MLSDMTimeoutError",
    "MLSDMConnectionError",
    "MLSDMValidationError",
    "MLSDMRateLimitError",
    "MLSDMAuthenticationError",
]
