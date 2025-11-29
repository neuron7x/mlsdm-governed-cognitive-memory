"""
MLSDM SDK: Public Python SDK for NeuroCognitiveEngine.

This module provides two client interfaces for interacting with
the NeuroCognitiveEngine:

1. NeuroCognitiveClient: Direct engine client (local in-process)
2. MLSDMHttpClient: HTTP client for remote API server

Both clients provide strictly-typed interfaces with proper exception handling.

## Quick Start

### Using the local engine client:
>>> from mlsdm.sdk import NeuroCognitiveClient
>>> client = NeuroCognitiveClient(backend="local_stub")
>>> result = client.generate("Hello, world!")
>>> print(result["response"])

### Using the HTTP client:
>>> from mlsdm.sdk import MLSDMHttpClient
>>> client = MLSDMHttpClient(base_url="http://localhost:8000")
>>> response = client.generate("Hello, world!")
>>> print(response.response)

## Error Handling

The HTTP client raises typed exceptions for error cases:
- MLSDMClientError: For 4xx client errors
- MLSDMServerError: For 5xx server errors
- MLSDMTimeoutError: For request timeouts
- MLSDMConnectionError: For connection failures
"""

from mlsdm.sdk.exceptions import (
    MLSDMClientError,
    MLSDMConnectionError,
    MLSDMError,
    MLSDMServerError,
    MLSDMTimeoutError,
)
from mlsdm.sdk.http_client import MLSDMHttpClient
from mlsdm.sdk.neuro_engine_client import NeuroCognitiveClient

__all__ = [
    # Clients
    "NeuroCognitiveClient",
    "MLSDMHttpClient",
    # Exceptions
    "MLSDMError",
    "MLSDMClientError",
    "MLSDMServerError",
    "MLSDMTimeoutError",
    "MLSDMConnectionError",
]

