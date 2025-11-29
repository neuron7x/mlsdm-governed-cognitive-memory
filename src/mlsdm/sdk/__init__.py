"""
MLSDM SDK: Public Python SDK for NeuroCognitiveEngine.

This module provides a high-level client interface for interacting with
the NeuroCognitiveEngine, supporting multiple backends and configurations.

SDK Contract Stability:
----------------------
The following are part of the stable SDK contract:

    - NeuroCognitiveClient: Main client class
    - GenerateResponseDTO: Typed response object
    - MLSDMError, MLSDMClientError, MLSDMServerError, MLSDMTimeoutError: Exception classes

Breaking changes to these will require a major version bump.
"""

from mlsdm.sdk.dto import GenerateResponseDTO
from mlsdm.sdk.exceptions import (
    MLSDMClientError,
    MLSDMConfigError,
    MLSDMError,
    MLSDMRateLimitError,
    MLSDMServerError,
    MLSDMTimeoutError,
    MLSDMValidationError,
)
from mlsdm.sdk.neuro_engine_client import NeuroCognitiveClient

__all__ = [
    # Main client
    "NeuroCognitiveClient",
    # DTOs
    "GenerateResponseDTO",
    # Exceptions
    "MLSDMError",
    "MLSDMClientError",
    "MLSDMValidationError",
    "MLSDMConfigError",
    "MLSDMServerError",
    "MLSDMTimeoutError",
    "MLSDMRateLimitError",
]
