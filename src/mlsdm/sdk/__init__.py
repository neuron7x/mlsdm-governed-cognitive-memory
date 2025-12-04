"""
MLSDM SDK: Public Python SDK for NeuroCognitiveEngine.

This module provides high-level client interfaces for interacting with
the NeuroCognitiveEngine and Neuro Memory Service.

Clients:
- NeuroCognitiveClient: Basic client for LLM generation with governance
- NeuroMemoryClient: Extended client with Memory API, Decision API, and Agent API

CORE-09 Contract:
- GenerateResponseDTO is the typed response DTO (stable contract)
- MLSDMClientError, MLSDMServerError, MLSDMTimeoutError for error handling
- GENERATE_RESPONSE_DTO_KEYS for contract validation
"""

from mlsdm.sdk.neuro_engine_client import (
    GENERATE_RESPONSE_DTO_KEYS,
    CognitiveStateDTO,
    GenerateResponseDTO,
    MLSDMClientError,
    MLSDMError,
    MLSDMServerError,
    MLSDMTimeoutError,
    NeuroCognitiveClient,
)
from mlsdm.sdk.neuro_memory_client import (
    AgentAction,
    AgentStepResult,
    ContourDecision,
    DecideResult,
    MemoryAppendResult,
    MemoryItem,
    MemoryQueryResult,
    NeuroMemoryClient,
    NeuroMemoryConnectionError,
    NeuroMemoryError,
    NeuroMemoryServiceError,
)

__all__ = [
    # Original client
    "NeuroCognitiveClient",
    "GenerateResponseDTO",
    "CognitiveStateDTO",
    "MLSDMError",
    "MLSDMClientError",
    "MLSDMServerError",
    "MLSDMTimeoutError",
    "GENERATE_RESPONSE_DTO_KEYS",
    # Extended client (Neuro Memory Service)
    "NeuroMemoryClient",
    "MemoryItem",
    "MemoryAppendResult",
    "MemoryQueryResult",
    "ContourDecision",
    "DecideResult",
    "AgentAction",
    "AgentStepResult",
    "NeuroMemoryError",
    "NeuroMemoryConnectionError",
    "NeuroMemoryServiceError",
]
