"""
MLSDM Contracts Module.

This module defines stable API contracts for internal services and external endpoints.

Contracts:
- errors: Standard error models (ApiError)
- engine_models: NeuroCognitiveEngine input/output contracts

CONTRACT STABILITY:
All models in this module are part of the stable API contract.
Do not modify field names or types without a major version bump.
"""

from mlsdm.contracts.engine_models import (
    EngineErrorInfo,
    EngineResult,
    EngineResultMeta,
    EngineTiming,
    EngineValidationStep,
)
from mlsdm.contracts.errors import ApiError

__all__ = [
    "ApiError",
    "EngineResult",
    "EngineErrorInfo",
    "EngineResultMeta",
    "EngineTiming",
    "EngineValidationStep",
]
