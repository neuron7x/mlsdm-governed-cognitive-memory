"""
MLSDM API module.

This module provides the FastAPI-based HTTP API for the MLSDM service.

Exports:
    - schemas: Centralized Pydantic schemas for API contracts
    - health: Health check endpoints
    - lifecycle: Application lifecycle management
"""

from mlsdm.api.schemas import (
    DetailedHealthStatus,
    ErrorDetail,
    ErrorResponse,
    EventInput,
    GenerateRequest,
    GenerateResponse,
    HealthStatus,
    InferRequest,
    InferResponse,
    ReadinessStatus,
    SimpleHealthStatus,
    StateResponse,
)

__all__ = [
    # Request schemas
    "GenerateRequest",
    "InferRequest",
    "EventInput",
    # Response schemas
    "GenerateResponse",
    "InferResponse",
    "StateResponse",
    # Error schemas
    "ErrorDetail",
    "ErrorResponse",
    # Health schemas
    "SimpleHealthStatus",
    "HealthStatus",
    "ReadinessStatus",
    "DetailedHealthStatus",
]
