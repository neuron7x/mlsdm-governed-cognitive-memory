"""
MLSDM API: FastAPI-based HTTP API for the NeuroCognitiveEngine.

This module provides:
- app: FastAPI application instance
- health: Health check endpoints router
- schemas: Pydantic models for API request/response schemas
- lifecycle: Application lifecycle management
- middleware: HTTP middleware components
"""

from mlsdm.api import health, lifecycle, middleware, schemas
from mlsdm.api.app import app

__all__ = [
    "app",
    "health",
    "lifecycle",
    "middleware",
    "schemas",
]
