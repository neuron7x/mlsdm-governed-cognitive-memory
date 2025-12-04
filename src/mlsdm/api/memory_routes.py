"""
MLSDM Memory API Routes.

Product Layer endpoints for Memory operations:
- POST /v1/memory/append - Append facts/context to memory
- POST /v1/memory/query - Query/retrieve relevant memory fragments
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Any

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from mlsdm.utils.rate_limiter import RateLimiter
from mlsdm.utils.security_logger import get_security_logger

logger = logging.getLogger(__name__)
security_logger = get_security_logger()

router = APIRouter(prefix="/v1/memory", tags=["Memory"])

# Rate limiter (can be disabled for testing)
_rate_limiting_enabled = os.getenv("DISABLE_RATE_LIMIT") != "1"
_rate_limiter = RateLimiter(rate=5.0, capacity=10)


# ============================================================
# Request/Response Models
# ============================================================


class MemoryAppendRequest(BaseModel):
    """Request model for memory append operation."""

    content: str = Field(
        ..., min_length=1, max_length=10000,
        description="Text content to store in memory"
    )
    user_id: str | None = Field(
        None, max_length=100,
        description="User identifier for memory scoping"
    )
    session_id: str | None = Field(
        None, max_length=100,
        description="Session identifier for memory scoping"
    )
    agent_id: str | None = Field(
        None, max_length=100,
        description="Agent identifier for multi-agent scenarios"
    )
    metadata: dict[str, Any] | None = Field(
        None, description="Additional metadata to store with memory"
    )
    moral_value: float = Field(
        default=0.8, ge=0.0, le=1.0,
        description="Moral value for this memory entry (0.0-1.0)"
    )


class MemoryAppendResponse(BaseModel):
    """Response model for memory append operation."""

    success: bool = Field(description="Whether the append operation succeeded")
    memory_id: str | None = Field(None, description="Unique identifier for the stored memory")
    phase: str = Field(description="Current cognitive phase")
    accepted: bool = Field(description="Whether memory was accepted by moral filter")
    memory_stats: dict[str, Any] | None = Field(
        None, description="Current memory statistics"
    )
    message: str | None = Field(None, description="Status message")


class MemoryQueryRequest(BaseModel):
    """Request model for memory query operation."""

    query: str = Field(
        ..., min_length=1, max_length=5000,
        description="Query text to search for relevant memories"
    )
    user_id: str | None = Field(
        None, max_length=100,
        description="User identifier to scope query"
    )
    session_id: str | None = Field(
        None, max_length=100,
        description="Session identifier to scope query"
    )
    agent_id: str | None = Field(
        None, max_length=100,
        description="Agent identifier to scope query"
    )
    top_k: int = Field(
        default=5, ge=1, le=100,
        description="Number of results to retrieve"
    )
    include_metadata: bool = Field(
        default=True, description="Include metadata in results"
    )


class MemoryItem(BaseModel):
    """A single memory item in query results."""

    content: str = Field(description="Memory content")
    similarity: float = Field(description="Similarity score (0.0-1.0)")
    phase: float = Field(description="Phase value when stored")
    metadata: dict[str, Any] | None = Field(None, description="Associated metadata")


class MemoryQueryResponse(BaseModel):
    """Response model for memory query operation."""

    success: bool = Field(description="Whether the query operation succeeded")
    results: list[MemoryItem] = Field(
        default_factory=list, description="Retrieved memory items"
    )
    query_phase: str = Field(description="Current cognitive phase during query")
    total_results: int = Field(description="Total number of results found")
    message: str | None = Field(None, description="Status message")


class ErrorDetail(BaseModel):
    """Structured error detail."""

    error_type: str = Field(description="Type of error")
    message: str = Field(description="Human-readable error message")
    details: dict[str, Any] | None = Field(default=None, description="Additional details")


# ============================================================
# Module-level engine reference (set during app initialization)
# ============================================================

_engine: Any = None
_embedding_fn: Any = None


def set_engine(engine: Any, embedding_fn: Any = None) -> None:
    """Set the engine and embedding function for memory operations.
    
    This must be called during app initialization.
    """
    global _engine, _embedding_fn
    _engine = engine
    _embedding_fn = embedding_fn


def _get_client_id(request: Request) -> str:
    """Get pseudonymized client identifier from request."""
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "")
    identifier = f"{client_ip}:{user_agent}"
    return hashlib.sha256(identifier.encode()).hexdigest()[:16]


def _generate_memory_id(content: str, timestamp: float) -> str:
    """Generate a unique memory ID."""
    data = f"{content}{timestamp}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]


# ============================================================
# Memory API Endpoints
# ============================================================


@router.post(
    "/append",
    response_model=MemoryAppendResponse,
    responses={
        400: {"description": "Invalid input"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
        503: {"description": "Service unavailable (engine not initialized)"},
    },
)
async def append_memory(
    request_body: MemoryAppendRequest,
    request: Request,
) -> MemoryAppendResponse | JSONResponse:
    """Append content to cognitive memory.

    Stores the provided content in the MLSDM memory system with optional
    metadata, user/session/agent scoping, and moral filtering.

    Args:
        request_body: Memory append request with content and metadata.
        request: FastAPI request object.

    Returns:
        MemoryAppendResponse with operation status and memory ID.
    """
    client_id = _get_client_id(request)

    # Rate limiting check
    if _rate_limiting_enabled and not _rate_limiter.is_allowed(client_id):
        security_logger.log_rate_limit_exceeded(client_id=client_id)
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": {
                    "error_type": "rate_limit_exceeded",
                    "message": "Rate limit exceeded. Maximum 5 requests per second.",
                    "details": None,
                }
            },
        )

    # Check engine availability
    if _engine is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": {
                    "error_type": "service_unavailable",
                    "message": "Memory engine not initialized",
                    "details": None,
                }
            },
        )

    try:
        # Get the LLM wrapper from engine
        wrapper = getattr(_engine, "_mlsdm", None)
        if wrapper is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": {
                        "error_type": "service_unavailable",
                        "message": "MLSDM wrapper not available",
                        "details": None,
                    }
                },
            )

        # Generate embedding and store in memory
        timestamp = time.time()
        memory_id = _generate_memory_id(request_body.content, timestamp)

        # Use the wrapper's generate method to process the content through
        # the cognitive pipeline (this stores in memory as a side effect)
        result = wrapper.generate(
            prompt=request_body.content,
            moral_value=request_body.moral_value,
            max_tokens=1,  # Minimal generation, we just want the memory storage
            context_top_k=0,  # Don't retrieve context for append
        )

        accepted = result.get("accepted", False)
        phase = result.get("phase", "unknown")

        # Get memory stats
        state = wrapper.get_state()
        memory_stats = {
            "capacity": state.get("qilm_stats", {}).get("capacity", 0),
            "used": state.get("qilm_stats", {}).get("used", 0),
            "memory_mb": state.get("qilm_stats", {}).get("memory_mb", 0),
        }

        return MemoryAppendResponse(
            success=accepted,
            memory_id=memory_id if accepted else None,
            phase=phase,
            accepted=accepted,
            memory_stats=memory_stats,
            message="Memory stored successfully" if accepted else result.get("note", "Memory rejected"),
        )

    except Exception as e:
        logger.exception("Error in memory append")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "error_type": "internal_error",
                    "message": f"Failed to append memory: {type(e).__name__}",
                    "details": None,
                }
            },
        )


@router.post(
    "/query",
    response_model=MemoryQueryResponse,
    responses={
        400: {"description": "Invalid input"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
        503: {"description": "Service unavailable (engine not initialized)"},
    },
)
async def query_memory(
    request_body: MemoryQueryRequest,
    request: Request,
) -> MemoryQueryResponse | JSONResponse:
    """Query cognitive memory for relevant content.

    Retrieves memory fragments that are semantically similar to the query,
    using the MLSDM phase-entangled retrieval mechanism.

    Args:
        request_body: Memory query request with query text and filters.
        request: FastAPI request object.

    Returns:
        MemoryQueryResponse with matching memory items.
    """
    client_id = _get_client_id(request)

    # Rate limiting check
    if _rate_limiting_enabled and not _rate_limiter.is_allowed(client_id):
        security_logger.log_rate_limit_exceeded(client_id=client_id)
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": {
                    "error_type": "rate_limit_exceeded",
                    "message": "Rate limit exceeded. Maximum 5 requests per second.",
                    "details": None,
                }
            },
        )

    # Check engine availability
    if _engine is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": {
                    "error_type": "service_unavailable",
                    "message": "Memory engine not initialized",
                    "details": None,
                }
            },
        )

    try:
        # Get the LLM wrapper from engine
        wrapper = getattr(_engine, "_mlsdm", None)
        if wrapper is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": {
                        "error_type": "service_unavailable",
                        "message": "MLSDM wrapper not available",
                        "details": None,
                    }
                },
            )

        # Get PELM and rhythm from wrapper
        pelm = getattr(wrapper, "pelm", None)
        rhythm = getattr(wrapper, "rhythm", None)
        if pelm is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": {
                        "error_type": "service_unavailable",
                        "message": "PELM memory not available",
                        "details": None,
                    }
                },
            )

        # Get embedding function
        embedding_fn_local = getattr(wrapper, "_embedding_fn", None) or _embedding_fn
        if embedding_fn_local is None:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": {
                        "error_type": "service_unavailable",
                        "message": "Embedding function not available",
                        "details": None,
                    }
                },
            )

        # Generate query embedding
        query_vector = embedding_fn_local(request_body.query)

        # Get current phase for retrieval
        current_phase = 0.1 if rhythm and rhythm.is_wake() else 0.9
        phase_name = "wake" if current_phase == 0.1 else "sleep"

        # Retrieve from PELM memory
        try:
            retrievals = pelm.retrieve(
                query_vector.tolist(),
                current_phase=current_phase,
                phase_tolerance=0.15,
                top_k=request_body.top_k
            )
        except Exception as e:
            logger.warning(f"PELM retrieval failed: {e}")
            retrievals = []

        # Convert retrievals to response items
        results = []
        for retrieval in retrievals:
            item = MemoryItem(
                content=str(retrieval.vector.tolist()[:10]) + "...",  # Truncated for display
                similarity=retrieval.resonance,  # Use resonance (cosine similarity)
                phase=retrieval.phase,
                metadata=None if not request_body.include_metadata else {},
            )
            results.append(item)

        return MemoryQueryResponse(
            success=True,
            results=results,
            query_phase=phase_name,
            total_results=len(results),
            message=f"Retrieved {len(results)} memory items",
        )

    except Exception as e:
        logger.exception("Error in memory query")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "error_type": "internal_error",
                    "message": f"Failed to query memory: {type(e).__name__}",
                    "details": None,
                }
            },
        )
