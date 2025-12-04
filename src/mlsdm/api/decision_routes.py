"""
MLSDM Decision API Routes.

Product Layer endpoints for Decision/LLM operations:
- POST /v1/decide - Decision-making with moral/risk governance
- POST /v1/agent/step - Agent step protocol for external LLM/agent integration
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Any, Literal

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from mlsdm.utils.rate_limiter import RateLimiter
from mlsdm.utils.security_logger import get_security_logger

logger = logging.getLogger(__name__)
security_logger = get_security_logger()

router = APIRouter(prefix="/v1", tags=["Decision"])

# Rate limiter (can be disabled for testing)
_rate_limiting_enabled = os.getenv("DISABLE_RATE_LIMIT") != "1"
_rate_limiter = RateLimiter(rate=5.0, capacity=10)


# ============================================================
# Request/Response Models
# ============================================================


class DecideRequest(BaseModel):
    """Request model for decision endpoint."""

    prompt: str = Field(
        ..., min_length=1, max_length=20000,
        description="Input prompt for decision-making"
    )
    context: str | None = Field(
        None, max_length=50000,
        description="Additional context for the decision"
    )
    user_id: str | None = Field(
        None, max_length=100,
        description="User identifier"
    )
    session_id: str | None = Field(
        None, max_length=100,
        description="Session identifier"
    )
    agent_id: str | None = Field(
        None, max_length=100,
        description="Agent identifier"
    )
    risk_level: Literal["low", "medium", "high", "critical"] = Field(
        default="medium",
        description="Risk level for this decision"
    )
    priority: Literal["low", "normal", "high", "urgent"] = Field(
        default="normal",
        description="Priority level for processing"
    )
    mode: Literal["standard", "cautious", "confident", "emergency"] = Field(
        default="standard",
        description="Decision mode affecting moral thresholds"
    )
    max_tokens: int = Field(
        default=512, ge=1, le=4096,
        description="Maximum tokens for response"
    )
    use_memory: bool = Field(
        default=True,
        description="Whether to use memory for context retrieval"
    )
    context_top_k: int = Field(
        default=5, ge=0, le=100,
        description="Number of memory items to retrieve for context"
    )
    metadata: dict[str, Any] | None = Field(
        None, description="Additional metadata for the decision"
    )


class ContourDecision(BaseModel):
    """Decision from a governance contour."""

    contour: str = Field(description="Name of the governance contour")
    passed: bool = Field(description="Whether the check passed")
    score: float | None = Field(None, description="Score from the contour (0.0-1.0)")
    threshold: float | None = Field(None, description="Threshold applied")
    notes: str | None = Field(None, description="Additional notes")


class DecideResponse(BaseModel):
    """Response model for decision endpoint."""

    response: str = Field(description="Generated response/decision")
    accepted: bool = Field(description="Whether the decision was accepted")
    phase: str = Field(description="Current cognitive phase")
    contour_decisions: list[ContourDecision] = Field(
        default_factory=list, 
        description="Decisions from governance contours (moral, risk, emergency)"
    )
    memory_context_used: int = Field(
        default=0, description="Number of memory items used for context"
    )
    risk_assessment: dict[str, Any] | None = Field(
        None, description="Risk assessment details"
    )
    timing: dict[str, float] | None = Field(
        None, description="Timing metrics in milliseconds"
    )
    decision_id: str | None = Field(
        None, description="Unique identifier for this decision"
    )
    message: str | None = Field(None, description="Status message")


class AgentStepRequest(BaseModel):
    """Request model for agent step endpoint."""

    agent_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Unique agent identifier"
    )
    user_id: str | None = Field(
        None, max_length=100,
        description="User identifier the agent is acting for"
    )
    session_id: str | None = Field(
        None, max_length=100,
        description="Session identifier"
    )
    observation: str = Field(
        ..., min_length=1, max_length=50000,
        description="Current observation/input for the agent"
    )
    internal_state: dict[str, Any] | None = Field(
        None, description="Agent's internal state to persist"
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        None, description="Tool calls from previous step"
    )
    tool_results: list[dict[str, Any]] | None = Field(
        None, description="Results from tool calls"
    )
    max_tokens: int = Field(
        default=512, ge=1, le=4096,
        description="Maximum tokens for response"
    )
    moral_value: float = Field(
        default=0.8, ge=0.0, le=1.0,
        description="Moral value for this step"
    )


class AgentAction(BaseModel):
    """Agent action to take."""

    action_type: Literal["respond", "tool_call", "wait", "terminate"] = Field(
        description="Type of action"
    )
    content: str | None = Field(None, description="Response content if action_type is 'respond'")
    tool_calls: list[dict[str, Any]] | None = Field(
        None, description="Tool calls if action_type is 'tool_call'"
    )


class AgentStepResponse(BaseModel):
    """Response model for agent step endpoint."""

    action: AgentAction = Field(description="Action for the agent to take")
    response: str = Field(description="Full response text")
    phase: str = Field(description="Current cognitive phase")
    accepted: bool = Field(description="Whether the step was accepted")
    updated_state: dict[str, Any] | None = Field(
        None, description="Updated internal state for the agent"
    )
    memory_updated: bool = Field(
        default=False, description="Whether memory was updated"
    )
    memory_context_used: int = Field(
        default=0, description="Number of memory items used"
    )
    step_id: str | None = Field(None, description="Unique identifier for this step")
    timing: dict[str, float] | None = Field(None, description="Timing metrics")
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


def set_engine(engine: Any) -> None:
    """Set the engine for decision operations.
    
    This must be called during app initialization.
    """
    global _engine
    _engine = engine


def _get_client_id(request: Request) -> str:
    """Get pseudonymized client identifier from request."""
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "")
    identifier = f"{client_ip}:{user_agent}"
    return hashlib.sha256(identifier.encode()).hexdigest()[:16]


def _generate_decision_id(prompt: str, timestamp: float) -> str:
    """Generate a unique decision ID."""
    data = f"{prompt}{timestamp}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _get_moral_value_for_mode(mode: str, risk_level: str) -> float:
    """Calculate moral value based on mode and risk level."""
    # Base moral values for different modes
    mode_values = {
        "standard": 0.5,
        "cautious": 0.7,
        "confident": 0.4,
        "emergency": 0.3,
    }
    
    # Risk level adjustments
    risk_adjustments = {
        "low": -0.1,
        "medium": 0.0,
        "high": 0.1,
        "critical": 0.2,
    }
    
    base = mode_values.get(mode, 0.5)
    adjustment = risk_adjustments.get(risk_level, 0.0)
    
    return min(1.0, max(0.0, base + adjustment))


# ============================================================
# Decision API Endpoints
# ============================================================


@router.post(
    "/decide",
    response_model=DecideResponse,
    responses={
        400: {"description": "Invalid input"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
        503: {"description": "Service unavailable"},
    },
)
async def decide(
    request_body: DecideRequest,
    request: Request,
) -> DecideResponse | JSONResponse:
    """Make a governed decision using the cognitive engine.

    This endpoint provides decision-making with:
    - Moral filtering based on mode and risk level
    - Memory-augmented context retrieval
    - Multi-contour governance (moral, risk, emergency)
    - Detailed timing and decision audit trail

    Args:
        request_body: Decision request with prompt and governance parameters.
        request: FastAPI request object.

    Returns:
        DecideResponse with decision, governance details, and timing.
    """
    client_id = _get_client_id(request)
    start_time = time.perf_counter()

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
                    "message": "Decision engine not initialized",
                    "details": None,
                }
            },
        )

    try:
        timestamp = time.time()
        decision_id = _generate_decision_id(request_body.prompt, timestamp)

        # Calculate moral value based on mode and risk level
        moral_value = _get_moral_value_for_mode(request_body.mode, request_body.risk_level)

        # Build prompt with context if provided
        full_prompt = request_body.prompt
        if request_body.context:
            full_prompt = f"Context:\n{request_body.context}\n\nQuery:\n{request_body.prompt}"

        # Generate using the engine
        result = _engine.generate(
            prompt=full_prompt,
            max_tokens=request_body.max_tokens,
            moral_value=moral_value,
            context_top_k=request_body.context_top_k if request_body.use_memory else 0,
            user_intent="decision",
        )

        # Extract state info
        mlsdm_state = result.get("mlsdm", {})
        phase = mlsdm_state.get("phase", "unknown")
        rejected_at = result.get("rejected_at")
        accepted = rejected_at is None and result.get("error") is None and bool(result.get("response"))

        # Build contour decisions
        contour_decisions = []
        
        # Moral contour
        contour_decisions.append(ContourDecision(
            contour="moral_filter",
            passed=accepted,
            score=moral_value,
            threshold=mlsdm_state.get("moral_threshold", 0.5),
            notes=f"mode={request_body.mode}, risk={request_body.risk_level}"
        ))

        # Risk contour (simulated)
        risk_passed = request_body.risk_level != "critical" or request_body.mode == "emergency"
        contour_decisions.append(ContourDecision(
            contour="risk_assessment",
            passed=risk_passed,
            score=0.8 if risk_passed else 0.3,
            threshold=0.5,
            notes=f"risk_level={request_body.risk_level}"
        ))

        # Emergency contour
        emergency_shutdown = mlsdm_state.get("emergency_shutdown", False)
        contour_decisions.append(ContourDecision(
            contour="emergency_state",
            passed=not emergency_shutdown,
            score=0.0 if emergency_shutdown else 1.0,
            threshold=0.5,
            notes="system_healthy" if not emergency_shutdown else "emergency_shutdown_active"
        ))

        # Calculate timing
        total_time = (time.perf_counter() - start_time) * 1000
        timing = result.get("timing", {})
        timing["total_endpoint"] = total_time

        return DecideResponse(
            response=result.get("response", ""),
            accepted=accepted,
            phase=phase,
            contour_decisions=contour_decisions,
            memory_context_used=mlsdm_state.get("context_items", 0),
            risk_assessment={
                "level": request_body.risk_level,
                "mode": request_body.mode,
                "computed_moral_value": moral_value,
            },
            timing=timing,
            decision_id=decision_id,
            message="Decision processed" if accepted else result.get("error", {}).get("message", "Decision rejected"),
        )

    except Exception as e:
        logger.exception("Error in decide endpoint")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "error_type": "internal_error",
                    "message": f"Failed to process decision: {type(e).__name__}",
                    "details": None,
                }
            },
        )


@router.post(
    "/agent/step",
    response_model=AgentStepResponse,
    responses={
        400: {"description": "Invalid input"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
        503: {"description": "Service unavailable"},
    },
)
async def agent_step(
    request_body: AgentStepRequest,
    request: Request,
) -> AgentStepResponse | JSONResponse:
    """Process a single step for an external agent.

    This endpoint provides a protocol for integrating external LLM agents
    with MLSDM as their memory and decision backend.

    The agent workflow:
    1. Agent sends observation and internal state
    2. MLSDM updates memory with the observation
    3. MLSDM retrieves relevant context
    4. MLSDM generates a response/action
    5. Agent receives action, updated state, and memory updates

    Args:
        request_body: Agent step request with observation and state.
        request: FastAPI request object.

    Returns:
        AgentStepResponse with action to take and updated state.
    """
    client_id = _get_client_id(request)
    start_time = time.perf_counter()

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
                    "message": "Agent engine not initialized",
                    "details": None,
                }
            },
        )

    try:
        timestamp = time.time()
        step_id = _generate_decision_id(f"{request_body.agent_id}:{request_body.observation}", timestamp)

        # Build prompt incorporating observation and tool results
        prompt_parts = [f"Agent ID: {request_body.agent_id}"]
        
        if request_body.tool_results:
            prompt_parts.append("\nTool Results:")
            for result in request_body.tool_results:
                prompt_parts.append(f"- {result.get('tool', 'unknown')}: {result.get('result', '')}")
        
        prompt_parts.append(f"\nObservation: {request_body.observation}")
        prompt_parts.append("\nRespond with the next action to take.")
        
        full_prompt = "\n".join(prompt_parts)

        # Generate response using engine
        result = _engine.generate(
            prompt=full_prompt,
            max_tokens=request_body.max_tokens,
            moral_value=request_body.moral_value,
            context_top_k=5,  # Always use memory for agent steps
            user_intent="agent",
        )

        # Extract state info
        mlsdm_state = result.get("mlsdm", {})
        phase = mlsdm_state.get("phase", "unknown")
        rejected_at = result.get("rejected_at")
        accepted = rejected_at is None and result.get("error") is None and bool(result.get("response"))

        response_text = result.get("response", "")

        # Determine action type based on response content
        action_type: Literal["respond", "tool_call", "wait", "terminate"] = "respond"
        tool_calls = None

        if not accepted:
            action_type = "wait"
        elif "terminate" in response_text.lower() or "goodbye" in response_text.lower():
            action_type = "terminate"
        elif "[tool:" in response_text.lower() or "use tool" in response_text.lower():
            action_type = "tool_call"
            # Parse tool calls from response (simplified)
            tool_calls = [{"tool": "extracted_tool", "params": {}}]

        action = AgentAction(
            action_type=action_type,
            content=response_text if action_type == "respond" else None,
            tool_calls=tool_calls,
        )

        # Update internal state
        updated_state = request_body.internal_state or {}
        updated_state["last_step_id"] = step_id
        updated_state["last_phase"] = phase
        updated_state["step_count"] = updated_state.get("step_count", 0) + 1

        # Calculate timing
        total_time = (time.perf_counter() - start_time) * 1000
        timing = result.get("timing", {})
        timing["total_endpoint"] = total_time

        return AgentStepResponse(
            action=action,
            response=response_text,
            phase=phase,
            accepted=accepted,
            updated_state=updated_state,
            memory_updated=accepted,
            memory_context_used=mlsdm_state.get("context_items", 0),
            step_id=step_id,
            timing=timing,
            message="Step processed" if accepted else "Step rejected",
        )

    except Exception as e:
        logger.exception("Error in agent step endpoint")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "error_type": "internal_error",
                    "message": f"Failed to process agent step: {type(e).__name__}",
                    "details": None,
                }
            },
        )
