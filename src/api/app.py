import hashlib
import logging
import os
from typing import Dict, List

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from src.core.memory_manager import MemoryManager
from src.utils.config_loader import ConfigLoader
from src.utils.input_validator import InputValidator
from src.utils.rate_limiter import RateLimiter
from src.utils.security_logger import SecurityEventType, get_security_logger

logger = logging.getLogger(__name__)
security_logger = get_security_logger()

app = FastAPI(title="mlsdm-governed-cognitive-memory", version="1.0.0")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

_config_path = os.getenv("CONFIG_PATH", "config/default_config.yaml")
_manager = MemoryManager(ConfigLoader.load_config(_config_path))

# Initialize rate limiter (5 RPS per client as per SECURITY_POLICY.md)
# Can be disabled in testing with DISABLE_RATE_LIMIT=1
_rate_limiting_enabled = os.getenv("DISABLE_RATE_LIMIT") != "1"
_rate_limiter = RateLimiter(rate=5.0, capacity=10)

# Initialize input validator
_validator = InputValidator()


def _get_client_id(request: Request) -> str:
    """Get pseudonymized client identifier from request.
    
    Uses SHA256 hash of IP + User-Agent to create a unique but
    non-PII identifier for rate limiting and audit logging.
    """
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "")
    
    # Create hash for pseudonymization (no PII stored)
    identifier = f"{client_ip}:{user_agent}"
    hashed = hashlib.sha256(identifier.encode()).hexdigest()[:16]
    return hashed


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """Authenticate user with enhanced security logging."""
    api_key = os.getenv("API_KEY")
    
    if api_key and token != api_key:
        security_logger.log_auth_failure(
            client_id="unknown",
            reason="Invalid token"
        )
        raise HTTPException(status_code=401, detail="Invalid authentication")
    
    security_logger.log_auth_success(client_id="unknown")
    return token


class EventInput(BaseModel):
    event_vector: List[float]
    moral_value: float


class StateResponse(BaseModel):
    L1_norm: float
    L2_norm: float
    L3_norm: float
    current_phase: str
    latent_events_count: int
    accepted_events_count: int
    total_events_processed: int
    moral_filter_threshold: float


@app.post("/v1/process_event/", response_model=StateResponse)
async def process_event(
    event: EventInput,
    request: Request,
    user: str = Depends(get_current_user)
) -> StateResponse:
    """Process event with comprehensive security validation.
    
    Implements rate limiting, input validation, and audit logging
    as specified in SECURITY_POLICY.md.
    """
    client_id = _get_client_id(request)
    
    # Rate limiting check (can be disabled for testing)
    if _rate_limiting_enabled and not _rate_limiter.is_allowed(client_id):
        security_logger.log_rate_limit_exceeded(client_id=client_id)
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Maximum 5 requests per second."
        )
    
    # Validate moral value
    try:
        moral_value = _validator.validate_moral_value(event.moral_value)
    except ValueError as e:
        security_logger.log_invalid_input(
            client_id=client_id,
            error_message=str(e)
        )
        raise HTTPException(status_code=400, detail=str(e))
    
    # Validate and convert vector
    try:
        vec = _validator.validate_vector(
            event.event_vector,
            expected_dim=_manager.dimension,
            normalize=False
        )
    except ValueError as e:
        security_logger.log_invalid_input(
            client_id=client_id,
            error_message=str(e)
        )
        raise HTTPException(status_code=400, detail=str(e))
    
    # Process the event
    await _manager.process_event(vec, moral_value)
    
    return await get_state(request, user)


@app.get("/v1/state/", response_model=StateResponse)
async def get_state(
    request: Request,
    user: str = Depends(get_current_user)
) -> StateResponse:
    """Get system state with rate limiting."""
    client_id = _get_client_id(request)
    
    # Rate limiting check (can be disabled for testing)
    if _rate_limiting_enabled and not _rate_limiter.is_allowed(client_id):
        security_logger.log_rate_limit_exceeded(client_id=client_id)
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Maximum 5 requests per second."
        )
    
    L1, L2, L3 = _manager.memory.get_state()
    metrics = _manager.metrics_collector.get_metrics()
    return StateResponse(
        L1_norm=float(np.linalg.norm(L1)),
        L2_norm=float(np.linalg.norm(L2)),
        L3_norm=float(np.linalg.norm(L3)),
        current_phase=_manager.rhythm.get_current_phase(),
        latent_events_count=int(metrics["latent_events_count"]),
        accepted_events_count=int(metrics["accepted_events_count"]),
        total_events_processed=int(metrics["total_events_processed"]),
        moral_filter_threshold=float(_manager.filter.threshold),
    )


@app.on_event("startup")
async def startup_event():
    """Log system startup."""
    security_logger.log_system_event(
        SecurityEventType.STARTUP,
        "MLSDM Governed Cognitive Memory API started",
        additional_data={
            "version": "1.0.0",
            "dimension": _manager.dimension
        }
    )


@app.on_event("shutdown")
async def shutdown_event():
    """Log system shutdown."""
    security_logger.log_system_event(
        SecurityEventType.SHUTDOWN,
        "MLSDM Governed Cognitive Memory API shutting down"
    )


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint (no authentication required)."""
    return {"status": "healthy"}
