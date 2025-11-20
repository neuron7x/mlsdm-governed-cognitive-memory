from typing import List, Dict

import logging
import os

import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from src.utils.config_loader import ConfigLoader
from src.core.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

app = FastAPI(title="mlsdm-governed-cognitive-memory", version="1.0.0")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

_config_path = os.getenv("CONFIG_PATH", "config/default_config.yaml")
_manager = MemoryManager(ConfigLoader.load_config(_config_path))


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    api_key = os.getenv("API_KEY")
    if api_key and token != api_key:
        raise HTTPException(status_code=401, detail="Invalid authentication")
    return token


class EventInput(BaseModel):  # type: ignore[misc]
    event_vector: List[float]
    moral_value: float


class StateResponse(BaseModel):  # type: ignore[misc]
    L1_norm: float
    L2_norm: float
    L3_norm: float
    current_phase: str
    latent_events_count: int
    accepted_events_count: int
    total_events_processed: int
    moral_filter_threshold: float


@app.post("/v1/process_event/", response_model=StateResponse)  # type: ignore[misc]
async def process_event(event: EventInput, user: str = Depends(get_current_user)) -> StateResponse:
    vec = np.array(event.event_vector, dtype=float)
    if vec.shape[0] != _manager.dimension:
        raise HTTPException(status_code=400, detail="Dimension mismatch.")
    await _manager.process_event(vec, event.moral_value)
    state: StateResponse = await get_state(user)
    return state


@app.get("/v1/state/", response_model=StateResponse)  # type: ignore[misc]
async def get_state(user: str = Depends(get_current_user)) -> StateResponse:
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


@app.get("/health")  # type: ignore[misc]
async def health() -> Dict[str, str]:
    return {"status": "healthy"}
