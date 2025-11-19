import asyncio
import yaml
import argparse
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from src.manager import CognitiveMemoryManager
from prometheus_client import start_http_server

with open("config/default_config.yaml") as f:
    config = yaml.safe_load(f)

manager = CognitiveMemoryManager(config)

app = FastAPI(title="MLSDM Governed Cognitive Memory")

class EventInput(BaseModel):
    event_vector: list[float]
    moral_value: float

@app.post("/v1/process")
async def process(payload: EventInput):
    try:
        state = await manager.process_event(np.array(payload.event_vector), payload.moral_value)
        return state
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", action="store_true")
    args = parser.parse_args()

    start_http_server(8001)  # Prometheus

    if args.api:
        uvicorn.run(app, host="0.0.0.0", port=8000)
