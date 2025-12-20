"""Deprecated service shim that delegates to the canonical MLSDM API app."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request

from mlsdm.api.app import GenerateRequest, create_app as _create_canonical_app, generate

if TYPE_CHECKING:
    from fastapi import FastAPI


def create_app() -> FastAPI:
    """Return the canonical FastAPI application."""
    app = _create_canonical_app()

    if getattr(app.state, "neuro_route_registered", False):
        return app

    router = APIRouter()

    @router.post("/v1/neuro/generate")
    async def neuro_generate(request_body: GenerateRequest, request: Request):
        return await generate(request_body, request)

    app.include_router(router)
    app.state.neuro_route_registered = True

    return app


def main() -> None:
    """Start the canonical HTTP API server (legacy shim)."""
    from mlsdm.serve import run_server

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    run_server(
        mode="neuro",
        host=host,
        port=port,
        log_level=os.environ.get("LOG_LEVEL", "info"),
        reload=os.environ.get("RELOAD", "").lower() == "true",
        config=os.environ.get("CONFIG_PATH"),
        backend=os.environ.get("LLM_BACKEND"),
        disable_rate_limit=os.environ.get("DISABLE_RATE_LIMIT") == "1",
    )


__all__ = ["create_app", "main"]
