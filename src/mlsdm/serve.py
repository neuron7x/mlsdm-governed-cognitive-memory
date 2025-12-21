"""Canonical server entrypoint for MLSDM services."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI


def get_app(mode: str) -> FastAPI:
    """Return the FastAPI app for the requested mode."""
    if mode == "api":
        from mlsdm.api.app import app as api_app

        return api_app

    if mode == "neuro":
        from mlsdm.service.neuro_engine_service import create_app

        return create_app()

    raise ValueError(f"Unsupported mode: {mode}")


def run_server(
    *,
    mode: str,
    host: str,
    port: int,
    log_level: str,
    reload: bool,
    config: str | None,
    backend: str | None,
    disable_rate_limit: bool,
    workers: int | None = None,
    timeout_keep_alive: int | None = None,
) -> int:
    """Start the server for the given mode using uvicorn."""
    if config:
        os.environ["CONFIG_PATH"] = config

    if backend:
        os.environ["LLM_BACKEND"] = backend

    if disable_rate_limit:
        os.environ["DISABLE_RATE_LIMIT"] = "1"

    if mode == "neuro":
        os.environ["HOST"] = host
        os.environ["PORT"] = str(port)

    import uvicorn

    uvicorn_kwargs: dict[str, object] = {
        "host": host,
        "port": port,
        "log_level": log_level,
        "reload": reload,
    }

    if workers is not None:
        uvicorn_kwargs["workers"] = workers

    if timeout_keep_alive is not None:
        uvicorn_kwargs["timeout_keep_alive"] = timeout_keep_alive

    uvicorn.run(get_app(mode), **uvicorn_kwargs)
    return 0


__all__ = ["get_app", "run_server"]
