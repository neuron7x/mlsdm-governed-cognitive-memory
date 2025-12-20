from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI


def get_app(mode: str) -> "FastAPI":
    """Return the FastAPI application for the given mode."""
    if mode == "api":
        from mlsdm.api.app import app

        return app
    if mode == "neuro":
        from mlsdm.service.neuro_engine_service import create_app

        return create_app()
    raise ValueError(f"Unsupported mode '{mode}'. Expected one of: 'api', 'neuro'.")


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
) -> None:
    """Configure environment and start uvicorn for the selected mode."""
    if config:
        os.environ["CONFIG_PATH"] = config
    if backend:
        os.environ["LLM_BACKEND"] = backend
    if disable_rate_limit:
        os.environ["DISABLE_RATE_LIMIT"] = "1"

    app = get_app(mode)

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

    uvicorn.run(app, **uvicorn_kwargs)


def main() -> int:  # pragma: no cover - convenience entrypoint
    """Optional module entrypoint for `python -m mlsdm.serve`."""
    import argparse

    parser = argparse.ArgumentParser(description="Run MLSDM services")
    parser.add_argument("--mode", choices=["api", "neuro"], default="api")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--backend")
    parser.add_argument("--disable-rate-limit", action="store_true")
    args = parser.parse_args()

    run_server(
        mode=args.mode,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
        config=args.config,
        backend=args.backend,
        disable_rate_limit=args.disable_rate_limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
