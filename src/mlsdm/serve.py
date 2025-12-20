from __future__ import annotations

import argparse
import os
from typing import Optional

from fastapi import FastAPI


def get_app(mode: str) -> FastAPI:
    """Return FastAPI application for the requested mode."""
    if mode == "api":
        from mlsdm.api.app import app

        return app
    if mode == "neuro":
        from mlsdm.service.neuro_engine_service import create_app

        return create_app()

    raise ValueError(f"Unknown mode '{mode}'. Expected 'api' or 'neuro'.")


def run_server(
    *,
    mode: str,
    host: str,
    port: int,
    log_level: str,
    reload: bool,
    config: Optional[str],
    backend: Optional[str],
    disable_rate_limit: bool,
) -> None:
    """Configure environment and run uvicorn for the selected mode."""
    if config:
        os.environ["CONFIG_PATH"] = config
    if backend:
        os.environ["LLM_BACKEND"] = backend
    if disable_rate_limit:
        os.environ["DISABLE_RATE_LIMIT"] = "1"

    app = get_app(mode)

    import uvicorn

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        reload=reload,
    )


def main() -> int:
    """CLI entrypoint for running MLSDM services."""
    parser = argparse.ArgumentParser(description="Run MLSDM HTTP services")
    parser.add_argument(
        "--mode",
        choices=["api", "neuro"],
        default="api",
        help="Service mode to run (default: api)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["local_stub", "openai"],
        help="LLM backend to use",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--disable-rate-limit",
        action="store_true",
        help="Disable rate limiting (for testing)",
    )

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
