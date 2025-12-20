"""Canonical runtime entrypoint for the MLSDM HTTP API."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlsdm.serve import get_app, run_server

if TYPE_CHECKING:
    from fastapi import FastAPI


def get_canonical_app() -> FastAPI:
    """Return the single canonical FastAPI application instance."""
    return get_app("api")


def serve(
    *,
    host: str,
    port: int,
    log_level: str = "info",
    reload: bool = False,
    workers: int | None = None,
    timeout_keep_alive: int | None = None,
    **_: object,
) -> int:
    """Start the canonical HTTP API server (delegates to mlsdm.serve)."""
    run_server(
        mode="api",
        host=host,
        port=port,
        log_level=log_level,
        reload=reload,
        workers=workers,
        timeout_keep_alive=timeout_keep_alive,
        config=None,
        backend=None,
        disable_rate_limit=False,
    )
    return 0


__all__ = ["serve", "get_canonical_app"]
