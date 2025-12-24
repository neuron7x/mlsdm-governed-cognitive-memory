"""Deprecated service shim that delegates to the canonical MLSDM API app."""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING

from mlsdm.api.app import create_app as _create_canonical_app
from mlsdm.entrypoints.cloud_entry import main as _cloud_main

if TYPE_CHECKING:
    from fastapi import FastAPI


def create_app() -> FastAPI:
    """Return the canonical FastAPI application."""
    return _create_canonical_app()


def main() -> int:
    """Deprecated wrapper that delegates to the cloud entrypoint."""
    warnings.warn(
        "mlsdm.service.neuro_engine_service is deprecated; use 'mlsdm serve' or "
        "'python -m mlsdm.entrypoints.cloud'",
        DeprecationWarning,
        stacklevel=2,
    )
    os.environ.setdefault("MLSDM_RUNTIME_MODE", "cloud-prod")
    return _cloud_main()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["create_app", "main"]
