#!/usr/bin/env python3
"""
Run NeuroCognitiveEngine HTTP API Service safely.

This script builds a safe launcher for the MLSDM service without mutating
sys.path. It delegates to the packaged CLI or uvicorn module.

Environment (fallback) variables:
- HOST or MLSDM_HOST (default: 0.0.0.0)
- PORT or MLSDM_PORT (default: 8000, must be 1-65535)

Usage:
    python examples/run_neuro_service.py --dry-run
    python examples/run_neuro_service.py --host 127.0.0.1 --port 8080 --reload
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


def _valid_port(value: str | int) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"Invalid port '{value}': must be an integer between 1 and 65535"
        ) from exc

    if not 1 <= port <= 65535:
        raise ValueError(f"Invalid port '{value}': must be between 1 and 65535")
    return port


def _resolve_host_port(args_host: str | None, args_port: int | None) -> tuple[str, int]:
    host = args_host or os.environ.get("MLSDM_HOST") or os.environ.get("HOST") or DEFAULT_HOST

    port_value = args_port if args_port is not None else os.environ.get("MLSDM_PORT") or os.environ.get("PORT")
    port = DEFAULT_PORT if port_value is None else _valid_port(port_value)

    return host, port


def _build_command(host: str, port: int, reload: bool) -> list[str]:
    command = [sys.executable, "-m", "mlsdm.cli", "serve", "--host", host, "--port", str(port)]
    if reload:
        command.append("--reload")
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the NeuroCognitiveEngine service safely.")
    parser.add_argument("--host", help="Host to bind the service. Falls back to MLSDM_HOST or HOST.", default=None)
    parser.add_argument(
        "--port",
        type=_valid_port,
        help="Port to bind the service (1-65535). Falls back to MLSDM_PORT or PORT.",
        default=None,
    )
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload (for development).")
    parser.add_argument("--dry-run", action="store_true", help="Print the command that would be executed and exit.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        host, port = _resolve_host_port(args.host, args.port)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    command = _build_command(host, port, args.reload)

    if args.dry_run:
        print("Dry-run: would execute:", " ".join(command))
        return 0

    print(f"🚀 Starting NeuroCognitiveEngine HTTP API Service on {host}:{port}...")
    print(f"   Backend: {os.environ.get('LLM_BACKEND', 'local_stub')}")
    print(f"   FSLGS: {os.environ.get('ENABLE_FSLGS', 'true')}")
    print(f"   Metrics: {os.environ.get('ENABLE_METRICS', 'true')}")
    print()
    print("API endpoints:")
    print(f"  - POST http://{host}:{port}/generate")
    print(f"  - POST http://{host}:{port}/infer")
    print(f"  - GET  http://{host}:{port}/health")
    print(f"  - GET  http://{host}:{port}/health/metrics")
    print(f"  - GET  http://{host}:{port}/docs (Swagger UI)")
    print()

    completed = subprocess.run(command, check=True)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
