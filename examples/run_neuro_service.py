#!/usr/bin/env python3
"""
Example launcher for the NeuroCognitiveEngine HTTP API.

Canonical start is `mlsdm serve`. This example delegates to that CLI.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import List


@contextmanager
def _argv(argv: List[str]):
    original = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = original


def main() -> int:
    print("⚠️  Example launcher: canonical start is `mlsdm serve`.")
    host = os.environ.get("HOST", "0.0.0.0")
    port = os.environ.get("PORT", "8000")
    backend = os.environ.get("LLM_BACKEND")
    config_path = os.environ.get("CONFIG_PATH", "config/default_config.yaml")

    argv = ["mlsdm", "serve", "--host", host, "--port", str(port), "--config", config_path]
    if backend:
        argv.extend(["--backend", backend])
    if os.environ.get("DISABLE_RATE_LIMIT") in ("1", "true", "yes", "on"):
        argv.append("--disable-rate-limit")

    print(f"Delegating to CLI: {' '.join(argv[1:])}")

    # Add src to path only when running as script
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from mlsdm.cli import main as cli_main

    with _argv(argv):
        return cli_main()


if __name__ == "__main__":
    sys.exit(main())
