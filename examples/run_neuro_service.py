#!/usr/bin/env python3
"""
Example wrapper that delegates to the canonical CLI via subprocess.

Canonical start: `mlsdm serve`

Usage:
    # Using local stub backend (default, no API key needed)
    python examples/run_neuro_service.py

    # Using OpenAI backend
    export OPENAI_API_KEY="sk-..."
    export LLM_BACKEND="openai"
    python examples/run_neuro_service.py

    # Custom host and port
    export HOST="127.0.0.1"
    export PORT="8080"
    python examples/run_neuro_service.py

    # Disable FSLGS governance
    export ENABLE_FSLGS="false"
    python examples/run_neuro_service.py
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    """Run the MLSDM server via subprocess with env-based configuration."""
    # Read configuration from environment with defaults
    host = os.environ.get("HOST", "0.0.0.0")
    port = os.environ.get("PORT", "8000")
    config_path = os.environ.get("CONFIG_PATH", "config/default_config.yaml")

    print("🧠 MLSDM NeuroCognitiveEngine (example wrapper)")
    print("Canonical: mlsdm serve")
    print(f"Backend: {os.environ.get('LLM_BACKEND', 'local_stub')}")
    print(f"Config: {config_path}")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print()

    # Use a constant argv; pass host/port/config via environment variables.
    # This avoids passing environment-derived values as command-line arguments,
    # which prevents Semgrep dangerous-subprocess-use-tainted-env-args finding.
    cmd = [sys.executable, "-m", "mlsdm.cli", "serve"]

    # Build subprocess environment with explicit overrides
    env = {
        **os.environ,
        "HOST": host,
        "PORT": port,
        "CONFIG_PATH": config_path,
    }

    # Use check=False and return the exit code to propagate errors correctly
    result = subprocess.run(cmd, env=env, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
