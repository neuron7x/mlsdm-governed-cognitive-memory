#!/usr/bin/env python3
"""
Example wrapper that delegates to the canonical CLI.

Canonical start: `mlsdm serve`
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
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

    cmd = [
        sys.executable,
        "-m",
        "mlsdm.cli",
        "serve",
        "--host",
        host,
        "--port",
        str(port),
        "--config",
        config_path,
    ]
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
