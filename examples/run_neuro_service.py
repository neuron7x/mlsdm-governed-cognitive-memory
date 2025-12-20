#!/usr/bin/env python3
"""
Run NeuroCognitiveEngine HTTP API Service.

This script starts the FastAPI HTTP server for the NeuroCognitiveEngine.

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

import os
import sys

if __name__ == "__main__":
    # Add src to path only when running as script
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from mlsdm.serve import run_server

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    log_level = os.environ.get("LOG_LEVEL", "info")
    reload = os.environ.get("RELOAD", "").lower() == "true"
    backend = os.environ.get("LLM_BACKEND")
    config_path = os.environ.get("CONFIG_PATH")
    disable_rate_limit = os.environ.get("DISABLE_RATE_LIMIT") == "1"

    print("🚀 Starting NeuroCognitiveEngine HTTP API Service...")
    print(f"   Backend: {backend or 'local_stub'}")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   FSLGS: {os.environ.get('ENABLE_FSLGS', 'true')}")
    print(f"   Metrics: {os.environ.get('ENABLE_METRICS', 'true')}")
    print()
    print("API endpoints:")
    print("  - POST http://localhost:8000/v1/neuro/generate")
    print("  - GET  http://localhost:8000/healthz")
    print("  - GET  http://localhost:8000/metrics")
    print("  - GET  http://localhost:8000/docs (Swagger UI)")
    print()

    run_server(
        mode="neuro",
        host=host,
        port=port,
        log_level=log_level,
        reload=reload,
        config=config_path,
        backend=backend,
        disable_rate_limit=disable_rate_limit,
    )
