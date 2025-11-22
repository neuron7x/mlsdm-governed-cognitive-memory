"""
NeuroCognitiveEngine HTTP API Service.

This module provides a FastAPI-based HTTP API for the NeuroCognitiveEngine,
including health checks, metrics endpoint, and generation endpoint.
"""

from __future__ import annotations

import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from mlsdm.engine import NeuroEngineConfig, build_neuro_engine_from_env
from mlsdm.observability.metrics import MetricsRegistry
from mlsdm.security import get_rate_limiter

# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """Request model for generate endpoint."""

    prompt: str = Field(..., min_length=1, description="Input text prompt to process")
    max_tokens: int | None = Field(
        None, ge=1, le=4096, description="Maximum number of tokens to generate"
    )
    moral_value: float | None = Field(
        None, ge=0.0, le=1.0, description="Moral threshold value"
    )
    user_intent: str | None = Field(
        None, description="User intent category (e.g., 'conversational', 'analytical')"
    )
    cognitive_load: float | None = Field(
        None, ge=0.0, le=1.0, description="Cognitive load value"
    )
    context_top_k: int | None = Field(
        None, ge=1, le=100, description="Number of top context items to retrieve"
    )


class GenerateResponse(BaseModel):
    """Response model for generate endpoint."""

    response: str = Field(description="Generated response text")
    governance: dict[str, Any] | None = Field(
        default_factory=dict, description="Governance state information"
    )
    mlsdm: dict[str, Any] | None = Field(
        default_factory=dict, description="MLSDM internal state"
    )
    timing: dict[str, float] = Field(
        default_factory=dict, description="Performance timing metrics"
    )
    validation_steps: list[dict[str, Any]] = Field(
        default_factory=list, description="Validation steps executed"
    )
    error: dict[str, Any] | None = Field(
        None, description="Error information if generation failed"
    )
    rejected_at: str | None = Field(
        None, description="Stage at which request was rejected"
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service status", examples=["ok", "degraded"])
    version: str | None = Field(None, description="Service version")
    backend: str = Field(description="Current LLM backend")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="NeuroCognitiveEngine API",
        description="HTTP API for NeuroCognitiveEngine with moral governance and FSLGS",
        version="0.1.0",
    )

    # Initialize engine at startup
    backend = os.environ.get("LLM_BACKEND", "local_stub")
    config_dim = int(os.environ.get("EMBEDDING_DIM", "384"))
    enable_fslgs = os.environ.get("ENABLE_FSLGS", "true").lower() == "true"
    enable_metrics = os.environ.get("ENABLE_METRICS", "true").lower() == "true"

    config = NeuroEngineConfig(
        dim=config_dim,
        enable_fslgs=enable_fslgs,
        enable_metrics=enable_metrics,
    )

    engine = build_neuro_engine_from_env(config=config)

    # Initialize metrics registry if enabled
    metrics_registry = None
    if enable_metrics:
        metrics_registry = MetricsRegistry()
    # Initialize rate limiter
    rate_limiter_requests = int(os.environ.get("RATE_LIMIT_REQUESTS", "100"))
    rate_limiter_window = int(os.environ.get("RATE_LIMIT_WINDOW", "60"))
    rate_limiter = get_rate_limiter(
        requests_per_window=rate_limiter_requests,
        window_seconds=rate_limiter_window,
    )

    # Store in app state
    app.state.engine = engine
    app.state.backend = backend
    app.state.metrics = metrics_registry
    app.state.rate_limiter = rate_limiter

    @app.post(
        "/v1/neuro/generate",
        response_model=GenerateResponse,
        status_code=status.HTTP_200_OK,
        tags=["Generation"],
    )
    async def generate(request_body: GenerateRequest, request: Request) -> dict[str, Any]:
        """Generate a response using the NeuroCognitiveEngine.

        This endpoint processes the input prompt through the complete cognitive pipeline,
        including moral filtering, memory retrieval, rhythm management, and optional
        FSLGS governance.

        Args:
            request_body: Generation request parameters.
            request: FastAPI request object.

        Returns:
            Generated response with governance information and timing metrics.

        Raises:
            HTTPException: If generation fails due to invalid input or internal error.
        """
        start_time = time.time()

        # Rate limiting check
        client_ip = request.client.host if request.client else "unknown"
        if request.app.state.rate_limiter and not request.app.state.rate_limiter.is_allowed(client_ip):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
            )

        try:
            # Build kwargs for engine
            kwargs: dict[str, Any] = {"prompt": request_body.prompt}
            if request_body.max_tokens is not None:
                kwargs["max_tokens"] = request_body.max_tokens
            if request_body.moral_value is not None:
                kwargs["moral_value"] = request_body.moral_value
            if request_body.user_intent is not None:
                kwargs["user_intent"] = request_body.user_intent
            if request_body.cognitive_load is not None:
                kwargs["cognitive_load"] = request_body.cognitive_load
            if request_body.context_top_k is not None:
                kwargs["context_top_k"] = request_body.context_top_k

            # Generate response
            result = request.app.state.engine.generate(**kwargs)

            # Record metrics if enabled
            if request.app.state.metrics:
                elapsed_ms = (time.time() - start_time) * 1000
                request.app.state.metrics.increment_requests_total()
                request.app.state.metrics.record_latency_total(elapsed_ms)

                if result.get("error") is not None:
                    error_type = result["error"].get("type", "unknown")
                    request.app.state.metrics.increment_errors_total(error_type)

                if result.get("rejected_at") is not None:
                    request.app.state.metrics.increment_rejections_total(result["rejected_at"])

            return result

        except Exception as e:
            # Record error metric
            if request.app.state.metrics:
                elapsed_ms = (time.time() - start_time) * 1000
                request.app.state.metrics.increment_requests_total()
                request.app.state.metrics.record_latency_total(elapsed_ms)
                request.app.state.metrics.increment_errors_total("internal_error")

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal error: {str(e)}",
            ) from e

    @app.get(
        "/healthz",
        response_model=HealthResponse,
        status_code=status.HTTP_200_OK,
        tags=["Health"],
    )
    async def healthz(request: Request) -> HealthResponse:
        """Health check endpoint.

        Returns the current status of the service, including backend information.

        Returns:
            Health status information.
        """
        return HealthResponse(
            status="ok",
            version="0.1.0",
            backend=request.app.state.backend,
        )

    @app.get(
        "/metrics",
        response_class=PlainTextResponse,
        status_code=status.HTTP_200_OK,
        tags=["Observability"],
    )
    async def metrics(request: Request) -> str:
        """Prometheus-compatible metrics endpoint.

        Returns metrics in Prometheus text format, including request counts,
        latencies, and system metrics.

        Returns:
            Prometheus-formatted metrics as plain text.
        """
        if request.app.state.metrics is None:
            return "# Metrics disabled\n"

        # Get metrics summary
        summary = request.app.state.metrics.get_summary()

        # Format as Prometheus-compatible text
        lines = []
        lines.append("# HELP neuro_requests_total Total number of requests")
        lines.append("# TYPE neuro_requests_total counter")
        lines.append(f"neuro_requests_total {summary['requests_total']}")
        lines.append("")

        lines.append("# HELP neuro_rejections_total Total number of rejections by stage")
        lines.append("# TYPE neuro_rejections_total counter")
        for stage, count in summary['rejections_total'].items():
            lines.append(f'neuro_rejections_total{{stage="{stage}"}} {count}')
        lines.append("")

        lines.append("# HELP neuro_errors_total Total number of errors by type")
        lines.append("# TYPE neuro_errors_total counter")
        for error_type, count in summary['errors_total'].items():
            lines.append(f'neuro_errors_total{{type="{error_type}"}} {count}')
        lines.append("")

        # Latency metrics
        latency_stats = summary['latency_stats']
        for latency_type, stats in latency_stats.items():
            if stats['count'] > 0:
                metric_name = f"neuro_latency_{latency_type}"
                lines.append(f"# HELP {metric_name} Latency in milliseconds")
                lines.append(f"# TYPE {metric_name} histogram")
                lines.append(f'{metric_name}{{quantile="0.5"}} {stats["p50"]:.2f}')
                lines.append(f'{metric_name}{{quantile="0.95"}} {stats["p95"]:.2f}')
                lines.append(f'{metric_name}{{quantile="0.99"}} {stats["p99"]:.2f}')
                lines.append(f'{metric_name}_sum {stats["mean"] * stats["count"]:.2f}')
                lines.append(f'{metric_name}_count {stats["count"]}')
                lines.append("")

        return "\n".join(lines)

    return app


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for running the service.

    Reads configuration from environment variables and starts the uvicorn server.

    Environment Variables:
        HOST: Server host (default: "0.0.0.0")
        PORT: Server port (default: 8000)
        LLM_BACKEND: Backend to use ("openai" or "local_stub", default: "local_stub")
        OPENAI_API_KEY: Required when LLM_BACKEND="openai"
        EMBEDDING_DIM: Embedding dimensionality (default: 384)
        ENABLE_FSLGS: Enable FSLGS governance (default: "true")
        ENABLE_METRICS: Enable metrics collection (default: "true")
    """
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))

    app = create_app()

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
