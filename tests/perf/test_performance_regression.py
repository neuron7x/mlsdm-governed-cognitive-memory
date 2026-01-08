"""Performance regression test suite."""
from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlsdm.api.app import app


@pytest.mark.benchmark
class TestPerformanceRegression:
    """Ensure no performance regressions."""

    def test_p50_latency_under_30ms(self, benchmark) -> None:
        """P50 latency must be under 30ms."""
        client = TestClient(app)

        def make_request():
            return client.post(
                "/generate",
                json={
                    "prompt": "Test",
                    "max_tokens": 50,
                    "moral_value": 0.8,
                },
            )

        benchmark(make_request)
        mean_latency = float(benchmark.stats["mean"])
        assert mean_latency < 0.030, f"P50 latency {mean_latency * 1000:.2f}ms exceeds 30ms"

    def test_p95_latency_under_120ms(self) -> None:
        """P95 latency must be under 120ms (production target)."""
        client = TestClient(app)
        latencies: list[float] = []

        for _ in range(100):
            start = time.perf_counter()
            client.post("/generate", json={"prompt": "Test", "max_tokens": 50})
            latencies.append((time.perf_counter() - start) * 1000)

        p95 = float(np.percentile(latencies, 95))
        assert p95 < 120.0, f"P95 latency {p95:.2f}ms exceeds 120ms"

    def test_p99_latency_under_200ms(self) -> None:
        """P99 latency must be under 200ms."""
        client = TestClient(app)
        latencies: list[float] = []

        for _ in range(100):
            start = time.perf_counter()
            client.post("/generate", json={"prompt": "Test", "max_tokens": 50})
            latencies.append((time.perf_counter() - start) * 1000)

        p99 = float(np.percentile(latencies, 99))
        assert p99 < 200.0, f"P99 latency {p99:.2f}ms exceeds 200ms"

    def test_throughput_over_100_rps(self) -> None:
        """System must handle 100+ requests per second."""
        client = TestClient(app)

        start = time.perf_counter()
        for _ in range(100):
            client.post("/generate", json={"prompt": "Test", "max_tokens": 50})
        duration = time.perf_counter() - start

        rps = 100 / duration if duration else 0.0
        assert rps >= 100, f"Throughput {rps:.2f} RPS below 100 RPS"

    def test_memory_stable_under_load(self) -> None:
        """Memory usage must be stable under sustained load."""
        import gc

        import psutil

        process = psutil.Process()
        client = TestClient(app)

        # Baseline
        gc.collect()
        baseline_mb = process.memory_info().rss / 1024 / 1024

        # Load test
        for _ in range(1000):
            client.post("/generate", json={"prompt": "Test", "max_tokens": 50})

        # Check memory
        gc.collect()
        after_mb = process.memory_info().rss / 1024 / 1024
        growth_mb = after_mb - baseline_mb

        assert growth_mb < 50, f"Memory grew {growth_mb:.2f}MB (max 50MB allowed)"
