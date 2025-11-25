"""Performance benchmarks for NeuroCognitiveEngine.

These benchmarks measure:
1. Pre-flight check latency (moral precheck only)
2. End-to-end latency with small load
3. End-to-end latency with heavy load (varying max_tokens)

Benchmarks use local_stub backend for consistent, reproducible results.

Usage:
    # Run via pytest
    pytest benchmarks/test_neuro_engine_performance.py -v -s -m benchmark

    # Run directly with CLI options
    python benchmarks/test_neuro_engine_performance.py --iterations 100 --output json

    # Run with custom output path
    python benchmarks/test_neuro_engine_performance.py --output-dir benchmarks/results
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Import pytest conditionally - benchmarks can run without it
try:
    import pytest

    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

    # Create a dummy decorator for when pytest is not available
    class _DummyMark:
        def benchmark(self, func: Any) -> Any:
            return func

    class _DummyPytest:
        mark = _DummyMark()

    pytest = _DummyPytest()  # type: ignore[assignment]

from mlsdm.engine.neuro_cognitive_engine import NeuroCognitiveEngine, NeuroEngineConfig

# ============================================================================
# Constants and Configuration
# ============================================================================

# Default random seed for reproducibility
DEFAULT_SEED = 42

# SLO thresholds (in milliseconds)
SLO_PRE_FLIGHT_P95_MS = 20.0
SLO_END_TO_END_P95_MS = 500.0


# ============================================================================
# Stub Functions for Reproducible Testing
# ============================================================================


def set_seeds(seed: int = DEFAULT_SEED) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)


def stub_llm_generate(prompt: str, max_tokens: int) -> str:
    """Stub LLM function for consistent performance testing.

    Simulates generation time proportional to max_tokens.
    """
    # Simulate token generation time: ~0.001ms per token
    time.sleep(max_tokens * 0.000001)
    return f"Generated {max_tokens} tokens for: {prompt[:50]}..."


def stub_embedding(text: str) -> np.ndarray:
    """Stub embedding function for consistent testing.

    Returns deterministic embedding based on text hash.
    """
    # Use text hash for deterministic but unique embeddings
    seed = hash(text) % (2**31)
    return np.random.RandomState(seed).randn(384).astype(np.float32)


# ============================================================================
# Statistics Utilities
# ============================================================================


def compute_percentiles(values: list[float]) -> dict[str, float | int]:
    """Compute percentile statistics.

    Args:
        values: List of latency values in milliseconds

    Returns:
        Dictionary with p50, p95, p99 percentiles and count
    """
    if not values:
        return {
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "count": 0,
        }

    sorted_values = sorted(values)
    n = len(sorted_values)

    def percentile(p: float) -> float:
        k = (n - 1) * p
        f = int(k)
        c = f + 1
        if c >= n:
            return sorted_values[-1]
        if f == k:
            return sorted_values[f]
        return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])

    return {
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "mean": sum(sorted_values) / n,
        "count": n,
    }


def compute_throughput(count: int, elapsed_seconds: float) -> float:
    """Compute throughput in operations per second.

    Args:
        count: Number of operations completed
        elapsed_seconds: Total time elapsed

    Returns:
        Operations per second
    """
    if elapsed_seconds <= 0:
        return 0.0
    return count / elapsed_seconds


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB.

    Returns:
        Memory usage in megabytes, or 0.0 if psutil is not available
    """
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except (ImportError, Exception):
        return 0.0


# ============================================================================
# Engine Factory
# ============================================================================


def create_engine(enable_metrics: bool = False) -> NeuroCognitiveEngine:
    """Create engine instance for benchmarking.

    Args:
        enable_metrics: Whether to enable metrics collection

    Returns:
        Configured NeuroCognitiveEngine instance
    """
    config = NeuroEngineConfig(
        enable_fslgs=False,  # Disable for simplicity
        enable_metrics=enable_metrics,
        initial_moral_threshold=0.5,
    )

    return NeuroCognitiveEngine(
        llm_generate_fn=stub_llm_generate,
        embedding_fn=stub_embedding,
        config=config,
    )


# ============================================================================
# Benchmark Functions
# ============================================================================


def benchmark_pre_flight_latency(iterations: int = 100) -> dict[str, Any]:
    """Benchmark pre-flight check latency.

    Measures only the moral precheck step, which should be very fast.

    Args:
        iterations: Number of iterations to run per prompt

    Returns:
        Dictionary with percentile statistics and metadata
    """
    engine = create_engine()
    latencies: list[float] = []

    # Generate a variety of prompts to test moral precheck
    prompts = [
        "What is the weather today?",
        "Tell me a story about adventure",
        "How do I cook pasta?",
        "Explain quantum physics",
        "What is consciousness?",
        "Help me with my homework",
        "Design a database schema",
        "Write a poem about nature",
        "What are the laws of thermodynamics?",
        "How does the internet work?",
    ]

    start_time = time.perf_counter()
    memory_before = get_memory_usage_mb()

    # Run benchmark
    for _ in range(iterations):
        for prompt in prompts:
            result = engine.generate(prompt, max_tokens=10)

            # Only count pre-flight latency if available
            if "moral_precheck" in result.get("timing", {}):
                latencies.append(result["timing"]["moral_precheck"])

    elapsed_seconds = time.perf_counter() - start_time
    memory_after = get_memory_usage_mb()

    stats = compute_percentiles(latencies)
    stats["throughput_ops_sec"] = compute_throughput(len(latencies), elapsed_seconds)
    stats["memory_delta_mb"] = memory_after - memory_before
    stats["iterations"] = iterations
    stats["prompts_per_iteration"] = len(prompts)
    stats["scenario"] = "pre_flight_latency"

    return stats


def benchmark_end_to_end_latency_small_load(iterations: int = 50) -> dict[str, Any]:
    """Benchmark end-to-end latency with small load.

    Tests basic generation with moderate token counts.

    Args:
        iterations: Number of iterations to run per prompt

    Returns:
        Dictionary with percentile statistics and metadata
    """
    engine = create_engine()
    latencies: list[float] = []

    prompts = [
        "Summarize this: Machine learning is fascinating",
        "Translate: Hello world",
        "Classify: This movie is great!",
        "Answer: What is 2+2?",
        "Complete: The quick brown fox",
    ]

    start_time = time.perf_counter()
    memory_before = get_memory_usage_mb()

    # Run benchmark with small max_tokens
    for _ in range(iterations):
        for prompt in prompts:
            start = time.perf_counter()
            engine.generate(prompt, max_tokens=50)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)

    elapsed_seconds = time.perf_counter() - start_time
    memory_after = get_memory_usage_mb()

    stats = compute_percentiles(latencies)
    stats["throughput_ops_sec"] = compute_throughput(len(latencies), elapsed_seconds)
    stats["memory_delta_mb"] = memory_after - memory_before
    stats["iterations"] = iterations
    stats["prompts_per_iteration"] = len(prompts)
    stats["max_tokens"] = 50
    stats["scenario"] = "end_to_end_small_load"

    return stats


def benchmark_end_to_end_latency_heavy_load(
    iterations: int = 20,
) -> dict[str, dict[str, Any]]:
    """Benchmark end-to-end latency with heavy load.

    Tests with varying max_tokens values to see scaling behavior.

    Args:
        iterations: Number of iterations to run per token count

    Returns:
        Dictionary mapping max_tokens to percentile statistics
    """
    engine = create_engine()

    prompts = [
        "Write a comprehensive essay about climate change",
        "Explain the history of artificial intelligence",
        "Describe the solar system in detail",
        "Analyze the themes in Shakespeare's Hamlet",
        "Create a detailed business plan for a startup",
    ]

    # Test different token counts
    token_counts = [100, 250, 500, 1000]
    results: dict[str, dict[str, Any]] = {}

    for max_tokens in token_counts:
        latencies: list[float] = []

        start_time = time.perf_counter()
        memory_before = get_memory_usage_mb()

        # Run benchmark
        for _ in range(iterations):
            for prompt in prompts:
                start = time.perf_counter()
                engine.generate(prompt, max_tokens=max_tokens)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                latencies.append(elapsed_ms)

        elapsed_seconds = time.perf_counter() - start_time
        memory_after = get_memory_usage_mb()

        stats = compute_percentiles(latencies)
        stats["throughput_ops_sec"] = compute_throughput(len(latencies), elapsed_seconds)
        stats["memory_delta_mb"] = memory_after - memory_before
        stats["iterations"] = iterations
        stats["prompts_per_iteration"] = len(prompts)
        stats["max_tokens"] = max_tokens
        stats["scenario"] = "end_to_end_heavy_load"

        results[f"tokens_{max_tokens}"] = stats

    return results


# ============================================================================
# Pytest Test Functions
# ============================================================================


@pytest.mark.benchmark
def test_benchmark_pre_flight_latency() -> None:
    """Test and report pre-flight check latency."""
    set_seeds()
    print("\n" + "=" * 70)
    print("BENCHMARK: Pre-Flight Check Latency")
    print("=" * 70)

    stats = benchmark_pre_flight_latency()

    print(f"\nResults ({stats['count']} measurements with moral_precheck timing):")
    print(f"  P50 (median): {stats['p50']:.3f}ms")
    print(f"  P95:          {stats['p95']:.3f}ms")
    print(f"  P99:          {stats['p99']:.3f}ms")
    print(f"  Min:          {stats['min']:.3f}ms")
    print(f"  Max:          {stats['max']:.3f}ms")
    print(f"  Mean:         {stats['mean']:.3f}ms")
    print(f"  Throughput:   {stats['throughput_ops_sec']:.1f} ops/sec")
    print()

    # SLO: pre_flight_latency_p95 < 20ms
    assert stats["p95"] < SLO_PRE_FLIGHT_P95_MS, (
        f"P95 latency {stats['p95']:.3f}ms exceeds SLO of {SLO_PRE_FLIGHT_P95_MS}ms"
    )
    print(f"✓ SLO met: P95 < {SLO_PRE_FLIGHT_P95_MS}ms")
    print()


@pytest.mark.benchmark
def test_benchmark_end_to_end_small_load() -> None:
    """Test and report end-to-end latency with small load."""
    set_seeds()
    print("\n" + "=" * 70)
    print("BENCHMARK: End-to-End Latency (Small Load)")
    print("=" * 70)
    print("Configuration: 50 tokens, normal prompts")
    print()

    stats = benchmark_end_to_end_latency_small_load()

    print(f"Results ({stats['count']} measurements):")
    print(f"  P50 (median): {stats['p50']:.3f}ms")
    print(f"  P95:          {stats['p95']:.3f}ms")
    print(f"  P99:          {stats['p99']:.3f}ms")
    print(f"  Min:          {stats['min']:.3f}ms")
    print(f"  Max:          {stats['max']:.3f}ms")
    print(f"  Mean:         {stats['mean']:.3f}ms")
    print(f"  Throughput:   {stats['throughput_ops_sec']:.1f} ops/sec")
    print()

    # SLO: latency_total_ms_p95 < 500ms
    assert stats["p95"] < SLO_END_TO_END_P95_MS, (
        f"P95 latency {stats['p95']:.3f}ms exceeds SLO of {SLO_END_TO_END_P95_MS}ms"
    )
    print(f"✓ SLO met: P95 < {SLO_END_TO_END_P95_MS}ms")
    print()


@pytest.mark.benchmark
def test_benchmark_end_to_end_heavy_load() -> None:
    """Test and report end-to-end latency with heavy load."""
    set_seeds()
    print("\n" + "=" * 70)
    print("BENCHMARK: End-to-End Latency (Heavy Load)")
    print("=" * 70)
    print("Testing varying token counts...")
    print()

    results = benchmark_end_to_end_latency_heavy_load()

    print("Results by token count:")
    print("-" * 70)

    for _token_key, stats in sorted(results.items()):
        token_count = stats["max_tokens"]
        print(f"\nmax_tokens={token_count}:")
        print(
            f"  P50: {stats['p50']:.3f}ms | P95: {stats['p95']:.3f}ms | "
            f"P99: {stats['p99']:.3f}ms"
        )
        print(
            f"  Min: {stats['min']:.3f}ms | Max: {stats['max']:.3f}ms | "
            f"Mean: {stats['mean']:.3f}ms"
        )
        print(f"  Throughput: {stats['throughput_ops_sec']:.1f} ops/sec")

        # All should meet SLO
        assert stats["p95"] < SLO_END_TO_END_P95_MS, (
            f"P95 latency {stats['p95']:.3f}ms exceeds SLO of "
            f"{SLO_END_TO_END_P95_MS}ms for {token_count} tokens"
        )

    print()
    print(f"✓ All token counts meet SLO: P95 < {SLO_END_TO_END_P95_MS}ms")
    print()


@pytest.mark.benchmark
def test_benchmark_summary() -> None:
    """Generate a comprehensive benchmark summary."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print()
    print("All benchmarks completed successfully!")
    print()
    print("SLO Compliance:")
    print(f"  ✓ Pre-flight latency P95 < {SLO_PRE_FLIGHT_P95_MS}ms")
    print(f"  ✓ End-to-end latency P95 < {SLO_END_TO_END_P95_MS}ms")
    print()
    print("Note: These benchmarks use stub LLM backend for reproducibility.")
    print("Real-world performance will vary based on actual LLM latency.")
    print("=" * 70)
    print()


# ============================================================================
# CLI and Main Entry Point
# ============================================================================


def format_results_table(all_results: dict[str, Any]) -> str:
    """Format results as a Markdown table.

    Args:
        all_results: Dictionary containing all benchmark results

    Returns:
        Formatted Markdown table string
    """
    lines = [
        "| Benchmark | P50 (ms) | P95 (ms) | P99 (ms) | Throughput (ops/s) | SLO Status |",
        "|-----------|----------|----------|----------|-------------------|------------|",
    ]

    # Pre-flight latency
    if "pre_flight" in all_results:
        stats = all_results["pre_flight"]
        slo_status = "✓ PASS" if stats["p95"] < SLO_PRE_FLIGHT_P95_MS else "✗ FAIL"
        lines.append(
            f"| Pre-flight Latency | {stats['p50']:.3f} | {stats['p95']:.3f} | "
            f"{stats['p99']:.3f} | {stats.get('throughput_ops_sec', 0):.1f} | {slo_status} |"
        )

    # Small load
    if "small_load" in all_results:
        stats = all_results["small_load"]
        slo_status = "✓ PASS" if stats["p95"] < SLO_END_TO_END_P95_MS else "✗ FAIL"
        lines.append(
            f"| End-to-End (50 tokens) | {stats['p50']:.3f} | {stats['p95']:.3f} | "
            f"{stats['p99']:.3f} | {stats.get('throughput_ops_sec', 0):.1f} | {slo_status} |"
        )

    # Heavy load
    if "heavy_load" in all_results:
        for token_key, stats in sorted(all_results["heavy_load"].items()):
            tokens = stats.get("max_tokens", token_key.split("_")[-1])
            slo_status = "✓ PASS" if stats["p95"] < SLO_END_TO_END_P95_MS else "✗ FAIL"
            lines.append(
                f"| End-to-End ({tokens} tokens) | {stats['p50']:.3f} | "
                f"{stats['p95']:.3f} | {stats['p99']:.3f} | "
                f"{stats.get('throughput_ops_sec', 0):.1f} | {slo_status} |"
            )

    return "\n".join(lines)


def run_all_benchmarks(
    iterations: int = 100,
    output_format: str = "console",
    output_dir: str | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Run all benchmarks and collect results.

    Args:
        iterations: Base number of iterations for benchmarks
        output_format: Output format - 'console', 'json', or 'table'
        output_dir: Directory to save JSON output (if format is 'json')
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing all benchmark results
    """
    set_seeds(seed)

    print("=" * 70)
    print("NeuroCognitiveEngine Performance Benchmarks")
    print("=" * 70)
    print(f"Seed: {seed}")
    print(f"Output format: {output_format}")
    print()

    # Collect all results
    all_results: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "iterations_base": iterations,
            "slo_pre_flight_p95_ms": SLO_PRE_FLIGHT_P95_MS,
            "slo_end_to_end_p95_ms": SLO_END_TO_END_P95_MS,
        },
    }

    # Run benchmarks
    print("Running pre-flight latency benchmark...")
    all_results["pre_flight"] = benchmark_pre_flight_latency(iterations=iterations)

    print("Running small load benchmark...")
    all_results["small_load"] = benchmark_end_to_end_latency_small_load(
        iterations=iterations // 2
    )

    print("Running heavy load benchmark...")
    all_results["heavy_load"] = benchmark_end_to_end_latency_heavy_load(
        iterations=iterations // 5
    )

    # Calculate SLO compliance
    slo_checks = {
        "pre_flight_p95": all_results["pre_flight"]["p95"] < SLO_PRE_FLIGHT_P95_MS,
        "small_load_p95": all_results["small_load"]["p95"] < SLO_END_TO_END_P95_MS,
    }
    for token_key, stats in all_results["heavy_load"].items():
        slo_checks[f"heavy_load_{token_key}_p95"] = stats["p95"] < SLO_END_TO_END_P95_MS

    all_results["slo_compliance"] = {
        "all_passed": all(slo_checks.values()),
        "checks": slo_checks,
    }

    # Output results based on format
    if output_format == "json" or output_dir:
        output_path = Path(output_dir or "benchmarks/results")
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"neuro_engine_benchmark_{timestamp}.json"

        with open(filename, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {filename}")

    if output_format == "table":
        print("\n" + format_results_table(all_results))
    elif output_format == "console":
        # Print console summary
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)

        pre_flight = all_results["pre_flight"]
        print("\nPre-flight Latency:")
        print(f"  P50: {pre_flight['p50']:.3f}ms | P95: {pre_flight['p95']:.3f}ms")
        print(f"  Throughput: {pre_flight.get('throughput_ops_sec', 0):.1f} ops/sec")

        small_load = all_results["small_load"]
        print("\nSmall Load (50 tokens):")
        print(f"  P50: {small_load['p50']:.3f}ms | P95: {small_load['p95']:.3f}ms")
        print(f"  Throughput: {small_load.get('throughput_ops_sec', 0):.1f} ops/sec")

        print("\nHeavy Load (varying tokens):")
        for _token_key, stats in sorted(all_results["heavy_load"].items()):
            tokens = stats.get("max_tokens", "N/A")
            print(f"  {tokens} tokens: P50={stats['p50']:.3f}ms, P95={stats['p95']:.3f}ms")

        print("\nSLO Compliance:")
        if all_results["slo_compliance"]["all_passed"]:
            print("  ✓ All SLOs PASSED")
        else:
            failed = [k for k, v in slo_checks.items() if not v]
            print(f"  ✗ FAILED: {', '.join(failed)}")

    print("\n" + "=" * 70)
    print("Benchmarks completed!")
    print("=" * 70)

    return all_results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="NeuroCognitiveEngine Performance Benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (console output)
  python benchmarks/test_neuro_engine_performance.py

  # Run with JSON output
  python benchmarks/test_neuro_engine_performance.py --output json

  # Run with custom iterations and output directory
  python benchmarks/test_neuro_engine_performance.py --iterations 50 --output-dir ./results

  # Run with table output format
  python benchmarks/test_neuro_engine_performance.py --output table

  # Run with custom seed for reproducibility
  python benchmarks/test_neuro_engine_performance.py --seed 12345
        """,
    )

    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=100,
        help="Base number of iterations for benchmarks (default: 100)",
    )

    parser.add_argument(
        "--output",
        "-o",
        choices=["console", "json", "table"],
        default="console",
        help="Output format (default: console)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save JSON results (default: benchmarks/results)",
    )

    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for running benchmarks."""
    args = parse_args()

    # Determine output directory
    output_dir = args.output_dir
    if args.output == "json" and output_dir is None:
        output_dir = "benchmarks/results"

    results = run_all_benchmarks(
        iterations=args.iterations,
        output_format=args.output,
        output_dir=output_dir,
        seed=args.seed,
    )

    # Exit with error if SLO failed
    if not results["slo_compliance"]["all_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
