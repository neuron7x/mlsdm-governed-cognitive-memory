"""
Property-based test for zero allocation after initialization.

Validates the claim that PELM memory does not grow after reaching capacity,
as it uses fixed pre-allocated buffers with eviction-based updates.
"""

import gc
from typing import Final

import numpy as np
import psutil
import pytest

from mlsdm.memory.phase_entangled_lattice_memory import PhaseEntangledLatticeMemory

# Constants for test configuration
_DIMENSION: Final[int] = 384
_FAST_TEST_CAPACITY: Final[int] = 1000
_PRODUCTION_CAPACITY: Final[int] = 20_000
_MEMORY_TOLERANCE_PERCENT: Final[float] = 5.0
_PHASE_VALUES: Final[int] = 11  # Number of distinct phase values to cycle through
_MB_DIVISOR: Final[float] = 1024.0 * 1024.0


def _stabilize_memory() -> None:
    """Force garbage collection to stabilize memory measurements."""
    gc.collect()
    gc.collect()  # Double collection for more stable baseline


def _measure_memory_mb() -> float:
    """Measure current process memory in MB with high precision."""
    process = psutil.Process()
    return process.memory_info().rss / _MB_DIVISOR


def _fill_pelm_to_capacity(
    pelm: PhaseEntangledLatticeMemory,
    capacity: int,
    dimension: int,
    verbose: bool = True,
) -> None:
    """
    Fill PELM to capacity with random vectors.

    Args:
        pelm: The PELM instance to fill
        capacity: Number of vectors to insert
        dimension: Vector dimension
        verbose: Whether to print progress
    """
    if verbose:
        print(f"\nFilling PELM to capacity ({capacity} vectors)...")

    # Use numpy for efficiency, convert to list for API
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    for i in range(capacity):
        vec = rng.standard_normal(dimension, dtype=np.float32).tolist()
        phase = float(i % _PHASE_VALUES) / float(_PHASE_VALUES - 1)
        pelm.entangle(vec, phase=phase)


def _validate_zero_growth(
    baseline_mb: float,
    after_mb: float,
    tolerance_percent: float,
    capacity: int,
) -> None:
    """
    Validate that memory growth is within acceptable tolerance.

    Args:
        baseline_mb: Baseline memory in MB
        after_mb: Memory after eviction in MB
        tolerance_percent: Maximum allowed growth percentage
        capacity: PELM capacity (for context in error messages)

    Raises:
        AssertionError: If memory growth exceeds tolerance
    """
    growth_mb = after_mb - baseline_mb
    growth_percent = (growth_mb / baseline_mb) * 100
    max_allowed_mb = baseline_mb * (1.0 + tolerance_percent / 100.0)

    print(f"Baseline memory at capacity: {baseline_mb:.2f} MB")
    print(f"Memory after eviction phase: {after_mb:.2f} MB")
    print(f"Memory growth: {growth_mb:.2f} MB ({growth_percent:.2f}%)")

    assert after_mb <= max_allowed_mb, (
        f"Memory grew from {baseline_mb:.2f} MB to {after_mb:.2f} MB "
        f"({growth_percent:.2f}% growth), exceeding {tolerance_percent}% tolerance. "
        f"Expected zero-growth behavior with eviction-based updates for "
        f"capacity={capacity} vectors."
    )

    print(f"✅ PASS: Memory growth {growth_percent:.2f}% is within {tolerance_percent}% tolerance")


def test_pelm_zero_growth_after_init():
    """
    Verify PELM memory does not grow significantly after reaching capacity.

    This test validates that once PELM reaches its configured capacity,
    additional insertions trigger eviction rather than memory growth,
    maintaining a fixed memory footprint.

    Methodology:
    1. Fill PELM to capacity with random vectors
    2. Stabilize memory with garbage collection
    3. Measure baseline memory
    4. Insert additional vectors (triggering eviction)
    5. Verify memory growth is within tolerance

    Tolerance: 5% growth allowed for:
    - Python garbage collection overhead
    - OS memory allocation granularity
    - Minor internal buffer adjustments
    - CI environment variability
    """
    # Create PELM with moderate capacity for faster testing
    pelm = PhaseEntangledLatticeMemory(
        dimension=_DIMENSION,
        capacity=_FAST_TEST_CAPACITY,
    )

    # Fill to capacity
    _fill_pelm_to_capacity(pelm, _FAST_TEST_CAPACITY, _DIMENSION)

    # Stabilize and measure baseline
    _stabilize_memory()
    baseline_mb = _measure_memory_mb()

    # Insert additional items (should trigger eviction, not growth)
    print(f"Inserting {_FAST_TEST_CAPACITY} additional vectors (expecting eviction)...")
    _fill_pelm_to_capacity(pelm, _FAST_TEST_CAPACITY, _DIMENSION, verbose=False)

    # Stabilize and measure after eviction
    _stabilize_memory()
    after_mb = _measure_memory_mb()

    # Validate zero-growth behavior
    _validate_zero_growth(
        baseline_mb,
        after_mb,
        _MEMORY_TOLERANCE_PERCENT,
        _FAST_TEST_CAPACITY,
    )


@pytest.mark.slow
def test_pelm_zero_growth_after_init_large_capacity():
    """
    Verify zero-growth property with production-sized capacity (20k vectors).

    This test uses the actual production configuration (dimension=384,
    capacity=20,000) to ensure the zero-growth property holds at scale.

    Performance: ~3-5 seconds to execute (marked as slow to allow skipping
    in fast CI runs).

    See test_pelm_zero_growth_after_init for detailed methodology.
    """
    # Create PELM with production configuration
    pelm = PhaseEntangledLatticeMemory(
        dimension=_DIMENSION,
        capacity=_PRODUCTION_CAPACITY,
    )

    # Fill to capacity with progress reporting
    print(f"\nFilling PELM to production capacity ({_PRODUCTION_CAPACITY} vectors)...")
    _fill_pelm_to_capacity(pelm, _PRODUCTION_CAPACITY, _DIMENSION)

    # Stabilize and measure baseline
    _stabilize_memory()
    baseline_mb = _measure_memory_mb()

    # Insert additional vectors to trigger eviction (with progress reporting)
    print(f"Inserting {_PRODUCTION_CAPACITY} additional vectors (expecting eviction)...")
    rng = np.random.default_rng(43)  # Different seed for second batch
    batch_size = 5000
    for batch_start in range(0, _PRODUCTION_CAPACITY, batch_size):
        batch_end = min(batch_start + batch_size, _PRODUCTION_CAPACITY)
        for i in range(batch_start, batch_end):
            vec = rng.standard_normal(_DIMENSION, dtype=np.float32).tolist()
            phase = float(i % _PHASE_VALUES) / float(_PHASE_VALUES - 1)
            pelm.entangle(vec, phase=phase)
        print(f"  Inserted {batch_end}/{_PRODUCTION_CAPACITY} vectors...")

    # Stabilize and measure after eviction
    _stabilize_memory()
    after_mb = _measure_memory_mb()

    # Validate zero-growth behavior
    _validate_zero_growth(
        baseline_mb,
        after_mb,
        _MEMORY_TOLERANCE_PERCENT,
        _PRODUCTION_CAPACITY,
    )
