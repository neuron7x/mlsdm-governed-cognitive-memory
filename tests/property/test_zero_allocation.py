"""
Property-based test for zero allocation after initialization.

Validates the claim that PELM memory does not grow after reaching capacity,
as it uses fixed pre-allocated buffers with eviction-based updates.
"""

import gc

import numpy as np
import psutil
import pytest

from mlsdm.memory.phase_entangled_lattice_memory import PhaseEntangledLatticeMemory


def test_pelm_zero_growth_after_init():
    """
    Verify PELM memory does not grow significantly after reaching capacity.
    
    This test validates that once PELM reaches its configured capacity,
    additional insertions trigger eviction rather than memory growth,
    maintaining a fixed memory footprint.
    
    Note: We allow a 5% tolerance for:
    - Python garbage collection overhead
    - OS memory allocation granularity
    - Minor internal buffer adjustments
    """
    # Create PELM with moderate capacity for faster testing
    dimension = 384
    capacity = 1000
    pelm = PhaseEntangledLatticeMemory(dimension=dimension, capacity=capacity)
    
    # Fill to capacity
    print(f"\nFilling PELM to capacity ({capacity} vectors)...")
    for i in range(capacity):
        vec = np.random.randn(dimension).astype(np.float32).tolist()
        pelm.entangle(vec, phase=float(i % 11) / 10.0)
    
    # Force garbage collection to stabilize memory baseline
    gc.collect()
    
    # Measure baseline memory after reaching capacity
    process = psutil.Process()
    baseline_mb = process.memory_info().rss / (1024 * 1024)
    print(f"Baseline memory at capacity: {baseline_mb:.2f} MB")
    
    # Insert 1000 more items (should trigger eviction, not growth)
    print(f"Inserting {capacity} additional vectors (expecting eviction)...")
    for i in range(capacity):
        vec = np.random.randn(dimension).astype(np.float32).tolist()
        pelm.entangle(vec, phase=float(i % 11) / 10.0)
    
    # Force garbage collection again
    gc.collect()
    
    # Measure memory after eviction phase
    after_mb = process.memory_info().rss / (1024 * 1024)
    print(f"Memory after eviction phase: {after_mb:.2f} MB")
    
    growth_mb = after_mb - baseline_mb
    growth_percent = (growth_mb / baseline_mb) * 100
    print(f"Memory growth: {growth_mb:.2f} MB ({growth_percent:.2f}%)")
    
    # Allow 5% growth for GC overhead and OS granularity
    # This is a reasonable tolerance for CI environments where memory
    # behavior can be less predictable than local development
    tolerance_percent = 5.0
    max_allowed_mb = baseline_mb * (1.0 + tolerance_percent / 100.0)
    
    assert after_mb <= max_allowed_mb, (
        f"Memory grew from {baseline_mb:.2f} MB to {after_mb:.2f} MB "
        f"({growth_percent:.2f}% growth), exceeding {tolerance_percent}% tolerance. "
        f"Expected zero-growth behavior with eviction-based updates."
    )
    
    print(f"✅ PASS: Memory growth {growth_percent:.2f}% is within {tolerance_percent}% tolerance")


def test_pelm_zero_growth_after_init_large_capacity():
    """
    Verify zero-growth property with production-sized capacity (20k vectors).
    
    This test uses the actual production configuration to ensure the
    zero-growth property holds at scale.
    
    Note: Marked as slow because filling 20k vectors takes ~3-5 seconds.
    """
    dimension = 384
    capacity = 20_000
    pelm = PhaseEntangledLatticeMemory(dimension=dimension, capacity=capacity)
    
    # Fill to capacity (using batches for progress reporting)
    print(f"\nFilling PELM to production capacity ({capacity} vectors)...")
    batch_size = 5000
    for batch_start in range(0, capacity, batch_size):
        batch_end = min(batch_start + batch_size, capacity)
        for i in range(batch_start, batch_end):
            vec = np.random.randn(dimension).astype(np.float32).tolist()
            pelm.entangle(vec, phase=float(i % 11) / 10.0)
        print(f"  Filled {batch_end}/{capacity} vectors...")
    
    # Force garbage collection
    gc.collect()
    
    # Measure baseline
    process = psutil.Process()
    baseline_mb = process.memory_info().rss / (1024 * 1024)
    print(f"Baseline memory at capacity: {baseline_mb:.2f} MB")
    
    # Insert additional vectors to trigger eviction
    print(f"Inserting {capacity} additional vectors (expecting eviction)...")
    for i in range(capacity):
        vec = np.random.randn(dimension).astype(np.float32).tolist()
        pelm.entangle(vec, phase=float(i % 11) / 10.0)
        
        # Progress update
        if (i + 1) % 5000 == 0:
            print(f"  Inserted {i + 1}/{capacity} additional vectors...")
    
    # Force garbage collection
    gc.collect()
    
    # Measure after eviction
    after_mb = process.memory_info().rss / (1024 * 1024)
    print(f"Memory after eviction phase: {after_mb:.2f} MB")
    
    growth_mb = after_mb - baseline_mb
    growth_percent = (growth_mb / baseline_mb) * 100
    print(f"Memory growth: {growth_mb:.2f} MB ({growth_percent:.2f}%)")
    
    # Allow 5% growth tolerance
    tolerance_percent = 5.0
    max_allowed_mb = baseline_mb * (1.0 + tolerance_percent / 100.0)
    
    assert after_mb <= max_allowed_mb, (
        f"Memory grew from {baseline_mb:.2f} MB to {after_mb:.2f} MB "
        f"({growth_percent:.2f}% growth), exceeding {tolerance_percent}% tolerance. "
        f"Expected zero-growth behavior with eviction-based updates."
    )
    
    print(f"✅ PASS: Memory growth {growth_percent:.2f}% is within {tolerance_percent}% tolerance")


# Mark the large capacity test as slow to allow skipping in fast CI runs
test_pelm_zero_growth_after_init_large_capacity = pytest.mark.slow(
    test_pelm_zero_growth_after_init_large_capacity
)
