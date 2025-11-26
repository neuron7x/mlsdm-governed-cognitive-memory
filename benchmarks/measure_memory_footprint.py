#!/usr/bin/env python3
"""
Memory footprint benchmark for MLSDM PELM.

Measures actual memory usage of Phase-Entangled Lattice Memory (PELM)
to verify the documented 29.37 MB footprint claim.

Run: python benchmarks/measure_memory_footprint.py
"""

import gc
import sys
import tracemalloc

import numpy as np


def measure_pelm_memory():
    """Measure PELM memory footprint."""
    from mlsdm.memory.phase_entangled_lattice_memory import PhaseEntangledLatticeMemory
    
    print("=" * 70)
    print("MLSDM MEMORY FOOTPRINT MEASUREMENT")
    print("=" * 70)
    
    # Force garbage collection before measurement
    gc.collect()
    
    # Start tracemalloc for accurate measurement
    tracemalloc.start()
    
    # Create PELM with production configuration
    dimension = 384
    capacity = 20_000
    
    print(f"\nConfiguration:")
    print(f"  Dimension: {dimension}")
    print(f"  Capacity: {capacity}")
    print(f"  Expected footprint: {capacity * dimension * 4 / (1024 * 1024):.2f} MB (data only)")
    
    # Create PELM
    pelm = PhaseEntangledLatticeMemory(dimension=dimension, capacity=capacity)
    
    # Get baseline memory
    current, peak = tracemalloc.get_traced_memory()
    print(f"\nAfter PELM creation (empty):")
    print(f"  Current: {current / (1024 * 1024):.2f} MB")
    print(f"  Peak: {peak / (1024 * 1024):.2f} MB")
    
    # Fill PELM with batch of vectors (faster than one at a time)
    print(f"\nFilling PELM with {capacity} vectors...")
    batch_size = 1000
    for batch_start in range(0, capacity, batch_size):
        batch_end = min(batch_start + batch_size, capacity)
        for i in range(batch_start, batch_end):
            # Use numpy for speed, convert to list for API
            vector = np.random.randn(dimension).astype(np.float32).tolist()
            pelm.entangle(vector, phase=float(i % 11) / 10.0)
        
        if batch_end % 5000 == 0 or batch_end == capacity:
            current, peak = tracemalloc.get_traced_memory()
            print(f"  At {batch_end} vectors: {current / (1024 * 1024):.2f} MB current")
    
    # Final measurement
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"\nFinal measurements (at capacity):")
    print(f"  Current memory: {current / (1024 * 1024):.2f} MB")
    print(f"  Peak memory: {peak / (1024 * 1024):.2f} MB")
    
    # Calculate theoretical size
    theoretical_data = capacity * dimension * 4  # 4 bytes per float32
    theoretical_mb = theoretical_data / (1024 * 1024)
    
    print(f"\nTheoretical calculation:")
    print(f"  Vector data: {capacity} × {dimension} × 4 bytes = {theoretical_mb:.2f} MB")
    print(f"  Actual measured: {current / (1024 * 1024):.2f} MB")
    print(f"  Overhead: {(current / (1024 * 1024)) - theoretical_mb:.2f} MB")
    
    # Verify against documented claim
    documented_footprint_mb = 29.37
    measured_mb = current / (1024 * 1024)
    
    print(f"\n{'=' * 70}")
    print("VERIFICATION AGAINST DOCUMENTED CLAIM")
    print("=" * 70)
    print(f"  Documented: {documented_footprint_mb:.2f} MB")
    print(f"  Measured: {measured_mb:.2f} MB")
    
    if measured_mb <= documented_footprint_mb * 1.1:  # Allow 10% margin
        print(f"  Status: ✅ VERIFIED (within 10% of documented value)")
    else:
        print(f"  Status: ⚠️ EXCEEDS documented value by {((measured_mb / documented_footprint_mb) - 1) * 100:.1f}%")
    
    print("=" * 70)
    
    return measured_mb


def measure_cognitive_controller_memory():
    """Measure full CognitiveController memory footprint."""
    from mlsdm.core.cognitive_controller import CognitiveController
    
    print("\n" + "=" * 70)
    print("COGNITIVE CONTROLLER MEMORY FOOTPRINT")
    print("=" * 70)
    
    gc.collect()
    tracemalloc.start()
    
    # Create controller with production settings
    dimension = 384  # Standard embedding dimension
    controller = CognitiveController(
        dim=dimension,
        capacity=20_000,
        wake_duration=8,
        sleep_duration=3,
        initial_moral_threshold=0.50
    )
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"\nAfter CognitiveController creation:")
    print(f"  Current: {current / (1024 * 1024):.2f} MB")
    print(f"  Peak: {peak / (1024 * 1024):.2f} MB")
    
    # Process some events
    print("\nProcessing 100 events...")
    for i in range(100):
        vector = np.random.randn(dimension).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        controller.process_event(vector, moral_value=0.8)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"\nAfter processing 100 events:")
    print(f"  Current: {current / (1024 * 1024):.2f} MB")
    print(f"  Peak: {peak / (1024 * 1024):.2f} MB")
    
    print("=" * 70)
    
    return current / (1024 * 1024)


def quick_memory_check():
    """Quick memory check for CI - verifies empty PELM footprint."""
    from mlsdm.memory.phase_entangled_lattice_memory import PhaseEntangledLatticeMemory
    
    gc.collect()
    tracemalloc.start()
    
    pelm = PhaseEntangledLatticeMemory(dimension=384, capacity=20_000)
    
    current, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    measured_mb = current / (1024 * 1024)
    documented_footprint_mb = 29.37
    
    print(f"Quick check: PELM initial footprint = {measured_mb:.2f} MB")
    print(f"Documented: {documented_footprint_mb:.2f} MB")
    
    # Verify pre-allocated memory is within expected range
    assert measured_mb >= 29.0, f"PELM footprint {measured_mb:.2f} MB is too small (expected ~29 MB)"
    assert measured_mb <= 35.0, f"PELM footprint {measured_mb:.2f} MB exceeds limit"
    
    print("✅ Memory footprint within expected range")
    return 0


def main():
    """Run memory benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MLSDM Memory Footprint Benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick check for CI")
    args = parser.parse_args()
    
    if args.quick:
        return quick_memory_check()
    
    print("\n" + "=" * 70)
    print("MLSDM MEMORY FOOTPRINT BENCHMARK")
    print("=" * 70)
    print("\nThis benchmark verifies documented memory claims.")
    print("Reference: SLO_SPEC.md, ARCHITECTURE_SPEC.md, CLAIMS_TRACEABILITY.md")
    
    pelm_mb = measure_pelm_memory()
    controller_mb = measure_cognitive_controller_memory()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  PELM (20k vectors, 384 dim): {pelm_mb:.2f} MB")
    print(f"  CognitiveController (full): {controller_mb:.2f} MB")
    print(f"\n  Documented footprint: 29.37 MB")
    print("=" * 70)
    
    return 0 if pelm_mb <= 35.0 else 1  # Allow some margin


if __name__ == "__main__":
    sys.exit(main())
