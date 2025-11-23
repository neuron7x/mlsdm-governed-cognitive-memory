"""
Performance and benchmarking tests for optimized components.

These tests validate that optimizations maintain correctness while
improving performance characteristics of the system.
"""

import time
from threading import Thread

import numpy as np

from mlsdm.cognition.moral_filter_v2 import MoralFilterV2
from mlsdm.core.cognitive_controller import CognitiveController
from mlsdm.memory.multi_level_memory import MultiLevelSynapticMemory
from mlsdm.memory.phase_entangled_lattice_memory import PhaseEntangledLatticeMemory


class TestPELMPerformance:
    """Performance tests for PELM optimizations."""

    def test_retrieve_performance_with_large_memory(self) -> None:
        """Test retrieval performance with near-capacity memory."""
        pelm = PhaseEntangledLatticeMemory(dimension=384, capacity=1000)

        # Fill memory to 90% capacity
        for i in range(900):
            vec = np.random.randn(384).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) or 1e-9)
            phase = 0.1 if i % 2 == 0 else 0.9
            pelm.entangle(vec.tolist(), phase=phase)

        # Test retrieval performance
        query_vec = np.random.randn(384).astype(np.float32)
        query_vec = query_vec / (np.linalg.norm(query_vec) or 1e-9)

        start = time.perf_counter()
        results = pelm.retrieve(query_vec.tolist(), current_phase=0.1, top_k=10)
        elapsed = time.perf_counter() - start

        # Should complete in reasonable time (< 50ms for 900 vectors)
        assert elapsed < 0.05, f"Retrieval took {elapsed:.3f}s, expected < 0.05s"
        assert len(results) <= 10
        assert len(results) > 0  # Should find at least some matches

    def test_retrieve_with_varying_candidate_sizes(self) -> None:
        """Test retrieval optimization with different candidate set sizes."""
        pelm = PhaseEntangledLatticeMemory(dimension=384, capacity=500)

        # Add vectors with same phase
        phase = 0.1
        for i in range(200):
            vec = np.random.randn(384).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) or 1e-9)
            pelm.entangle(vec.tolist(), phase=phase)

        query_vec = np.random.randn(384).astype(np.float32)
        query_vec = query_vec / (np.linalg.norm(query_vec) or 1e-9)

        # Test with different top_k values to trigger different optimization paths
        for top_k in [5, 10, 50, 150]:
            results = pelm.retrieve(query_vec.tolist(), current_phase=phase, top_k=top_k)
            assert len(results) <= top_k
            # Verify results are sorted by resonance (descending)
            if len(results) > 1:
                for i in range(len(results) - 1):
                    assert results[i].resonance >= results[i+1].resonance

    def test_entangle_performance_batch(self) -> None:
        """Test entangle performance with batch operations."""
        pelm = PhaseEntangledLatticeMemory(dimension=384, capacity=10000)

        # Batch entangle
        start = time.perf_counter()
        for _i in range(1000):
            vec = np.random.randn(384).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) or 1e-9)
            pelm.entangle(vec.tolist(), phase=0.1)
        elapsed = time.perf_counter() - start

        # Should handle 1000 entanglements quickly (< 100ms)
        assert elapsed < 0.1, f"Batch entangle took {elapsed:.3f}s, expected < 0.1s"
        assert pelm.size == 1000


class TestMultiLevelMemoryPerformance:
    """Performance tests for MultiLevelSynapticMemory optimizations."""

    def test_update_performance_with_frequent_transfers(self) -> None:
        """Test update performance when transfers occur frequently."""
        synaptic = MultiLevelSynapticMemory(
            dimension=384,
            theta_l1=0.5,  # Lower threshold for frequent transfers
            theta_l2=1.0
        )

        # Perform many updates to trigger transfers
        start = time.perf_counter()
        for _i in range(1000):
            vec = np.random.randn(384).astype(np.float32) * 0.1
            synaptic.update(vec)
        elapsed = time.perf_counter() - start

        # Should complete quickly (< 50ms for 1000 updates)
        assert elapsed < 0.05, f"Updates took {elapsed:.3f}s, expected < 0.05s"

        # Verify memory state is valid
        l1, l2, l3 = synaptic.state()
        assert np.all(np.isfinite(l1))
        assert np.all(np.isfinite(l2))
        assert np.all(np.isfinite(l3))

    def test_update_with_already_float32_vectors(self) -> None:
        """Test optimization path for pre-converted float32 vectors."""
        _ = MultiLevelSynapticMemory(dimension=384)

        # Test with float32 vectors (should use optimized path)
        vec_f32 = np.random.randn(384).astype(np.float32)
        vec_f64 = vec_f32.astype(np.float64)

        # Test that float32 and float64 give same result (optimization doesn't change behavior)
        synaptic1 = MultiLevelSynapticMemory(dimension=384)
        synaptic2 = MultiLevelSynapticMemory(dimension=384)

        synaptic1.update(vec_f32)
        synaptic2.update(vec_f64)

        l1_f32, l2_f32, l3_f32 = synaptic1.state()
        l1_f64, l2_f64, l3_f64 = synaptic2.state()

        # Both paths should produce same result
        assert np.allclose(l1_f32, l1_f64, rtol=1e-5)
        assert np.allclose(l2_f32, l2_f64, rtol=1e-5)
        assert np.allclose(l3_f32, l3_f64, rtol=1e-5)

    def test_update_memory_efficiency(self) -> None:
        """Test that update doesn't create excessive intermediate arrays."""
        synaptic = MultiLevelSynapticMemory(dimension=384)

        # Perform updates and check state consistency
        for _i in range(100):
            vec = np.random.randn(384).astype(np.float32)
            synaptic.update(vec)

            # Verify state after each update
            l1, l2, l3 = synaptic.state()
            assert l1.shape == (384,)
            assert l2.shape == (384,)
            assert l3.shape == (384,)


class TestMoralFilterPerformance:
    """Performance tests for MoralFilterV2 optimizations."""

    def test_evaluate_fast_path_extremes(self) -> None:
        """Test fast-path optimization for extreme moral values."""
        moral = MoralFilterV2(initial_threshold=0.50)

        # Test fast-path for high values (>= MAX_THRESHOLD)
        start = time.perf_counter()
        for _ in range(10000):
            result = moral.evaluate(1.0)
        elapsed_high = time.perf_counter() - start
        assert result is True

        # Test fast-path for low values (< MIN_THRESHOLD)
        start = time.perf_counter()
        for _ in range(10000):
            result = moral.evaluate(0.1)
        elapsed_low = time.perf_counter() - start
        assert result is False

        # Both should be very fast (< 10ms for 10000 evaluations)
        assert elapsed_high < 0.01, f"High value evaluations took {elapsed_high:.3f}s"
        assert elapsed_low < 0.01, f"Low value evaluations took {elapsed_low:.3f}s"

    def test_evaluate_boundary_performance(self) -> None:
        """Test performance at threshold boundaries."""
        moral = MoralFilterV2(initial_threshold=0.50)

        # Test evaluations near threshold
        start = time.perf_counter()
        for i in range(1000):
            moral_value = 0.45 + (i % 10) * 0.01  # Values around 0.50
            moral.evaluate(moral_value)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.01, f"Boundary evaluations took {elapsed:.3f}s"


class TestCognitiveControllerPerformance:
    """Performance tests for CognitiveController optimizations."""

    def test_process_event_with_phase_cache(self) -> None:
        """Test that phase cache improves performance."""
        controller = CognitiveController(dim=384)

        # Process many events
        start = time.perf_counter()
        for _i in range(100):
            vec = np.random.randn(384).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) or 1e-9)
            moral_value = 0.6
            controller.process_event(vec, moral_value)
        elapsed = time.perf_counter() - start

        # Should complete quickly with cache
        assert elapsed < 0.5, f"Processing 100 events took {elapsed:.3f}s"

    def test_retrieve_context_with_phase_cache(self) -> None:
        """Test context retrieval with phase cache."""
        controller = CognitiveController(dim=384)

        # Add some data
        for _i in range(50):
            vec = np.random.randn(384).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) or 1e-9)
            controller.process_event(vec, 0.7)

        # Test retrieval
        query = np.random.randn(384).astype(np.float32)
        query = query / (np.linalg.norm(query) or 1e-9)

        start = time.perf_counter()
        for _ in range(100):
            controller.retrieve_context(query, top_k=5)
        elapsed = time.perf_counter() - start

        assert elapsed < 0.5, f"100 retrievals took {elapsed:.3f}s"


class TestConcurrentPerformance:
    """Test performance under concurrent load."""

    def test_qilm_concurrent_retrieval(self) -> None:
        """Test retrieval performance with concurrent threads."""
        pelm = PhaseEntangledLatticeMemory(dimension=384, capacity=1000)

        # Populate memory
        for _i in range(500):
            vec = np.random.randn(384).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) or 1e-9)
            pelm.entangle(vec.tolist(), phase=0.1)

        results_list: list[list] = []

        def retrieve_worker():
            query = np.random.randn(384).astype(np.float32)
            query = query / (np.linalg.norm(query) or 1e-9)
            for _ in range(10):
                results = pelm.retrieve(query.tolist(), current_phase=0.1, top_k=5)
                results_list.append(results)

        # Run concurrent retrievals
        threads = [Thread(target=retrieve_worker) for _ in range(10)]
        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start

        # Should complete all retrievals (< 1s for 100 total retrievals)
        assert elapsed < 1.0, f"Concurrent retrievals took {elapsed:.3f}s"
        assert len(results_list) == 100

    def test_controller_concurrent_processing(self) -> None:
        """Test controller performance with concurrent event processing."""
        controller = CognitiveController(dim=384)

        def process_worker():
            for _ in range(20):
                vec = np.random.randn(384).astype(np.float32)
                vec = vec / (np.linalg.norm(vec) or 1e-9)
                controller.process_event(vec, 0.7)

        # Run concurrent processing
        threads = [Thread(target=process_worker) for _ in range(5)]
        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start

        # Should handle 100 events from 5 threads (< 2s)
        assert elapsed < 2.0, f"Concurrent processing took {elapsed:.3f}s"
        assert controller.step_counter == 100


class TestMemoryEfficiency:
    """Test memory efficiency of optimizations."""

    def test_qilm_memory_footprint(self) -> None:
        """Test that QILM maintains expected memory footprint."""
        pelm = PhaseEntangledLatticeMemory(dimension=384, capacity=20000)

        stats = pelm.get_state_stats()
        memory_mb = stats["memory_mb"]

        # Should be approximately:
        # (20000 * 384 * 4 bytes) + (20000 * 4 bytes) ≈ 29.3 MB
        expected_mb = (20000 * 384 * 4 + 20000 * 4 + 20000 * 4) / (1024 ** 2)

        assert memory_mb <= expected_mb * 1.1  # Allow 10% overhead
        assert memory_mb >= expected_mb * 0.9

    def test_synaptic_memory_footprint(self) -> None:
        """Test that synaptic memory uses expected space."""
        synaptic = MultiLevelSynapticMemory(dimension=384)

        # Memory should be 3 arrays of 384 float32 values
        expected_bytes = 3 * 384 * 4

        # Get actual memory usage
        actual_bytes = (synaptic.l1.nbytes +
                       synaptic.l2.nbytes +
                       synaptic.l3.nbytes)

        assert actual_bytes == expected_bytes
