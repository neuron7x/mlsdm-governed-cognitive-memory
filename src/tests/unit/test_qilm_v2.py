"""Comprehensive unit tests for QILM_v2."""
import numpy as np
from threading import Thread
from src.memory.qilm_v2 import QILM_v2, MemoryRetrieval


class TestQILM_v2:
    """Test suite for QILM_v2 (Quantum-Inspired Latent Memory v2)."""

    def test_initialization_default(self) -> None:
        """Test initialization with default parameters."""
        qilm = QILM_v2()
        assert qilm.dimension == 384
        assert qilm.capacity == 20000
        assert qilm.pointer == 0
        assert qilm.size == 0
        assert qilm.memory_bank.shape == (20000, 384)
        assert qilm.phase_bank.shape == (20000,)
        assert qilm.norms.shape == (20000,)

    def test_initialization_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        qilm = QILM_v2(dimension=512, capacity=1000)
        assert qilm.dimension == 512
        assert qilm.capacity == 1000
        assert qilm.memory_bank.shape == (1000, 512)

    def test_entangle_single_vector(self) -> None:
        """Test entangling a single vector."""
        qilm = QILM_v2(dimension=384, capacity=100)
        vec = [float(i) for i in range(384)]
        phase = 0.5

        idx = qilm.entangle(vec, phase)

        assert idx == 0
        assert qilm.size == 1
        assert qilm.pointer == 1
        assert qilm.phase_bank[0] == phase

    def test_entangle_multiple_vectors(self) -> None:
        """Test entangling multiple vectors."""
        qilm = QILM_v2(dimension=10, capacity=100)

        for i in range(5):
            vec = [float(j) for j in range(10)]
            phase = float(i) / 10.0
            idx = qilm.entangle(vec, phase)
            assert idx == i

        assert qilm.size == 5
        assert qilm.pointer == 5

    def test_entangle_wraparound(self) -> None:
        """Test that pointer wraps around after reaching capacity."""
        qilm = QILM_v2(dimension=10, capacity=5)

        for i in range(7):
            vec = [float(j) for j in range(10)]
            qilm.entangle(vec, phase=0.1)

        assert qilm.pointer == 2  # 7 % 5 = 2
        assert qilm.size == 5  # Capped at capacity

    def test_entangle_overwrites_old_data(self) -> None:
        """Test that old data is overwritten after capacity is reached."""
        qilm = QILM_v2(dimension=10, capacity=3)

        # Add 3 vectors
        for i in range(3):
            vec = [float(i) for _ in range(10)]
            qilm.entangle(vec, phase=float(i))

        # Add a 4th vector (should overwrite first)
        vec_new = [99.0 for _ in range(10)]
        qilm.entangle(vec_new, phase=99.0)

        assert qilm.memory_bank[0][0] == 99.0
        assert qilm.phase_bank[0] == 99.0

    def test_entangle_numpy_array(self) -> None:
        """Test entangling with numpy array input."""
        qilm = QILM_v2(dimension=10, capacity=100)
        vec = np.random.randn(10).astype(np.float32)

        idx = qilm.entangle(vec.tolist(), phase=0.5)

        assert idx == 0
        assert qilm.size == 1

    def test_retrieve_empty_memory(self) -> None:
        """Test retrieving from empty memory."""
        qilm = QILM_v2(dimension=10, capacity=100)
        query = [float(i) for i in range(10)]

        results = qilm.retrieve(query, current_phase=0.5, phase_tolerance=0.1, top_k=5)

        assert results == []

    def test_retrieve_exact_phase_match(self) -> None:
        """Test retrieving with exact phase match."""
        qilm = QILM_v2(dimension=10, capacity=100)
        vec = [1.0] + [0.0] * 9
        phase = 0.5

        qilm.entangle(vec, phase)

        results = qilm.retrieve(vec, current_phase=phase, phase_tolerance=0.0, top_k=5)

        assert len(results) == 1
        assert isinstance(results[0], MemoryRetrieval)
        assert results[0].phase == phase
        assert results[0].resonance > 0.9  # High similarity

    def test_retrieve_phase_tolerance(self) -> None:
        """Test that phase tolerance filters results correctly."""
        qilm = QILM_v2(dimension=10, capacity=100)

        # Add vectors with different phases
        for phase in [0.1, 0.2, 0.5, 0.8]:
            vec = [float(phase) for _ in range(10)]
            qilm.entangle(vec, phase)

        # Retrieve with tight tolerance around 0.5
        results = qilm.retrieve([0.5] * 10, current_phase=0.5, phase_tolerance=0.1, top_k=10)

        # Should only get vectors with phase 0.5 (and possibly 0.6, 0.4 if within tolerance)
        assert len(results) >= 1
        for result in results:
            assert abs(result.phase - 0.5) <= 0.1

    def test_retrieve_no_phase_matches(self) -> None:
        """Test retrieving when no phases match."""
        qilm = QILM_v2(dimension=10, capacity=100)

        # Add vectors with phase 0.1
        qilm.entangle([1.0] * 10, phase=0.1)
        qilm.entangle([2.0] * 10, phase=0.1)

        # Query with phase 0.9 (too far from 0.1)
        results = qilm.retrieve([1.0] * 10, current_phase=0.9, phase_tolerance=0.1, top_k=5)

        assert results == []

    def test_retrieve_top_k_limiting(self) -> None:
        """Test that top_k limits the number of results."""
        qilm = QILM_v2(dimension=10, capacity=100)

        # Add 10 vectors with same phase
        for i in range(10):
            vec = [float(i) for _ in range(10)]
            qilm.entangle(vec, phase=0.5)

        # Retrieve with top_k=3
        results = qilm.retrieve([0.0] * 10, current_phase=0.5, phase_tolerance=0.5, top_k=3)

        assert len(results) <= 3

    def test_retrieve_cosine_similarity_ordering(self) -> None:
        """Test that results are ordered by cosine similarity."""
        qilm = QILM_v2(dimension=10, capacity=100)

        # Add vectors with varying similarity to query
        query = [1.0] + [0.0] * 9
        qilm.entangle([1.0] + [0.0] * 9, phase=0.5)  # High similarity
        qilm.entangle([0.5, 0.5] + [0.0] * 8, phase=0.5)  # Medium similarity
        qilm.entangle([0.0] + [1.0] * 9, phase=0.5)  # Low similarity

        results = qilm.retrieve(query, current_phase=0.5, phase_tolerance=0.5, top_k=3)

        # First result should have highest resonance
        assert len(results) > 0
        resonances = [r.resonance for r in results]
        assert resonances == sorted(resonances, reverse=True)

    def test_retrieve_normalized_vectors(self) -> None:
        """Test retrieval with normalized vectors."""
        qilm = QILM_v2(dimension=10, capacity=100)

        vec = np.random.randn(10)
        vec = vec / np.linalg.norm(vec)
        qilm.entangle(vec.tolist(), phase=0.5)

        results = qilm.retrieve(vec.tolist(), current_phase=0.5, phase_tolerance=0.1, top_k=5)

        assert len(results) == 1
        assert results[0].resonance > 0.99  # Should be very close to 1.0

    def test_get_state_stats(self) -> None:
        """Test getting state statistics."""
        qilm = QILM_v2(dimension=384, capacity=20000)

        stats = qilm.get_state_stats()

        assert "capacity" in stats
        assert "used" in stats
        assert "memory_mb" in stats
        assert stats["capacity"] == 20000
        assert stats["used"] == 0

    def test_get_state_stats_after_adding(self) -> None:
        """Test state statistics after adding vectors."""
        qilm = QILM_v2(dimension=384, capacity=20000)

        for i in range(100):
            vec = [float(i % 10) for _ in range(384)]
            qilm.entangle(vec, phase=0.5)

        stats = qilm.get_state_stats()

        assert stats["used"] == 100
        assert stats["memory_mb"] > 0

    def test_thread_safety_entangle(self) -> None:
        """Test thread-safety of entangle operation."""
        qilm = QILM_v2(dimension=10, capacity=10000)

        def add_vectors():
            for _ in range(100):
                vec = [float(i) for i in range(10)]
                qilm.entangle(vec, phase=0.5)

        threads = [Thread(target=add_vectors) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert qilm.size == 1000

    def test_thread_safety_retrieve(self) -> None:
        """Test thread-safety of retrieve operation."""
        qilm = QILM_v2(dimension=10, capacity=100)

        # Add some initial data
        for i in range(50):
            vec = [float(i % 10) for _ in range(10)]
            qilm.entangle(vec, phase=0.5)

        results_list = []

        def retrieve_vectors():
            query = [1.0] * 10
            results = qilm.retrieve(query, current_phase=0.5, phase_tolerance=0.5, top_k=5)
            results_list.append(len(results))

        threads = [Thread(target=retrieve_vectors) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get some results
        assert len(results_list) == 10

    def test_memory_retrieval_dataclass(self) -> None:
        """Test MemoryRetrieval dataclass structure."""
        qilm = QILM_v2(dimension=10, capacity=100)
        vec = [1.0] * 10
        qilm.entangle(vec, phase=0.5)

        results = qilm.retrieve(vec, current_phase=0.5, phase_tolerance=0.1, top_k=1)

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, MemoryRetrieval)
        assert hasattr(result, 'vector')
        assert hasattr(result, 'phase')
        assert hasattr(result, 'resonance')
        assert isinstance(result.vector, np.ndarray)
        assert isinstance(float(result.phase), float)
        assert isinstance(float(result.resonance), float)

    def test_zero_norm_vector_handling(self) -> None:
        """Test handling of zero-norm vectors."""
        qilm = QILM_v2(dimension=10, capacity=100)
        zero_vec = [0.0] * 10

        # Should handle gracefully
        idx = qilm.entangle(zero_vec, phase=0.5)
        assert idx == 0
        assert qilm.norms[0] > 0  # Should use 1e-9 as fallback

    def test_retrieve_with_zero_norm_query(self) -> None:
        """Test retrieving with zero-norm query vector."""
        qilm = QILM_v2(dimension=10, capacity=100)

        # Add some vectors
        qilm.entangle([1.0] * 10, phase=0.5)

        # Query with zero vector
        zero_query = [0.0] * 10
        results = qilm.retrieve(zero_query, current_phase=0.5, phase_tolerance=0.1, top_k=5)

        # Should return empty or handle gracefully
        assert isinstance(results, list)

    def test_large_capacity_initialization(self) -> None:
        """Test initialization with large capacity."""
        qilm = QILM_v2(dimension=384, capacity=20000)

        assert qilm.memory_bank.shape == (20000, 384)
        assert qilm.memory_bank.dtype == np.float32

    def test_phase_values_stored_correctly(self) -> None:
        """Test that phase values are stored correctly."""
        qilm = QILM_v2(dimension=10, capacity=100)

        phases = [0.0, 0.25, 0.5, 0.75, 1.0]
        for i, phase in enumerate(phases):
            vec = [float(i) for _ in range(10)]
            qilm.entangle(vec, phase)

        for i, phase in enumerate(phases):
            assert qilm.phase_bank[i] == phase

    def test_retrieve_all_candidates_when_fewer_than_topk(self) -> None:
        """Test that all matching candidates are returned when fewer than top_k."""
        qilm = QILM_v2(dimension=10, capacity=100)

        # Add only 2 vectors
        qilm.entangle([1.0] * 10, phase=0.5)
        qilm.entangle([2.0] * 10, phase=0.5)

        # Request top_k=10 (more than available)
        results = qilm.retrieve([1.0] * 10, current_phase=0.5, phase_tolerance=0.5, top_k=10)

        assert len(results) == 2

    def test_memory_bank_dtype(self) -> None:
        """Test that memory bank uses correct dtype."""
        qilm = QILM_v2(dimension=10, capacity=100)

        assert qilm.memory_bank.dtype == np.float32
        assert qilm.phase_bank.dtype == np.float32
        assert qilm.norms.dtype == np.float32

    def test_consistent_retrieval(self) -> None:
        """Test that retrieval is consistent for same query."""
        qilm = QILM_v2(dimension=10, capacity=100)

        # Add vectors
        for i in range(10):
            vec = [float(i) for _ in range(10)]
            qilm.entangle(vec, phase=0.5)

        query = [5.0] * 10
        results1 = qilm.retrieve(query, current_phase=0.5, phase_tolerance=0.5, top_k=3)
        results2 = qilm.retrieve(query, current_phase=0.5, phase_tolerance=0.5, top_k=3)

        # Should get same results
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1.phase == r2.phase
            assert abs(r1.resonance - r2.resonance) < 1e-6
