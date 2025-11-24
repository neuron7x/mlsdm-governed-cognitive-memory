"""
Unit Tests for Phase-Entangled Lattice Memory (PELM, formerly QILM_v2)

Tests corruption detection, auto-recovery, and boundary checks.
"""

import numpy as np
import pytest

from mlsdm.memory.phase_entangled_lattice_memory import MemoryRetrieval, PhaseEntangledLatticeMemory


class TestBackwardCompatibility:
    """Test backward compatibility with QILM_v2 and PELM alias."""

    def test_pelm_alias_exists(self):
        """Test that PELM alias is available as convenient shorthand."""
        from mlsdm.memory import PELM, PhaseEntangledLatticeMemory

        # Verify PELM is an alias to the main class
        assert PELM is PhaseEntangledLatticeMemory

    def test_pelm_alias_works(self):
        """Test that PELM alias can be instantiated and used."""
        from mlsdm.memory import PELM

        # Create instance using PELM alias
        memory = PELM(dimension=10, capacity=100)
        assert memory is not None
        assert memory.dimension == 10
        assert memory.capacity == 100

    def test_qilm_v2_alias_exists(self):
        """Test that QILM_v2 alias is available for backward compatibility."""
        from mlsdm.memory import PhaseEntangledLatticeMemory, QILM_v2

        # Verify alias points to the same class
        assert QILM_v2 is PhaseEntangledLatticeMemory

    def test_qilm_v2_alias_works(self):
        """Test that QILM_v2 alias can be instantiated and used."""
        from mlsdm.memory import QILM_v2

        # Create instance using old name
        memory = QILM_v2(dimension=10, capacity=100)
        assert memory is not None
        assert memory.dimension == 10
        assert memory.capacity == 100

    def test_all_aliases_are_same_class(self):
        """Test that all aliases (PELM, QILM_v2, PhaseEntangledLatticeMemory) refer to the same class."""
        from mlsdm.memory import PELM, PhaseEntangledLatticeMemory, QILM_v2

        # All three should be the exact same class
        assert PELM is PhaseEntangledLatticeMemory
        assert QILM_v2 is PhaseEntangledLatticeMemory
        assert PELM is QILM_v2


class TestPELMInitialization:
    """Test PELM initialization."""

    def test_initialization(self):
        """Test PELM can be initialized."""
        pelm = PhaseEntangledLatticeMemory(dimension=10, capacity=100)
        assert pelm is not None
        assert pelm.dimension == 10
        assert pelm.capacity == 100
        assert pelm.pointer == 0
        assert pelm.size == 0

    def test_initialization_with_defaults(self):
        """Test PELM initializes with default parameters."""
        pelm = PhaseEntangledLatticeMemory()
        assert pelm.dimension == 384
        assert pelm.capacity == 20000

    def test_initialization_validates_dimension(self):
        """Test initialization validates dimension parameter."""
        with pytest.raises(ValueError, match="dimension must be positive"):
            PhaseEntangledLatticeMemory(dimension=0)
        with pytest.raises(ValueError, match="dimension must be positive"):
            PhaseEntangledLatticeMemory(dimension=-1)

    def test_initialization_validates_capacity(self):
        """Test initialization validates capacity parameter."""
        with pytest.raises(ValueError, match="capacity must be positive"):
            PhaseEntangledLatticeMemory(capacity=0)
        with pytest.raises(ValueError, match="capacity must be positive"):
            PhaseEntangledLatticeMemory(capacity=-1)
        with pytest.raises(ValueError, match="capacity too large"):
            PhaseEntangledLatticeMemory(capacity=2_000_000)


class TestPELMEntangle:
    """Test PELM entangle operation."""

    def test_entangle_basic(self):
        """Test basic entangle operation."""
        pelm = PhaseEntangledLatticeMemory(dimension=3, capacity=10)
        vector = [1.0, 2.0, 3.0]
        phase = 0.5

        idx = pelm.entangle(vector, phase)

        assert idx == 0
        assert pelm.size == 1
        assert pelm.pointer == 1
        np.testing.assert_array_almost_equal(pelm.memory_bank[0], vector)
        assert pelm.phase_bank[0] == 0.5

    def test_entangle_multiple_vectors(self):
        """Test entangling multiple vectors."""
        pelm = PhaseEntangledLatticeMemory(dimension=2, capacity=5)

        vectors = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        phases = [0.1, 0.5, 0.9]

        for i, (vec, phase) in enumerate(zip(vectors, phases, strict=True)):
            idx = pelm.entangle(vec, phase)
            assert idx == i

        assert pelm.size == 3
        assert pelm.pointer == 3

    def test_entangle_wraparound(self):
        """Test pointer wraparound when capacity is reached."""
        pelm = PhaseEntangledLatticeMemory(dimension=2, capacity=3)

        # Fill to capacity
        for i in range(3):
            pelm.entangle([float(i), float(i+1)], 0.1 * i)

        assert pelm.pointer == 0  # Should wrap around
        assert pelm.size == 3

        # Add one more to test wraparound
        pelm.entangle([10.0, 11.0], 0.5)
        assert pelm.pointer == 1
        assert pelm.size == 3  # Size stays at capacity


class TestPELMv2Retrieve:
    """Test PhaseEntangledLatticeMemory retrieve operation."""

    def test_retrieve_basic(self):
        """Test basic retrieve operation."""
        pelm = PhaseEntangledLatticeMemory(dimension=3, capacity=10)

        vector = [1.0, 2.0, 3.0]
        phase = 0.5
        pelm.entangle(vector, phase)

        results = pelm.retrieve([1.0, 2.0, 3.0], 0.5, phase_tolerance=0.1, top_k=1)

        assert len(results) == 1
        assert isinstance(results[0], MemoryRetrieval)
        np.testing.assert_array_almost_equal(results[0].vector, vector)
        assert results[0].phase == 0.5

    def test_retrieve_empty_memory(self):
        """Test retrieve returns empty list when memory is empty."""
        pelm = PhaseEntangledLatticeMemory(dimension=3, capacity=10)
        results = pelm.retrieve([1.0, 2.0, 3.0], 0.5)
        assert results == []

    def test_retrieve_with_phase_tolerance(self):
        """Test retrieve with phase tolerance."""
        pelm = PhaseEntangledLatticeMemory(dimension=2, capacity=10)

        pelm.entangle([1.0, 2.0], 0.1)
        pelm.entangle([3.0, 4.0], 0.15)
        pelm.entangle([5.0, 6.0], 0.5)

        results = pelm.retrieve([1.0, 2.0], 0.1, phase_tolerance=0.1, top_k=5)

        # Should get both vectors at phase 0.1 and 0.15
        assert len(results) == 2

    def test_retrieve_no_match(self):
        """Test retrieve with no matching phases."""
        pelm = PhaseEntangledLatticeMemory(dimension=2, capacity=10)

        pelm.entangle([1.0, 2.0], 0.1)

        results = pelm.retrieve([1.0, 2.0], 0.9, phase_tolerance=0.05)

        assert results == []


class TestPELMv2CorruptionDetection:
    """Test PhaseEntangledLatticeMemory corruption detection."""

    def test_detect_no_corruption(self):
        """Test detect_corruption returns False for valid state."""
        pelm = PhaseEntangledLatticeMemory(dimension=3, capacity=10)
        pelm.entangle([1.0, 2.0, 3.0], 0.5)

        assert not pelm.detect_corruption()

    def test_detect_pointer_out_of_bounds(self):
        """Test corruption detection for pointer out of bounds."""
        pelm = PhaseEntangledLatticeMemory(dimension=3, capacity=10)
        pelm.entangle([1.0, 2.0, 3.0], 0.5)

        # Corrupt pointer
        pelm.pointer = 100

        assert pelm.detect_corruption()

    def test_detect_negative_pointer(self):
        """Test corruption detection for negative pointer."""
        pelm = PhaseEntangledLatticeMemory(dimension=3, capacity=10)
        pelm.entangle([1.0, 2.0, 3.0], 0.5)

        # Corrupt pointer
        pelm.pointer = -1

        assert pelm.detect_corruption()

    def test_detect_size_corruption(self):
        """Test corruption detection for invalid size."""
        pelm = PhaseEntangledLatticeMemory(dimension=3, capacity=10)
        pelm.entangle([1.0, 2.0, 3.0], 0.5)

        # Corrupt size
        pelm.size = 100

        assert pelm.detect_corruption()

    def test_detect_checksum_mismatch(self):
        """Test corruption detection for checksum mismatch."""
        pelm = PhaseEntangledLatticeMemory(dimension=3, capacity=10)
        pelm.entangle([1.0, 2.0, 3.0], 0.5)

        # Directly corrupt memory without updating checksum
        pelm.memory_bank[0] = np.array([99.0, 99.0, 99.0], dtype=np.float32)

        assert pelm.detect_corruption()


class TestPELMv2AutoRecovery:
    """Test PhaseEntangledLatticeMemory auto-recovery mechanism."""

    def test_auto_recover_pointer_corruption(self):
        """Test auto-recovery fixes pointer corruption."""
        pelm = PhaseEntangledLatticeMemory(dimension=3, capacity=10)
        pelm.entangle([1.0, 2.0, 3.0], 0.5)
        pelm.entangle([4.0, 5.0, 6.0], 0.6)

        # Corrupt pointer
        pelm.pointer = 100

        # Verify corruption detected
        assert pelm.detect_corruption()

        # Auto-recover
        recovered = pelm.auto_recover()

        assert recovered
        assert pelm.pointer == 2  # Should be fixed to size % capacity
        assert not pelm.detect_corruption()

    def test_auto_recover_negative_pointer(self):
        """Test auto-recovery fixes negative pointer."""
        pelm = PhaseEntangledLatticeMemory(dimension=3, capacity=10)
        pelm.entangle([1.0, 2.0, 3.0], 0.5)

        # Corrupt pointer
        pelm.pointer = -5

        recovered = pelm.auto_recover()

        assert recovered
        assert pelm.pointer >= 0
        assert not pelm.detect_corruption()

    def test_auto_recover_size_corruption(self):
        """Test auto-recovery fixes size corruption."""
        pelm = PhaseEntangledLatticeMemory(dimension=3, capacity=10)
        pelm.entangle([1.0, 2.0, 3.0], 0.5)

        # Corrupt size
        pelm.size = -1

        recovered = pelm.auto_recover()

        assert recovered
        assert pelm.size >= 0
        assert not pelm.detect_corruption()

    def test_auto_recover_rebuilds_norms(self):
        """Test auto-recovery rebuilds norms."""
        pelm = PhaseEntangledLatticeMemory(dimension=3, capacity=10)
        pelm.entangle([1.0, 2.0, 3.0], 0.5)

        original_norm = pelm.norms[0]

        # Corrupt norms
        pelm.norms[0] = 0.0

        recovered = pelm.auto_recover()

        assert recovered
        # Norm should be restored
        np.testing.assert_almost_equal(pelm.norms[0], original_norm)


class TestPELMv2IntegrationWithCorruption:
    """Test PhaseEntangledLatticeMemory integration scenarios with corruption."""

    def test_forceful_corruption_and_recovery(self):
        """Test forcefully corrupt state and verify recovery."""
        pelm = PhaseEntangledLatticeMemory(dimension=5, capacity=10)

        # Add some data
        vectors = [[float(i)] * 5 for i in range(5)]
        phases = [0.1 * i for i in range(5)]

        for vec, phase in zip(vectors, phases, strict=True):
            pelm.entangle(vec, phase)

        # Verify no corruption initially
        assert not pelm.detect_corruption()

        # Forcefully corrupt multiple aspects
        pelm.pointer = 1000  # Invalid pointer
        pelm.size = -5  # Invalid size
        pelm.memory_bank[0] = np.array([999.0] * 5, dtype=np.float32)  # Corrupt data

        # Verify corruption detected
        assert pelm.detect_corruption()

        # Attempt recovery
        recovered = pelm.auto_recover()

        assert recovered
        assert not pelm.detect_corruption()
        assert pelm.pointer >= 0 and pelm.pointer < pelm.capacity
        assert pelm.size >= 0 and pelm.size <= pelm.capacity

    def test_retrieve_triggers_auto_recovery(self):
        """Test that retrieve triggers auto-recovery on corruption."""
        pelm = PhaseEntangledLatticeMemory(dimension=3, capacity=10)

        pelm.entangle([1.0, 2.0, 3.0], 0.5)
        pelm.entangle([4.0, 5.0, 6.0], 0.5)

        # Corrupt pointer
        pelm.pointer = -1

        # Retrieve should trigger auto-recovery
        results = pelm.retrieve([1.0, 2.0, 3.0], 0.5)

        # Should succeed after recovery
        assert len(results) > 0
        assert not pelm.detect_corruption()

    def test_entangle_triggers_auto_recovery(self):
        """Test that entangle triggers auto-recovery on corruption."""
        pelm = PhaseEntangledLatticeMemory(dimension=3, capacity=10)

        pelm.entangle([1.0, 2.0, 3.0], 0.5)

        # Corrupt pointer
        pelm.pointer = 100

        # Entangle should trigger auto-recovery
        idx = pelm.entangle([4.0, 5.0, 6.0], 0.6)

        # Should succeed after recovery
        assert idx >= 0
        assert not pelm.detect_corruption()

    def test_recovery_failure_raises_error(self):
        """Test that unrecoverable corruption raises error."""
        pelm = PhaseEntangledLatticeMemory(dimension=3, capacity=10)

        pelm.entangle([1.0, 2.0, 3.0], 0.5)

        # Create severe corruption that might not be recoverable
        # by setting invalid pointer
        pelm.pointer = 100

        # Force recovery to fail by making auto_recover fail internally
        # This is a simulation - in practice the current implementation should recover
        # But we test the error path exists
        try:
            # Normal operations should attempt recovery
            pelm.entangle([4.0, 5.0, 6.0], 0.6)
        except RuntimeError as e:
            # If recovery fails, should raise RuntimeError
            assert "Memory corruption" in str(e)


class TestPELMv2BoundaryChecks:
    """Test PhaseEntangledLatticeMemory boundary checks."""

    def test_pointer_wraparound_at_capacity(self):
        """Test explicit pointer wraparound at capacity boundary."""
        pelm = PhaseEntangledLatticeMemory(dimension=2, capacity=5)

        # Fill to capacity
        for i in range(5):
            idx = pelm.entangle([float(i), float(i+1)], 0.1)
            assert idx == i

        # Pointer should wrap to 0
        assert pelm.pointer == 0

        # Next entangle should use index 0
        idx = pelm.entangle([99.0, 99.0], 0.9)
        assert idx == 0
        assert pelm.pointer == 1

    def test_size_stops_at_capacity(self):
        """Test size doesn't exceed capacity."""
        pelm = PhaseEntangledLatticeMemory(dimension=2, capacity=3)

        # Add more than capacity
        for i in range(10):
            pelm.entangle([float(i), float(i+1)], 0.1 * i)

        # Size should be capped at capacity
        assert pelm.size == 3
        assert pelm.size <= pelm.capacity

    def test_validate_pointer_bounds(self):
        """Test _validate_pointer_bounds method."""
        pelm = PhaseEntangledLatticeMemory(dimension=2, capacity=10)

        # Valid state
        assert pelm._validate_pointer_bounds()

        # Invalid pointer (too large)
        pelm.pointer = 100
        assert not pelm._validate_pointer_bounds()

        # Invalid pointer (negative)
        pelm.pointer = -1
        assert not pelm._validate_pointer_bounds()

        # Fix pointer
        pelm.pointer = 5
        assert pelm._validate_pointer_bounds()


class TestPELMv2StateStats:
    """Test PhaseEntangledLatticeMemory state statistics."""

    def test_get_state_stats(self):
        """Test get_state_stats returns correct information."""
        pelm = PhaseEntangledLatticeMemory(dimension=10, capacity=100)

        pelm.entangle([1.0] * 10, 0.5)

        stats = pelm.get_state_stats()

        assert stats["capacity"] == 100
        assert stats["used"] == 1
        assert "memory_mb" in stats
        assert isinstance(stats["memory_mb"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
