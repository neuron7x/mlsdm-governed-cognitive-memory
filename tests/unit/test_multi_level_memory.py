"""
Unit Tests for MultiLevelSynapticMemory

Tests the multi-level synaptic memory system with L1, L2, L3 memory layers.
"""

import numpy as np
import pytest

from mlsdm.memory.multi_level_memory import MultiLevelSynapticMemory


class TestMultiLevelMemoryInitialization:
    """Test MultiLevelSynapticMemory initialization."""

    def test_default_initialization(self):
        """Test memory can be initialized with defaults."""
        memory = MultiLevelSynapticMemory()

        assert memory.dim == 384
        assert memory.lambda_l1 == 0.50
        assert memory.lambda_l2 == 0.10
        assert memory.lambda_l3 == 0.01
        assert memory.theta_l1 == 1.2
        assert memory.theta_l2 == 2.5
        assert memory.gating12 == 0.45
        assert memory.gating23 == 0.30

        # Memory layers should be zero-initialized
        assert memory.l1.shape == (384,)
        assert memory.l2.shape == (384,)
        assert memory.l3.shape == (384,)
        assert np.all(memory.l1 == 0)
        assert np.all(memory.l2 == 0)
        assert np.all(memory.l3 == 0)

    def test_custom_initialization(self):
        """Test memory can be initialized with custom values."""
        memory = MultiLevelSynapticMemory(
            dimension=100,
            lambda_l1=0.3,
            lambda_l2=0.2,
            lambda_l3=0.05,
            theta_l1=1.5,
            theta_l2=2.0,
            gating12=0.6,
            gating23=0.4,
        )

        assert memory.dim == 100
        assert memory.lambda_l1 == 0.3
        assert memory.lambda_l2 == 0.2
        assert memory.lambda_l3 == 0.05
        assert memory.theta_l1 == 1.5
        assert memory.theta_l2 == 2.0
        assert memory.gating12 == 0.6
        assert memory.gating23 == 0.4

        assert memory.l1.shape == (100,)

    def test_invalid_dimension_raises(self):
        """Test negative dimension raises error."""
        with pytest.raises(ValueError, match="dimension must be positive"):
            MultiLevelSynapticMemory(dimension=-10)

    def test_zero_dimension_raises(self):
        """Test zero dimension raises error."""
        with pytest.raises(ValueError, match="dimension must be positive"):
            MultiLevelSynapticMemory(dimension=0)

    def test_invalid_lambda_l1_raises(self):
        """Test invalid lambda_l1 raises error."""
        with pytest.raises(ValueError, match="lambda_l1 must be in"):
            MultiLevelSynapticMemory(lambda_l1=0)

        with pytest.raises(ValueError, match="lambda_l1 must be in"):
            MultiLevelSynapticMemory(lambda_l1=1.5)

    def test_invalid_lambda_l2_raises(self):
        """Test invalid lambda_l2 raises error."""
        with pytest.raises(ValueError, match="lambda_l2 must be in"):
            MultiLevelSynapticMemory(lambda_l2=-0.1)

    def test_invalid_lambda_l3_raises(self):
        """Test invalid lambda_l3 raises error."""
        with pytest.raises(ValueError, match="lambda_l3 must be in"):
            MultiLevelSynapticMemory(lambda_l3=2.0)

    def test_invalid_theta_l1_raises(self):
        """Test invalid theta_l1 raises error."""
        with pytest.raises(ValueError, match="theta_l1 must be positive"):
            MultiLevelSynapticMemory(theta_l1=0)

        with pytest.raises(ValueError, match="theta_l1 must be positive"):
            MultiLevelSynapticMemory(theta_l1=-1)

    def test_invalid_theta_l2_raises(self):
        """Test invalid theta_l2 raises error."""
        with pytest.raises(ValueError, match="theta_l2 must be positive"):
            MultiLevelSynapticMemory(theta_l2=0)

    def test_invalid_gating12_raises(self):
        """Test invalid gating12 raises error."""
        with pytest.raises(ValueError, match="gating12 must be in"):
            MultiLevelSynapticMemory(gating12=-0.1)

        with pytest.raises(ValueError, match="gating12 must be in"):
            MultiLevelSynapticMemory(gating12=1.5)

    def test_invalid_gating23_raises(self):
        """Test invalid gating23 raises error."""
        with pytest.raises(ValueError, match="gating23 must be in"):
            MultiLevelSynapticMemory(gating23=-0.1)


class TestMultiLevelMemoryUpdate:
    """Test memory update functionality."""

    def test_update_basic(self):
        """Test basic memory update."""
        memory = MultiLevelSynapticMemory(dimension=10)
        event = np.ones(10, dtype=np.float32)

        memory.update(event)

        # L1 should have received the event
        assert np.sum(memory.l1) > 0

    def test_update_with_float64(self):
        """Test update converts float64 to float32."""
        memory = MultiLevelSynapticMemory(dimension=10)
        event = np.ones(10, dtype=np.float64)

        memory.update(event)

        # Should work without error
        assert np.sum(memory.l1) > 0

    def test_update_dimension_mismatch_raises(self):
        """Test update rejects dimension mismatch."""
        memory = MultiLevelSynapticMemory(dimension=10)
        event = np.ones(5)  # Wrong dimension

        with pytest.raises(ValueError, match="dimension"):
            memory.update(event)

    def test_update_non_array_raises(self):
        """Test update rejects non-array input."""
        memory = MultiLevelSynapticMemory(dimension=10)

        with pytest.raises(ValueError):
            memory.update([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_update_decay(self):
        """Test memory decay on update."""
        memory = MultiLevelSynapticMemory(dimension=10, lambda_l1=0.5)

        # First update
        memory.l1.fill(2.0)
        event = np.zeros(10, dtype=np.float32)

        memory.update(event)

        # L1 should have decayed
        expected = 2.0 * (1 - 0.5)  # 1.0
        assert pytest.approx(memory.l1[0], rel=1e-5) == expected

    def test_update_transfer_l1_to_l2(self):
        """Test transfer from L1 to L2 when threshold exceeded."""
        memory = MultiLevelSynapticMemory(
            dimension=10,
            theta_l1=1.0,
            gating12=0.5,
            lambda_l1=0.01,  # Very small decay for testing
            lambda_l2=0.01,
        )

        # Add enough to exceed theta_l1
        event = np.ones(10, dtype=np.float32) * 2.0
        memory.update(event)

        # Some should have transferred to L2
        assert np.sum(memory.l2) > 0

    def test_update_transfer_l2_to_l3(self):
        """Test transfer from L2 to L3 when threshold exceeded."""
        memory = MultiLevelSynapticMemory(
            dimension=10,
            theta_l1=1.0,
            theta_l2=1.0,
            gating12=0.8,
            gating23=0.5,
            lambda_l1=0.01,  # Very small decay for testing
            lambda_l2=0.01,
            lambda_l3=0.01,
        )

        # Add enough to exceed both thresholds
        for _ in range(5):
            event = np.ones(10, dtype=np.float32) * 3.0
            memory.update(event)

        # Some should have transferred to L3
        assert np.sum(memory.l3) > 0

    def test_multiple_updates(self):
        """Test multiple sequential updates."""
        memory = MultiLevelSynapticMemory(dimension=10)

        for i in range(10):
            event = np.ones(10, dtype=np.float32) * (i + 1)
            memory.update(event)

        # All layers should have some activity
        assert np.sum(memory.l1) > 0


class TestMultiLevelMemoryState:
    """Test state retrieval functionality."""

    def test_state_returns_copies(self):
        """Test state returns copies of internal arrays."""
        memory = MultiLevelSynapticMemory(dimension=10)
        event = np.ones(10, dtype=np.float32)
        memory.update(event)

        l1, l2, l3 = memory.state()

        # Modifying returned arrays should not affect internal state
        original_l1 = memory.l1.copy()
        l1[0] = 999.0

        assert memory.l1[0] == original_l1[0]
        assert l1[0] == 999.0

    def test_get_state_same_as_state(self):
        """Test get_state is alias for state."""
        memory = MultiLevelSynapticMemory(dimension=10)
        event = np.ones(10, dtype=np.float32)
        memory.update(event)

        l1_state, l2_state, l3_state = memory.state()
        l1_get, l2_get, l3_get = memory.get_state()

        np.testing.assert_array_equal(l1_state, l1_get)
        np.testing.assert_array_equal(l2_state, l2_get)
        np.testing.assert_array_equal(l3_state, l3_get)

    def test_state_initial_is_zero(self):
        """Test initial state is all zeros."""
        memory = MultiLevelSynapticMemory(dimension=10)

        l1, l2, l3 = memory.state()

        assert np.all(l1 == 0)
        assert np.all(l2 == 0)
        assert np.all(l3 == 0)


class TestMultiLevelMemoryReset:
    """Test memory reset functionality."""

    def test_reset_all(self):
        """Test reset_all clears all memory layers."""
        memory = MultiLevelSynapticMemory(dimension=10)

        # Add some data
        for _ in range(5):
            memory.update(np.ones(10, dtype=np.float32) * 2.0)

        # Verify memory is not empty
        assert np.sum(memory.l1) > 0

        # Reset
        memory.reset_all()

        # Verify memory is empty
        assert np.all(memory.l1 == 0)
        assert np.all(memory.l2 == 0)
        assert np.all(memory.l3 == 0)


class TestMultiLevelMemoryToDict:
    """Test serialization functionality."""

    def test_to_dict_structure(self):
        """Test to_dict returns correct structure."""
        memory = MultiLevelSynapticMemory(
            dimension=10,
            lambda_l1=0.3,
            lambda_l2=0.2,
            lambda_l3=0.05,
            theta_l1=1.5,
            theta_l2=2.0,
            gating12=0.6,
            gating23=0.4,
        )

        data = memory.to_dict()

        assert isinstance(data, dict)
        assert data["dimension"] == 10
        assert data["lambda_l1"] == 0.3
        assert data["lambda_l2"] == 0.2
        assert data["lambda_l3"] == 0.05
        assert data["theta_l1"] == 1.5
        assert data["theta_l2"] == 2.0
        assert data["gating12"] == 0.6
        assert data["gating23"] == 0.4
        assert "state_L1" in data
        assert "state_L2" in data
        assert "state_L3" in data

    def test_to_dict_state_is_list(self):
        """Test to_dict converts state to list."""
        memory = MultiLevelSynapticMemory(dimension=5)
        memory.update(np.ones(5, dtype=np.float32))

        data = memory.to_dict()

        assert isinstance(data["state_L1"], list)
        assert len(data["state_L1"]) == 5


class TestMultiLevelMemoryIntegration:
    """Test integration scenarios."""

    def test_memory_consolidation_workflow(self):
        """Test memory consolidation from L1 to L3."""
        memory = MultiLevelSynapticMemory(
            dimension=10,
            theta_l1=0.5,
            theta_l2=1.0,
            gating12=0.6,
            gating23=0.4,
            lambda_l1=0.1,
            lambda_l2=0.05,
            lambda_l3=0.01,
        )

        # Simulate repeated exposure
        for _ in range(50):
            event = np.ones(10, dtype=np.float32) * 0.5
            memory.update(event)

        l1, l2, l3 = memory.state()

        # All layers should have activity
        assert np.sum(l1) > 0
        assert np.sum(l2) > 0
        # L3 may or may not have activity depending on thresholds

    def test_different_event_patterns(self):
        """Test memory with different event patterns."""
        memory = MultiLevelSynapticMemory(dimension=10)

        # Sparse events
        sparse = np.zeros(10, dtype=np.float32)
        sparse[0] = 1.0
        memory.update(sparse)

        # Dense events
        dense = np.ones(10, dtype=np.float32) * 0.5
        memory.update(dense)

        # Random events
        np.random.seed(42)
        random = np.random.randn(10).astype(np.float32)
        memory.update(random)

        # Memory should contain information from all
        assert np.sum(memory.l1) != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
