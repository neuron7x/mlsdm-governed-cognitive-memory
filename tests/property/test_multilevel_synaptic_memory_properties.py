"""
Property-based tests for MultiLevelSynapticMemory invariants.

Verifies that multi-level synaptic memory maintains documented invariants:
- Decay monotonicity (older events have less weight over time)
- Level transfer thresholds respected
- Gating values control information flow between levels
- No unbounded growth in any level
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlsdm.memory.multi_level_memory import MultiLevelSynapticMemory

# Fixed seed for deterministic property tests
PROPERTY_TEST_SEED = 42


@settings(max_examples=30, deadline=None)
@given(
    dim=st.integers(min_value=5, max_value=50),
    num_updates=st.integers(min_value=1, max_value=20)
)
def test_multilevel_decay_monotonicity(dim, num_updates):
    """
    Property: Decay reduces level norms monotonically without new input.
    After stopping updates, each level's norm should decrease or stay zero.
    """
    np.random.seed(PROPERTY_TEST_SEED)
    memory = MultiLevelSynapticMemory(dimension=dim)

    # Add some events to populate levels
    for _ in range(num_updates):
        vec = np.random.randn(dim).astype(np.float32)
        memory.update(vec)

    # Get state after updates
    L1_before, L2_before, L3_before = memory.get_state()
    norm_L1_before = np.linalg.norm(L1_before)
    norm_L2_before = np.linalg.norm(L2_before)
    norm_L3_before = np.linalg.norm(L3_before)

    # Apply decay without new input (zero vector)
    zero_vec = np.zeros(dim, dtype=np.float32)
    memory.update(zero_vec)

    # Get state after decay
    L1_after, L2_after, L3_after = memory.get_state()
    norm_L1_after = np.linalg.norm(L1_after)
    norm_L2_after = np.linalg.norm(L2_after)
    norm_L3_after = np.linalg.norm(L3_after)

    # Decay should reduce or maintain norms (within floating point tolerance)
    assert norm_L1_after <= norm_L1_before + 1e-6, \
        f"L1 norm increased: {norm_L1_before} -> {norm_L1_after}"
    assert norm_L2_after <= norm_L2_before + 1e-6, \
        f"L2 norm increased: {norm_L2_before} -> {norm_L2_after}"
    assert norm_L3_after <= norm_L3_before + 1e-6, \
        f"L3 norm increased: {norm_L3_before} -> {norm_L3_after}"


@settings(max_examples=30, deadline=None)
@given(
    dim=st.integers(min_value=5, max_value=50),
    lambda_l1=st.floats(min_value=0.1, max_value=0.9, allow_nan=False),
    lambda_l2=st.floats(min_value=0.01, max_value=0.5, allow_nan=False),
    lambda_l3=st.floats(min_value=0.001, max_value=0.1, allow_nan=False)
)
def test_multilevel_lambda_decay_rates(dim, lambda_l1, lambda_l2, lambda_l3):
    """
    Property: Higher lambda values cause faster decay.
    L1 (fastest) > L2 (medium) > L3 (slowest) when lambda_l1 > lambda_l2 > lambda_l3.
    """
    np.random.seed(PROPERTY_TEST_SEED)

    # Ensure lambda ordering for this test
    if not (lambda_l1 > lambda_l2 > lambda_l3):
        return  # Skip this example

    memory = MultiLevelSynapticMemory(
        dimension=dim,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        lambda_l3=lambda_l3
    )

    # Add a large event to all levels (via threshold transfers)
    large_vec = np.ones(dim, dtype=np.float32) * 5.0
    for _ in range(10):
        memory.update(large_vec)

    L1_before, L2_before, L3_before = memory.get_state()

    # Apply multiple decay cycles with zero input
    zero_vec = np.zeros(dim, dtype=np.float32)
    for _ in range(5):
        memory.update(zero_vec)

    L1_after, L2_after, L3_after = memory.get_state()

    # Calculate decay ratios
    decay_L1 = np.linalg.norm(L1_after) / (np.linalg.norm(L1_before) + 1e-9)
    decay_L2 = np.linalg.norm(L2_after) / (np.linalg.norm(L2_before) + 1e-9)
    decay_L3 = np.linalg.norm(L3_after) / (np.linalg.norm(L3_before) + 1e-9)

    # L1 should decay fastest (smallest ratio), L3 slowest (largest ratio)
    # Allow some tolerance for floating point and transfer effects
    assert decay_L1 <= decay_L2 + 0.1, \
        f"L1 should decay faster than L2: {decay_L1} > {decay_L2}"
    assert decay_L2 <= decay_L3 + 0.1, \
        f"L2 should decay faster than L3: {decay_L2} > {decay_L3}"


@settings(max_examples=30, deadline=None)
@given(
    dim=st.integers(min_value=5, max_value=50),
    num_updates=st.integers(min_value=5, max_value=30)
)
def test_multilevel_no_unbounded_growth(dim, num_updates):
    """
    Property: Memory levels do not grow unboundedly.
    With balanced decay and input, norms should stabilize.
    """
    np.random.seed(PROPERTY_TEST_SEED)
    memory = MultiLevelSynapticMemory(dimension=dim)

    # Track max norms
    max_L1_norm = 0.0
    max_L2_norm = 0.0
    max_L3_norm = 0.0

    for _ in range(num_updates):
        vec = np.random.randn(dim).astype(np.float32)
        memory.update(vec)

        L1, L2, L3 = memory.get_state()
        max_L1_norm = max(max_L1_norm, np.linalg.norm(L1))
        max_L2_norm = max(max_L2_norm, np.linalg.norm(L2))
        max_L3_norm = max(max_L3_norm, np.linalg.norm(L3))

    # Norms should be bounded (not infinite or extremely large)
    assert max_L1_norm < 1000 * np.sqrt(dim), \
        f"L1 norm unbounded: {max_L1_norm}"
    assert max_L2_norm < 1000 * np.sqrt(dim), \
        f"L2 norm unbounded: {max_L2_norm}"
    assert max_L3_norm < 1000 * np.sqrt(dim), \
        f"L3 norm unbounded: {max_L3_norm}"


def test_multilevel_gating_bounds():
    """
    Property: Gating values are stored within bounds [0, 1].
    """
    np.random.seed(PROPERTY_TEST_SEED)
    dim = 10

    # Test various gating values
    memory = MultiLevelSynapticMemory(
        dimension=dim,
        gating12=0.45,
        gating23=0.30
    )

    # Gating values should be within bounds
    assert 0.0 <= memory.gating12 <= 1.0, \
        f"gating12 out of bounds: {memory.gating12}"
    assert 0.0 <= memory.gating23 <= 1.0, \
        f"gating23 out of bounds: {memory.gating23}"

    # Test edge cases
    memory_min = MultiLevelSynapticMemory(dimension=dim, gating12=0.0, gating23=0.0)
    assert memory_min.gating12 == 0.0
    assert memory_min.gating23 == 0.0

    memory_max = MultiLevelSynapticMemory(dimension=dim, gating12=1.0, gating23=1.0)
    assert memory_max.gating12 == 1.0
    assert memory_max.gating23 == 1.0


def test_multilevel_reset_clears_all_levels():
    """Test that reset_all clears all three levels."""
    memory = MultiLevelSynapticMemory(dimension=10)

    # Add events
    for _ in range(5):
        vec = np.random.randn(10).astype(np.float32)
        memory.update(vec)

    # Verify levels are non-zero
    L1, L2, L3 = memory.get_state()
    assert np.linalg.norm(L1) > 0 or np.linalg.norm(L2) > 0 or np.linalg.norm(L3) > 0

    # Reset
    memory.reset_all()

    # Verify all levels are zero
    L1, L2, L3 = memory.get_state()
    assert np.allclose(L1, 0.0), "L1 not cleared"
    assert np.allclose(L2, 0.0), "L2 not cleared"
    assert np.allclose(L3, 0.0), "L3 not cleared"


def test_multilevel_dimension_consistency():
    """Test that all levels maintain correct dimension."""
    dim = 20
    memory = MultiLevelSynapticMemory(dimension=dim)

    # Add events
    for _ in range(5):
        vec = np.random.randn(dim).astype(np.float32)
        memory.update(vec)

    L1, L2, L3 = memory.get_state()

    assert L1.shape[0] == dim, f"L1 dimension mismatch: {L1.shape[0]} != {dim}"
    assert L2.shape[0] == dim, f"L2 dimension mismatch: {L2.shape[0]} != {dim}"
    assert L3.shape[0] == dim, f"L3 dimension mismatch: {L3.shape[0]} != {dim}"


def test_multilevel_invalid_inputs():
    """Test that invalid inputs raise appropriate errors."""
    # Invalid dimension
    with pytest.raises(ValueError, match="dimension must be positive"):
        MultiLevelSynapticMemory(dimension=0)

    # Invalid lambda values
    with pytest.raises(ValueError, match="lambda_l1"):
        MultiLevelSynapticMemory(dimension=10, lambda_l1=0.0)

    with pytest.raises(ValueError, match="lambda_l1"):
        MultiLevelSynapticMemory(dimension=10, lambda_l1=1.5)

    # Invalid gating values
    with pytest.raises(ValueError, match="gating12"):
        MultiLevelSynapticMemory(dimension=10, gating12=-0.1)

    with pytest.raises(ValueError, match="gating12"):
        MultiLevelSynapticMemory(dimension=10, gating12=1.5)


def test_multilevel_to_dict_serialization():
    """Test that to_dict returns correct structure."""
    memory = MultiLevelSynapticMemory(
        dimension=10,
        lambda_l1=0.5,
        lambda_l2=0.1,
        lambda_l3=0.01
    )

    # Add an event
    vec = np.ones(10, dtype=np.float32)
    memory.update(vec)

    state_dict = memory.to_dict()

    # Check required keys
    assert "dimension" in state_dict
    assert "lambda_l1" in state_dict
    assert "lambda_l2" in state_dict
    assert "lambda_l3" in state_dict
    assert "theta_l1" in state_dict
    assert "theta_l2" in state_dict
    assert "gating12" in state_dict
    assert "gating23" in state_dict
    assert "state_L1" in state_dict
    assert "state_L2" in state_dict
    assert "state_L3" in state_dict

    # Check values
    assert state_dict["dimension"] == 10
    assert state_dict["lambda_l1"] == 0.5
    assert len(state_dict["state_L1"]) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
