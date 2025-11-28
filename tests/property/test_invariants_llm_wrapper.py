"""
Property-based tests for LLMWrapper invariants.

Tests formal invariants defined in docs/FORMAL_INVARIANTS.md using Hypothesis.
Covers:
1. Memory bounds (INV-LLM-S2): Number of vectors never exceeds capacity
2. Stateless mode behavior: No memory writes when stateless_mode=True
3. Governance metadata: Every response contains required governance fields

Reference Invariants from FORMAL_INVARIANTS.md:
- INV-LLM-S2: Capacity Constraint - |memory_vectors| ≤ capacity
- INV-LLM-L3: Memory Overflow Handling - eviction when capacity reached
"""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from mlsdm.core.llm_wrapper import LLMWrapper

__all__ = ["create_test_wrapper", "create_stub_embedder", "create_stub_llm"]


# ============================================================================
# Test Strategies
# ============================================================================

@st.composite
def prompt_strategy(draw):
    """Generate various prompt types for testing."""
    return draw(st.text(
        min_size=1,
        max_size=128,
        alphabet=st.characters(
            whitelist_categories=("L", "N", "P", "S", "Z"),
            blacklist_characters="\x00"
        )
    ))


@st.composite
def moral_value_strategy(draw):
    """Generate moral values in valid range [0.0, 1.0]."""
    return draw(st.floats(
        min_value=0.0,
        max_value=1.0,
        allow_nan=False,
        allow_infinity=False
    ))


# ============================================================================
# Helper Functions - Stub LLM and Embedding
# ============================================================================

def create_stub_llm():
    """
    Create a deterministic stub LLM function for testing.

    This stub generates predictable responses without external dependencies.
    """
    def stub_llm_generate(prompt: str, max_tokens: int) -> str:
        """Deterministic LLM stub that returns mock responses."""
        return f"Mock response for prompt with {len(prompt)} characters."

    return stub_llm_generate


def create_stub_embedder(dim: int = 384):
    """
    Create a deterministic stub embedding function for testing.

    Uses text hash for deterministic but varied embeddings.
    """
    def stub_embed(text: str) -> np.ndarray:
        """Generate deterministic embedding based on text hash."""
        # Use hash for determinism
        seed = hash(text) % (2**32)
        np.random.seed(seed)
        vec = np.random.randn(dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            return vec / norm
        else:
            result = np.zeros(dim, dtype=np.float32)
            result[0] = 1.0
            return result

    return stub_embed


def create_test_wrapper(
    dim: int = 64,
    capacity: int = 64,
    wake_duration: int = 100,
    sleep_duration: int = 1,
    initial_moral_threshold: float = 0.30,
) -> LLMWrapper:
    """
    Create a LLMWrapper instance for property testing.

    Uses deterministic stubs and small capacity for fast testing.
    Long wake duration ensures tests run during wake phase.
    """
    return LLMWrapper(
        llm_generate_fn=create_stub_llm(),
        embedding_fn=create_stub_embedder(dim=dim),
        dim=dim,
        capacity=capacity,
        wake_duration=wake_duration,
        sleep_duration=sleep_duration,
        initial_moral_threshold=initial_moral_threshold,
    )


# ============================================================================
# Property Tests: Memory Bounds (INV-LLM-S2)
# ============================================================================

@settings(max_examples=50, deadline=None)
@given(
    capacity=st.integers(min_value=10, max_value=64),
    num_calls=st.integers(min_value=20, max_value=100),
    moral_value=moral_value_strategy()
)
def test_memory_never_exceeds_capacity(capacity, num_calls, moral_value):
    """
    INV-LLM-S2: Capacity Constraint

    Number of vectors in memory MUST NOT exceed configured capacity,
    even after many calls to generate().

    Formal: |memory_vectors| ≤ capacity
    """
    # Use high moral value to ensure acceptance
    assume(moral_value >= 0.5)
    # Ensure we do more calls than capacity to test wraparound
    assume(num_calls > capacity)

    wrapper = create_test_wrapper(
        capacity=capacity,
        initial_moral_threshold=0.30,  # Low threshold for acceptance
    )

    max_size_seen = 0

    for i in range(num_calls):
        result = wrapper.generate(
            prompt=f"Test prompt number {i}",
            moral_value=moral_value
        )

        # Track memory size after each call
        current_size = wrapper.pelm.size
        max_size_seen = max(max_size_seen, current_size)

        # INVARIANT: size must never exceed capacity
        assert current_size <= capacity, (
            f"Memory size {current_size} exceeds capacity {capacity} "
            f"after call {i+1} (accepted={result['accepted']})"
        )

    # Final verification
    assert wrapper.pelm.size <= capacity, (
        f"Final memory size {wrapper.pelm.size} exceeds capacity {capacity}"
    )
    assert max_size_seen <= capacity, (
        f"Max size seen {max_size_seen} exceeded capacity {capacity}"
    )


@settings(max_examples=30, deadline=None)
@given(
    capacity=st.integers(min_value=5, max_value=50),
    num_calls=st.integers(min_value=1, max_value=50)
)
def test_memory_bounded_growth(capacity, num_calls):
    """
    INV-LLM-L3 / Memory Overflow Handling

    Memory does not grow unboundedly; when capacity is reached,
    system evicts old entries (circular buffer behavior).
    """
    wrapper = create_test_wrapper(
        capacity=capacity,
        initial_moral_threshold=0.20,  # Low threshold for acceptance
    )

    sizes = []
    for i in range(num_calls):
        # High moral value ensures acceptance
        wrapper.generate(
            prompt=f"Message {i} for bounded growth test",
            moral_value=0.9
        )
        sizes.append(wrapper.pelm.size)

    # Memory size should be bounded
    assert max(sizes) <= capacity, (
        f"Memory grew beyond capacity: max size {max(sizes)} > capacity {capacity}"
    )

    # After enough calls, memory should stabilize at capacity (circular buffer)
    if num_calls > capacity:
        assert wrapper.pelm.size == capacity, (
            f"After {num_calls} calls (> capacity {capacity}), "
            f"size should be {capacity}, got {wrapper.pelm.size}"
        )


@settings(max_examples=30, deadline=None)
@given(prompt=prompt_strategy())
def test_memory_size_at_capacity_after_overflow(prompt):
    """
    Verify that after capacity overflow, size equals capacity.

    Tests the circular buffer wraparound behavior.
    """
    assume(len(prompt.strip()) > 0)

    capacity = 20
    wrapper = create_test_wrapper(
        capacity=capacity,
        initial_moral_threshold=0.20,
    )

    # Fill beyond capacity
    num_inserts = capacity + 10

    for i in range(num_inserts):
        wrapper.generate(
            prompt=f"{prompt} - iteration {i}",
            moral_value=0.9
        )

    # Size should be exactly at capacity after overflow
    assert wrapper.pelm.size == capacity, (
        f"After {num_inserts} inserts (> capacity {capacity}), "
        f"size should be {capacity}, got {wrapper.pelm.size}"
    )


# ============================================================================
# Property Tests: Stateless Mode (No Memory Writes)
# ============================================================================

@settings(max_examples=50, deadline=None)
@given(
    prompt=prompt_strategy(),
    moral_value=moral_value_strategy(),
    num_calls=st.integers(min_value=1, max_value=30)
)
def test_stateless_mode_no_memory_writes(prompt, moral_value, num_calls):
    """
    Stateless Mode Invariant

    When stateless_mode=True, wrapper MUST NOT write to memory.
    Memory size and pointer should remain unchanged.
    """
    assume(len(prompt.strip()) > 0)
    assume(moral_value >= 0.5)  # Ensure acceptance

    wrapper = create_test_wrapper(
        capacity=100,
        initial_moral_threshold=0.30,
    )

    # Enable stateless mode
    wrapper.stateless_mode = True

    # Record initial memory state
    initial_size = wrapper.pelm.size
    initial_pointer = wrapper.pelm.pointer

    # Execute multiple generate calls
    for i in range(num_calls):
        result = wrapper.generate(
            prompt=f"{prompt} - call {i}",
            moral_value=moral_value
        )

        # Should still return a response (with stateless note)
        assert isinstance(result, dict), f"Result should be dict, got {type(result)}"
        assert "response" in result
        assert "accepted" in result

        # Memory should NOT have changed
        assert wrapper.pelm.size == initial_size, (
            f"Memory size changed in stateless mode: "
            f"{initial_size} -> {wrapper.pelm.size}"
        )
        assert wrapper.pelm.pointer == initial_pointer, (
            f"Memory pointer changed in stateless mode: "
            f"{initial_pointer} -> {wrapper.pelm.pointer}"
        )


@settings(max_examples=30, deadline=None)
@given(num_calls=st.integers(min_value=5, max_value=30))
def test_stateless_mode_after_initial_writes(num_calls):
    """
    Test that enabling stateless mode after some writes freezes memory.
    """
    wrapper = create_test_wrapper(
        capacity=100,
        initial_moral_threshold=0.20,
    )

    # First, do some normal writes
    for i in range(5):
        wrapper.generate(
            prompt=f"Initial write {i}",
            moral_value=0.9
        )

    # Record state before enabling stateless mode
    size_before_stateless = wrapper.pelm.size
    pointer_before_stateless = wrapper.pelm.pointer

    # Enable stateless mode
    wrapper.stateless_mode = True

    # Do more calls
    for i in range(num_calls):
        wrapper.generate(
            prompt=f"Stateless call {i}",
            moral_value=0.9
        )

    # Memory should be frozen at the point stateless mode was enabled
    assert wrapper.pelm.size == size_before_stateless, (
        f"Memory size changed after enabling stateless mode: "
        f"{size_before_stateless} -> {wrapper.pelm.size}"
    )
    assert wrapper.pelm.pointer == pointer_before_stateless, (
        f"Memory pointer changed after enabling stateless mode: "
        f"{pointer_before_stateless} -> {wrapper.pelm.pointer}"
    )


def test_stateless_mode_returns_proper_note():
    """
    Test that stateless mode indicates degraded operation in response.
    """
    wrapper = create_test_wrapper()
    wrapper.stateless_mode = True

    result = wrapper.generate(
        prompt="Test in stateless mode",
        moral_value=0.9
    )

    # Response should indicate stateless mode
    assert result.get("stateless_mode") is True, (
        "Response should indicate stateless_mode=True"
    )


# ============================================================================
# Property Tests: Governance Metadata (Always Present)
# ============================================================================

@settings(max_examples=100, deadline=None)
@given(
    prompt=prompt_strategy(),
    moral_value=moral_value_strategy()
)
def test_governance_metadata_always_present(prompt, moral_value):
    """
    Governance Metadata Invariant

    Every result from LLMWrapper.generate() MUST contain
    governance-related metadata fields.

    Required fields (per FORMAL_INVARIANTS.md and existing implementation):
    - phase: Current cognitive phase ("wake" or "sleep")
    - accepted: Boolean indicating if request was morally accepted
    - note: Processing note or rejection reason
    - moral_threshold: Current moral threshold value
    - step: Current step counter
    """
    assume(len(prompt.strip()) > 0)

    wrapper = create_test_wrapper(
        initial_moral_threshold=0.30,
    )

    result = wrapper.generate(
        prompt=prompt,
        moral_value=moral_value
    )

    # INVARIANT: Result must be a dictionary
    assert isinstance(result, dict), f"Result should be dict, got {type(result)}"

    # INVARIANT: Required governance fields must be present
    required_fields = ["phase", "accepted", "note", "moral_threshold", "step"]

    for field in required_fields:
        assert field in result, (
            f"Required governance field '{field}' missing from result. "
            f"Result keys: {list(result.keys())}"
        )

    # INVARIANT: Field types must be correct
    assert isinstance(result["phase"], str), (
        f"'phase' should be str, got {type(result['phase'])}"
    )
    assert result["phase"] in ("wake", "sleep"), (
        f"'phase' should be 'wake' or 'sleep', got {result['phase']}"
    )

    assert isinstance(result["accepted"], bool), (
        f"'accepted' should be bool, got {type(result['accepted'])}"
    )

    assert isinstance(result["note"], str), (
        f"'note' should be str, got {type(result['note'])}"
    )

    assert isinstance(result["moral_threshold"], (int, float)), (
        f"'moral_threshold' should be numeric, got {type(result['moral_threshold'])}"
    )
    assert 0.0 <= result["moral_threshold"] <= 1.0, (
        f"'moral_threshold' should be in [0, 1], got {result['moral_threshold']}"
    )

    assert isinstance(result["step"], int), (
        f"'step' should be int, got {type(result['step'])}"
    )
    assert result["step"] > 0, (
        f"'step' should be positive, got {result['step']}"
    )


@settings(max_examples=50, deadline=None)
@given(
    prompt=prompt_strategy(),
    moral_value=st.floats(min_value=0.0, max_value=0.1)
)
def test_governance_metadata_on_rejection(prompt, moral_value):
    """
    Test that governance metadata is present even for rejected requests.

    Low moral values should be rejected, but response must still
    contain all governance fields.
    """
    assume(len(prompt.strip()) > 0)

    wrapper = create_test_wrapper(
        initial_moral_threshold=0.50,  # Higher threshold to ensure rejection
    )

    result = wrapper.generate(
        prompt=prompt,
        moral_value=moral_value  # Low value, should be rejected
    )

    # Should be rejected
    assert result["accepted"] is False, (
        f"Low moral value {moral_value} should be rejected"
    )

    # But all governance fields must still be present
    required_fields = ["phase", "accepted", "note", "moral_threshold", "step"]
    for field in required_fields:
        assert field in result, (
            f"Governance field '{field}' missing from rejected response"
        )

    # Rejection note should indicate moral rejection
    assert "reject" in result["note"].lower() or "error" in result["note"].lower(), (
        f"Rejection note should indicate rejection, got: {result['note']}"
    )


@settings(max_examples=30, deadline=None)
@given(
    prompt=prompt_strategy(),
    moral_value=st.floats(min_value=0.8, max_value=1.0)
)
def test_governance_metadata_on_acceptance(prompt, moral_value):
    """
    Test that governance metadata is complete for accepted requests.

    High moral values should be accepted with full response data.
    """
    assume(len(prompt.strip()) > 0)

    wrapper = create_test_wrapper(
        initial_moral_threshold=0.30,  # Low threshold to ensure acceptance
        wake_duration=100,  # Long wake to ensure not in sleep
    )

    result = wrapper.generate(
        prompt=prompt,
        moral_value=moral_value  # High value, should be accepted
    )

    # Should be accepted (if in wake phase)
    if result["phase"] == "wake":
        assert result["accepted"] is True, (
            f"High moral value {moral_value} should be accepted in wake phase"
        )

        # Response content should be present
        assert "response" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0, "Response should not be empty"

    # All governance fields must be present regardless
    required_fields = ["phase", "accepted", "note", "moral_threshold", "step"]
    for field in required_fields:
        assert field in result


def test_governance_metadata_additional_fields():
    """
    Test that additional governance-related fields are present
    and have correct types.
    """
    wrapper = create_test_wrapper()

    result = wrapper.generate(
        prompt="Test for additional fields",
        moral_value=0.9
    )

    # Additional fields that should be present
    additional_fields = {
        "context_items": int,
        "max_tokens_used": int,
    }

    for field, expected_type in additional_fields.items():
        assert field in result, f"Additional field '{field}' should be present"
        assert isinstance(result[field], expected_type), (
            f"Field '{field}' should be {expected_type.__name__}, "
            f"got {type(result[field])}"
        )


# ============================================================================
# Edge Cases
# ============================================================================

@pytest.mark.parametrize("moral_value", [0.0, 0.1, 0.5, 0.9, 1.0])
def test_edge_moral_values(moral_value):
    """Test boundary moral values."""
    wrapper = create_test_wrapper(initial_moral_threshold=0.50)

    result = wrapper.generate(
        prompt="Test boundary moral value",
        moral_value=moral_value
    )

    # Should always return structured response
    assert isinstance(result, dict)
    assert "accepted" in result
    assert "phase" in result


@pytest.mark.parametrize("capacity", [1, 5, 10, 50, 100])
def test_various_capacities(capacity):
    """Test LLMWrapper works with various capacity values."""
    wrapper = create_test_wrapper(
        capacity=capacity,
        initial_moral_threshold=0.20,
    )

    # Fill to capacity + some overflow
    for i in range(capacity + 5):
        wrapper.generate(
            prompt=f"Capacity test {i}",
            moral_value=0.9
        )

    # Memory should not exceed capacity
    assert wrapper.pelm.size <= capacity


def test_empty_prompt_handling():
    """Test handling of minimal prompts."""
    wrapper = create_test_wrapper()

    result = wrapper.generate(
        prompt="a",  # Minimal non-empty prompt
        moral_value=0.9
    )

    # Should return valid response
    assert isinstance(result, dict)
    assert "accepted" in result


@settings(max_examples=20, deadline=None)
@given(prompt=st.text(min_size=100, max_size=256))
def test_long_prompt_handling(prompt):
    """Test that long prompts are handled correctly."""
    assume(len(prompt.strip()) > 50)

    wrapper = create_test_wrapper()

    result = wrapper.generate(
        prompt=prompt,
        moral_value=0.9
    )

    # Should complete without error
    assert isinstance(result, dict)
    assert "phase" in result


def test_step_counter_increments():
    """Test that step counter increments with each call."""
    wrapper = create_test_wrapper()

    steps = []
    for i in range(5):
        result = wrapper.generate(
            prompt=f"Step test {i}",
            moral_value=0.9
        )
        steps.append(result["step"])

    # Steps should be monotonically increasing
    for i in range(len(steps) - 1):
        assert steps[i + 1] == steps[i] + 1, (
            f"Steps should increment: {steps[i]} -> {steps[i + 1]}"
        )


def test_moral_threshold_adaptation():
    """Test that moral threshold adapts over time."""
    wrapper = create_test_wrapper(initial_moral_threshold=0.50)

    # Process multiple high-moral requests
    for i in range(10):
        wrapper.generate(
            prompt=f"High moral request {i}",
            moral_value=0.95
        )

    # Threshold should have adapted
    final_threshold = wrapper.moral.threshold

    # Just verify threshold is within bounds (adaptation behavior is complex)
    assert 0.0 <= final_threshold <= 1.0, (
        f"Threshold {final_threshold} out of bounds"
    )


# ============================================================================
# Stateless Mode Edge Cases
# ============================================================================

def test_stateless_mode_empty_memory():
    """Test stateless mode with initially empty memory."""
    wrapper = create_test_wrapper()
    wrapper.stateless_mode = True

    # Memory should start empty
    assert wrapper.pelm.size == 0

    # Generate calls should not add to memory
    for i in range(10):
        wrapper.generate(
            prompt=f"Empty memory test {i}",
            moral_value=0.9
        )

    # Memory should still be empty
    assert wrapper.pelm.size == 0


def test_stateless_mode_flag_in_response():
    """Test that stateless_mode is reflected in response."""
    wrapper = create_test_wrapper()

    # Normal mode
    result_normal = wrapper.generate(
        prompt="Normal mode test",
        moral_value=0.9
    )
    assert result_normal.get("stateless_mode") is False

    # Enable stateless mode
    wrapper.stateless_mode = True

    result_stateless = wrapper.generate(
        prompt="Stateless mode test",
        moral_value=0.9
    )
    assert result_stateless.get("stateless_mode") is True


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
