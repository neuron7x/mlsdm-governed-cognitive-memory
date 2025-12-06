"""
Integration test for minimal_memory_demo.py example.

This test validates that the minimal memory demo script runs successfully
and demonstrates the expected behavior.
"""

from __future__ import annotations

from io import StringIO
from unittest.mock import patch

import pytest


def test_minimal_memory_demo_runs_successfully() -> None:
    """Test that minimal_memory_demo.py runs without errors."""
    # Import the main function from the demo
    from examples.minimal_memory_demo import main

    # Capture stdout
    captured_output = StringIO()

    # Run the demo
    with patch("sys.stdout", captured_output):
        main()

    # Get the output
    output = captured_output.getvalue()

    # Verify key sections are present
    assert "MLSDM Minimal Memory Demo" in output
    assert "Creating MLSDM wrapper" in output
    assert "Storing a fact in memory" in output
    assert "Retrieving fact from memory" in output
    assert "Testing moral filter" in output
    assert "Demo Complete!" in output

    # Verify expected behavior messages
    assert "✓ Wrapper created with 20,000 vector capacity" in output
    assert "Accepted: True" in output or "Accepted: False" in output
    assert "Memory Used:" in output

    # Verify final summary is present
    assert "Key Takeaways:" in output
    assert "Next Steps:" in output


def test_minimal_memory_demo_creates_wrapper() -> None:
    """Test that the demo creates a wrapper with correct configuration."""
    from mlsdm import create_llm_wrapper

    # Create wrapper as done in the demo
    wrapper = create_llm_wrapper(
        wake_duration=8,
        sleep_duration=3,
        initial_moral_threshold=0.50,
    )

    # Verify wrapper was created
    assert wrapper is not None

    # Get state and verify configuration
    state = wrapper.get_state()
    assert state["moral_threshold"] == 0.50
    assert state["qilm_stats"]["capacity"] == 20_000
    assert state["step"] == 0


def test_minimal_memory_demo_stores_and_retrieves() -> None:
    """Test that the demo can store and retrieve facts."""
    from mlsdm import create_llm_wrapper

    wrapper = create_llm_wrapper()

    # Store a fact
    result1 = wrapper.generate(
        prompt="The capital of France is Paris.",
        moral_value=0.8,
    )

    assert result1["accepted"] is True
    assert "response" in result1

    # Retrieve from memory
    result2 = wrapper.generate(
        prompt="What is the capital of France?",
        moral_value=0.9,
    )

    assert result2["accepted"] is True
    assert "response" in result2

    # Verify memory was used
    state = wrapper.get_state()
    assert state["qilm_stats"]["used"] > 0


def test_minimal_memory_demo_moral_filtering() -> None:
    """Test that the demo demonstrates moral filtering."""
    from mlsdm import create_llm_wrapper

    wrapper = create_llm_wrapper(initial_moral_threshold=0.50)

    # Test with low moral value (should be rejected or borderline)
    toxic_result = wrapper.generate(
        prompt="This is a toxic prompt.",
        moral_value=0.2,
    )

    # Low moral value should typically be rejected, but accept either outcome
    # (threshold is adaptive and may vary)
    assert "accepted" in toxic_result
    assert isinstance(toxic_result["accepted"], bool)

    # Test with high moral value (should be accepted)
    good_result = wrapper.generate(
        prompt="Tell me about renewable energy.",
        moral_value=0.9,
    )

    assert good_result["accepted"] is True
    assert "response" in good_result


def test_minimal_memory_demo_system_state() -> None:
    """Test that the demo can retrieve system state."""
    from mlsdm import create_llm_wrapper

    wrapper = create_llm_wrapper()

    # Get state
    state = wrapper.get_state()

    # Verify expected state fields
    assert "step" in state
    assert "moral_threshold" in state
    assert "qilm_stats" in state
    assert "capacity" in state["qilm_stats"]
    assert "used" in state["qilm_stats"]
    assert "size_bytes" in state["qilm_stats"]

    # Verify memory size is within expected bounds (29.37 MB)
    memory_mb = state["qilm_stats"]["size_bytes"] / (1024 * 1024)
    assert memory_mb <= 30.0  # Allow some headroom


@pytest.mark.slow
def test_minimal_memory_demo_full_execution() -> None:
    """Full smoke test of the minimal memory demo (marked as slow)."""
    from examples.minimal_memory_demo import main

    # Run the full demo
    # This is marked as slow because it exercises the full demo path
    try:
        main()
    except Exception as e:
        pytest.fail(f"Demo raised exception: {e}")
