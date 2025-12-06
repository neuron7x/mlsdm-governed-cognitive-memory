#!/usr/bin/env python3
"""
Minimal Memory Demo for MLSDM.

This example demonstrates the core features of MLSDM:
1. Storing facts in memory
2. Retrieving facts from memory
3. Moral filtering (blocking toxic inputs)

This is the simplest way to see MLSDM in action.

Usage:
    python examples/minimal_memory_demo.py

Prerequisites:
    pip install -r requirements.txt
"""

from __future__ import annotations

from mlsdm import create_llm_wrapper


def main() -> None:
    """Run minimal memory demo showing storage, retrieval, and moral filtering."""
    print("=" * 70)
    print("MLSDM Minimal Memory Demo")
    print("=" * 70)
    print()

    # Create MLSDM wrapper with stub LLM (no API key needed)
    # Default configuration: 20k vectors × 384 dimensions = 29.37 MB fixed memory
    print("1. Creating MLSDM wrapper...")
    wrapper = create_llm_wrapper(
        wake_duration=8,
        sleep_duration=3,
        initial_moral_threshold=0.50,
    )
    # Note: These values come from LLMWrapper defaults (capacity=20_000, dim=384)
    print("   ✓ Wrapper created with 20,000 vector capacity (29.37 MB fixed)")
    print()

    # Store a fact in memory
    print("2. Storing a fact in memory...")
    result1 = wrapper.generate(
        prompt="The capital of France is Paris.",
        moral_value=0.8,  # High moral value = acceptable content
    )
    print("   Prompt: 'The capital of France is Paris.'")
    print(f"   Accepted: {result1['accepted']}")
    print(f"   Phase: {result1['phase']}")
    print(f"   Response: {result1['response'][:100]}...")
    print()

    # Retrieve from memory (context is stored)
    print("3. Retrieving fact from memory...")
    result2 = wrapper.generate(
        prompt="What is the capital of France?",
        moral_value=0.9,
    )
    print("   Prompt: 'What is the capital of France?'")
    print(f"   Accepted: {result2['accepted']}")
    print(f"   Response: {result2['response'][:100]}...")
    print()

    # Check system state
    print("4. Checking system state...")
    state = wrapper.get_state()
    print(f"   Step: {state['step']}")
    print(f"   Moral Threshold: {state['moral_threshold']:.2f}")
    print(f"   Memory Used: {state['qilm_stats']['used']}/{state['qilm_stats']['capacity']}")
    print(f"   Memory Size: {state['qilm_stats']['size_bytes'] / (1024 * 1024):.2f} MB")
    print()

    # Test moral filtering with low moral value (simulating toxic input)
    print("5. Testing moral filter (blocking toxic input)...")
    toxic_result = wrapper.generate(
        prompt="This is a toxic prompt that should be blocked.",
        moral_value=0.2,  # Low moral value = likely to be rejected
    )
    print("   Prompt: 'This is a toxic prompt that should be blocked.'")
    print("   Moral Value: 0.2 (low)")
    print(f"   Accepted: {toxic_result['accepted']}")
    if not toxic_result["accepted"]:
        print("   ✓ Toxic input was BLOCKED by moral filter")
    else:
        print("   ⚠ Toxic input was ACCEPTED (threshold may be low)")
    print()

    # Test moral filtering with high moral value
    print("6. Testing moral filter (accepting good input)...")
    good_result = wrapper.generate(
        prompt="Tell me about the benefits of renewable energy.",
        moral_value=0.9,  # High moral value = acceptable content
    )
    print("   Prompt: 'Tell me about the benefits of renewable energy.'")
    print("   Moral Value: 0.9 (high)")
    print(f"   Accepted: {good_result['accepted']}")
    if good_result["accepted"]:
        print("   ✓ Good input was ACCEPTED by moral filter")
        print(f"   Response: {good_result['response'][:100]}...")
    print()

    # Show final state
    print("7. Final system state...")
    final_state = wrapper.get_state()
    print(f"   Total steps: {final_state['step']}")
    print(f"   Final moral threshold: {final_state['moral_threshold']:.2f}")
    print(f"   Memory vectors stored: {final_state['qilm_stats']['used']}")
    print()

    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print()
    print("Key Takeaways:")
    print("  • MLSDM stores facts in bounded memory (29.37 MB fixed)")
    print("  • Moral filter blocks low-value inputs (adaptive threshold)")
    print("  • Memory retrieval provides context for subsequent prompts")
    print("  • System is fully observable (metrics, state, tracing)")
    print()
    print("Next Steps:")
    print("  • See README.md for full documentation")
    print("  • Try examples/example_basic_sdk.py for SDK usage")
    print("  • See docs/MLSDM_VALIDATION_REPORT.md for test results")
    print("  • See docs/MLSDM_POSITIONING.md for use cases")
    print()


if __name__ == "__main__":
    main()
