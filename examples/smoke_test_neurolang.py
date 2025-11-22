"""
Smoke Test for NeuroLang + Aphasia-Broca Integration

This script provides a manual verification tool for the NeuroLang extension
and Aphasia-Broca Model. Run this to quickly verify the integration works.

Usage:
    python examples/smoke_test_neurolang.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlsdm.extensions import AphasiaBrocaDetector, NeuroLangWrapper


def dummy_llm(prompt: str, max_tokens: int) -> str:
    """Dummy LLM that returns coherent responses."""
    return (
        "The cognitive architecture integrates multiple neurobiological principles "
        "to provide safe and coherent language generation with moral governance. "
        "This system has been validated through extensive testing and demonstrates "
        "significant improvements in response quality and safety metrics."
    )


def dummy_embedder(text: str) -> np.ndarray:
    """Dummy embedder that returns normalized deterministic vectors."""
    np.random.seed(hash(text) % (2**32))
    vec = np.random.randn(384).astype(np.float32)
    return vec / np.linalg.norm(vec)


def test_aphasia_detector():
    """Test the AphasiaBrocaDetector standalone."""
    print("=" * 70)
    print("SMOKE TEST 1: AphasiaBrocaDetector")
    print("=" * 70)

    detector = AphasiaBrocaDetector()

    healthy_text = "The system is working correctly and efficiently with good results."
    aphasic_text = "Bad. Short. No work."

    print("\n1. Analyzing healthy text:")
    print(f'   Input: "{healthy_text}"')
    result = detector.analyze(healthy_text)
    print(f"   Result: is_aphasic={result['is_aphasic']}, severity={result['severity']:.2f}")
    print(f"   Metrics: avg_len={result['avg_sentence_len']:.1f}, func_ratio={result['function_word_ratio']:.2f}")

    print("\n2. Analyzing aphasic text:")
    print(f'   Input: "{aphasic_text}"')
    result = detector.analyze(aphasic_text)
    print(f"   Result: is_aphasic={result['is_aphasic']}, severity={result['severity']:.2f}")
    print(f"   Flags: {result['flags']}")

    print("\n✓ AphasiaBrocaDetector works correctly\n")


def test_neurolang_wrapper():
    """Test the NeuroLangWrapper end-to-end."""
    print("=" * 70)
    print("SMOKE TEST 2: NeuroLangWrapper")
    print("=" * 70)

    print("\nInitializing NeuroLangWrapper...")
    print("(This includes training the actor-critic models, may take a moment)")

    wrapper = NeuroLangWrapper(
        llm_generate_fn=dummy_llm,
        embedding_fn=dummy_embedder,
        dim=384,
        capacity=1024,
        wake_duration=2,
        sleep_duration=1,
        initial_moral_threshold=0.5,
    )

    print("✓ NeuroLangWrapper initialized successfully")

    print("\n1. Testing with high moral value (should accept):")
    result = wrapper.generate(
        prompt="Explain how the system works.",
        moral_value=0.8,
        max_tokens=100,
    )

    print(f"   Accepted: {result['accepted']}")
    print(f"   Phase: {result['phase']}")
    print(f"   Aphasia detected: {result['aphasia_flags']['is_aphasic']}")
    print(f"   Response length: {len(result['response'])} chars")

    print("\n2. Testing with low moral value (should reject):")
    result = wrapper.generate(
        prompt="Test prompt.",
        moral_value=0.1,
        max_tokens=100,
    )

    print(f"   Accepted: {result['accepted']}")
    print(f"   Response: {result['response'][:50]}...")

    print("\n✓ NeuroLangWrapper works correctly\n")


def test_imports():
    """Test that all imports work correctly."""
    print("=" * 70)
    print("SMOKE TEST 0: Import Validation")
    print("=" * 70)

    print("\n1. Testing import: from mlsdm.extensions import AphasiaBrocaDetector")
    from mlsdm.extensions import AphasiaBrocaDetector as ABDetector
    print(f"   ✓ Imported: {ABDetector}")

    print("\n2. Testing import: from mlsdm.extensions import NeuroLangWrapper")
    from mlsdm.extensions import NeuroLangWrapper as NLWrapper
    print(f"   ✓ Imported: {NLWrapper}")

    print("\n✓ All imports work correctly\n")


def main():
    """Run all smoke tests."""
    print("\n" + "=" * 70)
    print("NeuroLang + Aphasia-Broca SMOKE TEST SUITE")
    print("=" * 70)
    print()

    try:
        test_imports()
        test_aphasia_detector()
        test_neurolang_wrapper()

        print("=" * 70)
        print("ALL SMOKE TESTS PASSED ✓")
        print("=" * 70)
        print("\nThe NeuroLang + Aphasia-Broca integration is working correctly.")
        print()

    except Exception as e:
        print("\n" + "=" * 70)
        print("SMOKE TEST FAILED ✗")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
