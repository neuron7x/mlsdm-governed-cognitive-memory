#!/usr/bin/env python3
"""
Smoke test script for NeuroLangWrapper.

This script provides a quick verification that NeuroLangWrapper
can be instantiated and used for generation.

Usage:
    python scripts/smoke_neurolang_wrapper.py --prompt "Your test prompt"
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlsdm.extensions import NeuroLangWrapper


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Smoke test for NeuroLangWrapper"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Quick smoke test for NeuroLangWrapper.",
        help="Prompt to test with (default: 'Quick smoke test for NeuroLangWrapper.')"
    )
    return parser.parse_args(argv)


def dummy_llm(prompt: str, max_tokens: int) -> str:
    """Dummy LLM that returns coherent responses."""
    return (
        "This is a coherent answer that demonstrates how the system can respond "
        "in full sentences with proper grammar and structure."
    )


def dummy_embedder(text: str):
    """Dummy embedder that returns normalized deterministic vectors."""
    vec = np.ones(384, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def main(argv=None):
    args = parse_args(argv)

    wrapper = NeuroLangWrapper(
        llm_generate_fn=dummy_llm,
        embedding_fn=dummy_embedder,
        dim=384,
        capacity=512,
        wake_duration=2,
        sleep_duration=1,
        initial_moral_threshold=0.5,
    )

    result = wrapper.generate(
        prompt=args.prompt,
        moral_value=0.8,
        max_tokens=64,
    )
    print("Response:", result["response"])
    print("Phase:", result["phase"])
    print("Accepted:", result["accepted"])
    print("Aphasia flags:", result["aphasia_flags"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
