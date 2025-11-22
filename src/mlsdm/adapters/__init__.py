"""
LLM adapters for NeuroCognitiveEngine.

This module provides adapters for different LLM backends:
- OpenAI (cloud-based)
- Local stub (deterministic mock for testing)
"""

from .local_stub_adapter import build_local_stub_llm_adapter
from .openai_adapter import build_openai_llm_adapter

__all__ = [
    "build_openai_llm_adapter",
    "build_local_stub_llm_adapter",
]
