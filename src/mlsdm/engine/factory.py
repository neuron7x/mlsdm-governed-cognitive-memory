"""
Factory for building NeuroCognitiveEngine instances from environment configuration.

This module provides a convenient way to instantiate NeuroCognitiveEngine
with different LLM backends based on environment variables.
"""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING

import numpy as np

from mlsdm.adapters import build_local_stub_llm_adapter, build_openai_llm_adapter
from mlsdm.engine.neuro_cognitive_engine import (
    NeuroCognitiveEngine,
    NeuroEngineConfig,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def build_stub_embedding_fn(dim: int = 384) -> Callable[[str], np.ndarray]:
    """
    Build a deterministic stub embedding function.

    This function creates embeddings based on a hash of the text,
    ensuring deterministic results for testing.

    Args:
        dim: Dimensionality of the embedding vector (default: 384).

    Returns:
        A function (text: str) -> np.ndarray that returns deterministic embeddings.

    Example:
        >>> embed_fn = build_stub_embedding_fn(384)
        >>> vec = embed_fn("test text")
        >>> assert vec.shape == (384,)
    """

    def embedding_fn(text: str) -> np.ndarray:
        """
        Generate a deterministic embedding for the given text.

        Args:
            text: Input text to embed.

        Returns:
            A deterministic embedding vector of shape (dim,).
        """
        # Create a deterministic hash-based seed
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        seed = int(text_hash[:8], 16) % (2**31)

        # Generate deterministic random vector
        rng = np.random.RandomState(seed)
        vector = rng.randn(dim)

        # Normalize to unit length
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        return vector.astype(np.float32)

    return embedding_fn


def build_neuro_engine_from_env(
    config: NeuroEngineConfig | None = None,
) -> NeuroCognitiveEngine:
    """
    Build a NeuroCognitiveEngine instance from environment variables.

    Environment Variables:
        LLM_BACKEND: Backend to use ("openai" or "local_stub", default: "local_stub").
        OPENAI_API_KEY: Required when LLM_BACKEND="openai".
        OPENAI_MODEL: Optional OpenAI model name (default: "gpt-3.5-turbo").
        EMBEDDING_DIM: Embedding dimensionality (default: 384).

    Args:
        config: Optional NeuroEngineConfig to use. If None, uses default config.

    Returns:
        A configured NeuroCognitiveEngine instance.

    Raises:
        ValueError: If LLM_BACKEND is invalid or required environment variables are missing.

    Example:
        >>> os.environ["LLM_BACKEND"] = "local_stub"
        >>> engine = build_neuro_engine_from_env()
        >>> result = engine.generate("Hello, world!")
    """
    # Get backend choice from environment
    backend = os.environ.get("LLM_BACKEND", "local_stub").lower()

    # Get embedding dimension
    dim = int(os.environ.get("EMBEDDING_DIM", "384"))

    # Build LLM adapter based on backend
    if backend == "openai":
        llm_generate_fn = build_openai_llm_adapter()
    elif backend == "local_stub":
        llm_generate_fn = build_local_stub_llm_adapter()
    else:
        raise ValueError(
            f"Invalid LLM_BACKEND: {backend}. Valid options are: 'openai', 'local_stub'"
        )

    # Build embedding function (using stub for now)
    # In production, this could be replaced with real embeddings
    # (sentence-transformers, OpenAI embeddings, etc.)
    embedding_fn = build_stub_embedding_fn(dim=dim)

    # Use provided config or create default with specified dim
    if config is None:
        config = NeuroEngineConfig(dim=dim)
    # Note: We don't override config.dim if provided, to respect user's choice

    # Build and return engine
    return NeuroCognitiveEngine(
        llm_generate_fn=llm_generate_fn,
        embedding_fn=embedding_fn,
        config=config,
    )
