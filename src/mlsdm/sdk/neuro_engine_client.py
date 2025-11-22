"""
NeuroCognitiveClient: High-level Python SDK for NeuroCognitiveEngine.

This module provides a convenient client interface for generating responses
using the NeuroCognitiveEngine with configurable backends.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from mlsdm.engine import NeuroEngineConfig, build_neuro_engine_from_env


class NeuroCognitiveClient:
    """High-level client for interacting with NeuroCognitiveEngine.

    This client provides a simple interface for generating cognitive responses
    with support for multiple backends (local_stub, openai) and optional configuration.

    Args:
        backend: LLM backend to use ("local_stub" or "openai"). Defaults to "local_stub".
        config: Optional NeuroEngineConfig for customizing engine behavior.
        api_key: Optional API key for OpenAI backend. If not provided, will use OPENAI_API_KEY env var.
        model: Optional model name for OpenAI backend. Defaults to "gpt-3.5-turbo".

    Example:
        >>> # Using local stub backend (no API key required)
        >>> client = NeuroCognitiveClient(backend="local_stub")
        >>> result = client.generate("Hello, world!")
        >>> print(result["response"])

        >>> # Using OpenAI backend
        >>> client = NeuroCognitiveClient(
        ...     backend="openai",
        ...     api_key="sk-...",
        ...     model="gpt-4"
        ... )
        >>> result = client.generate("Explain quantum computing")
        >>> print(result["response"])

        >>> # With custom configuration
        >>> from mlsdm.engine import NeuroEngineConfig
        >>> config = NeuroEngineConfig(
        ...     dim=512,
        ...     enable_fslgs=False,
        ...     initial_moral_threshold=0.6
        ... )
        >>> client = NeuroCognitiveClient(backend="local_stub", config=config)
        >>> result = client.generate("Tell me a story")
    """

    def __init__(
        self,
        backend: Literal["openai", "local_stub"] = "local_stub",
        config: NeuroEngineConfig | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the NeuroCognitiveClient.

        Args:
            backend: LLM backend to use ("local_stub" or "openai").
            config: Optional NeuroEngineConfig for customizing engine behavior.
            api_key: Optional API key for OpenAI backend.
            model: Optional model name for OpenAI backend.

        Raises:
            ValueError: If backend is invalid or required credentials are missing.
        """
        # Validate backend
        if backend not in ["openai", "local_stub"]:
            raise ValueError(
                f"Invalid backend: {backend}. Valid options are: 'openai', 'local_stub'"
            )

        # Set environment variables for factory
        os.environ["LLM_BACKEND"] = backend

        # Handle OpenAI-specific configuration
        if backend == "openai":
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            elif "OPENAI_API_KEY" not in os.environ:
                raise ValueError(
                    "OpenAI backend requires api_key parameter or OPENAI_API_KEY environment variable"
                )

            if model:
                os.environ["OPENAI_MODEL"] = model

        # Store configuration
        self._backend = backend
        self._config = config

        # Build engine using factory
        self._engine = build_neuro_engine_from_env(config=config)

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        moral_value: float | None = None,
        user_intent: str | None = None,
        cognitive_load: float | None = None,
        context_top_k: int | None = None,
    ) -> dict[str, Any]:
        """Generate a response using the NeuroCognitiveEngine.

        This method processes the input prompt through the complete cognitive pipeline,
        including moral filtering, memory retrieval, rhythm management, and optional
        FSLGS governance.

        Args:
            prompt: Input text prompt to process.
            max_tokens: Maximum number of tokens to generate (default: 512).
            moral_value: Moral threshold value between 0.0 and 1.0 (default: 0.5).
            user_intent: User intent category (default: "conversational").
            cognitive_load: Cognitive load value between 0.0 and 1.0 (default: 0.5).
            context_top_k: Number of top context items to retrieve (default: 5).

        Returns:
            Dictionary containing:
                - response (str): Generated response text.
                - governance (dict): Governance state information.
                - mlsdm (dict): MLSDM internal state.
                - timing (dict): Performance timing metrics in milliseconds.
                - validation_steps (list): Validation steps executed during generation.
                - error (dict | None): Error information if generation failed.
                - rejected_at (str | None): Stage at which request was rejected, if any.

        Example:
            >>> client = NeuroCognitiveClient()
            >>> result = client.generate(
            ...     prompt="What is consciousness?",
            ...     max_tokens=256,
            ...     moral_value=0.7,
            ...     user_intent="philosophical"
            ... )
            >>> print(f"Response: {result['response']}")
            >>> print(f"Timing: {result['timing']}")
        """
        # Build kwargs for engine.generate()
        kwargs: dict[str, Any] = {"prompt": prompt}

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if moral_value is not None:
            kwargs["moral_value"] = moral_value
        if user_intent is not None:
            kwargs["user_intent"] = user_intent
        if cognitive_load is not None:
            kwargs["cognitive_load"] = cognitive_load
        if context_top_k is not None:
            kwargs["context_top_k"] = context_top_k

        # Call engine and return result
        return self._engine.generate(**kwargs)

    @property
    def backend(self) -> str:
        """Get the current backend name."""
        return self._backend

    @property
    def config(self) -> NeuroEngineConfig | None:
        """Get the engine configuration."""
        return self._config
