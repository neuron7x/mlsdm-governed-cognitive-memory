"""
NeuroCognitiveClient: High-level Python SDK for NeuroCognitiveEngine.

This module provides a convenient client interface for generating responses
using the NeuroCognitiveEngine with configurable backends.

SDK Contract Stability:
----------------------
The following methods and their signatures are part of the stable SDK contract:

    - generate(prompt, ...) -> GenerateResponseDTO
    - backend (property)
    - config (property)

Breaking changes to these will require a major version bump.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal

from mlsdm.adapters import LLMProviderError, LLMTimeoutError
from mlsdm.engine import NeuroEngineConfig, build_neuro_engine_from_env
from mlsdm.sdk.dto import GenerateResponseDTO
from mlsdm.sdk.exceptions import (
    MLSDMConfigError,
    MLSDMServerError,
    MLSDMTimeoutError,
    MLSDMValidationError,
)

# SDK uses logging, not print statements
_logger = logging.getLogger(__name__)

# Default timeout for SDK operations (in seconds)
DEFAULT_TIMEOUT_SECONDS = 30.0


class NeuroCognitiveClient:
    """High-level client for interacting with NeuroCognitiveEngine.

    This client provides a simple interface for generating cognitive responses
    with support for multiple backends (local_stub, openai) and optional configuration.

    The client provides:
    - Typed response DTOs (GenerateResponseDTO)
    - SDK-specific exceptions (MLSDMClientError, MLSDMServerError, etc.)
    - Default timeout handling
    - Proper error handling for HTTP-like errors

    Args:
        backend: LLM backend to use ("local_stub" or "openai"). Defaults to "local_stub".
        config: Optional NeuroEngineConfig for customizing engine behavior.
        api_key: Optional API key for OpenAI backend. If not provided, will use OPENAI_API_KEY env var.
        model: Optional model name for OpenAI backend. Defaults to "gpt-3.5-turbo".
        timeout: Timeout for operations in seconds. Defaults to 30.0.

    Example:
        >>> # Using local stub backend (no API key required)
        >>> client = NeuroCognitiveClient(backend="local_stub")
        >>> result = client.generate("Hello, world!")
        >>> print(result.response)  # Typed DTO access

        >>> # Using OpenAI backend
        >>> client = NeuroCognitiveClient(
        ...     backend="openai",
        ...     api_key="sk-...",
        ...     model="gpt-4"
        ... )
        >>> result = client.generate("Explain quantum computing")
        >>> print(result.response)

        >>> # With custom configuration
        >>> from mlsdm.engine import NeuroEngineConfig
        >>> config = NeuroEngineConfig(
        ...     dim=512,
        ...     enable_fslgs=False,
        ...     initial_moral_threshold=0.6
        ... )
        >>> client = NeuroCognitiveClient(backend="local_stub", config=config)
        >>> result = client.generate("Tell me a story")

    Raises:
        MLSDMClientError: For client-side errors (invalid input, config errors)
        MLSDMServerError: For server-side errors (internal errors)
        MLSDMTimeoutError: For timeout errors
    """

    def __init__(
        self,
        backend: Literal["openai", "local_stub"] = "local_stub",
        config: NeuroEngineConfig | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        """Initialize the NeuroCognitiveClient.

        Args:
            backend: LLM backend to use ("local_stub" or "openai").
            config: Optional NeuroEngineConfig for customizing engine behavior.
            api_key: Optional API key for OpenAI backend.
            model: Optional model name for OpenAI backend.
            timeout: Timeout for operations in seconds. Defaults to 30.0.

        Raises:
            ValueError: If backend is invalid or required credentials are missing.
            MLSDMConfigError: If configuration is invalid.
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
        self._timeout = timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS

        # Build engine using factory
        try:
            self._engine = build_neuro_engine_from_env(config=config)
        except Exception as e:
            _logger.error("Failed to initialize engine: %s", e)
            raise MLSDMConfigError(
                f"Failed to initialize engine: {e}",
                details={"backend": backend, "error": str(e)},
            ) from e

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        moral_value: float | None = None,
        user_intent: str | None = None,
        cognitive_load: float | None = None,
        context_top_k: int | None = None,
    ) -> GenerateResponseDTO:
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
            GenerateResponseDTO containing:
                - response (str): Generated response text.
                - phase (str): Current cognitive phase.
                - accepted (bool): Whether the request was accepted.
                - metrics (dict | None): Performance timing metrics.
                - safety_flags (dict | None): Safety validation results.
                - memory_stats (dict | None): Memory state statistics.
                - latency_ms (float | None): Total latency in milliseconds.

        Raises:
            MLSDMValidationError: If input validation fails.
            MLSDMServerError: If an internal error occurs.
            MLSDMTimeoutError: If the request times out.

        Example:
            >>> client = NeuroCognitiveClient()
            >>> result = client.generate(
            ...     prompt="What is consciousness?",
            ...     max_tokens=256,
            ...     moral_value=0.7,
            ...     user_intent="philosophical"
            ... )
            >>> print(f"Response: {result.response}")
            >>> print(f"Phase: {result.phase}")
            >>> if result.latency_ms:
            ...     print(f"Latency: {result.latency_ms:.2f}ms")
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

        # Call engine with exception handling
        try:
            raw_result = self._engine.generate(**kwargs)
            return GenerateResponseDTO.from_dict(raw_result)
        except LLMTimeoutError as e:
            _logger.error("Request timed out: %s", e)
            raise MLSDMTimeoutError(
                f"Request timed out: {e}",
                timeout_seconds=getattr(e, "timeout_seconds", self._timeout),
                operation="generate",
                details={"prompt_length": len(prompt)},
            ) from e
        except LLMProviderError as e:
            _logger.error("Provider error: %s", e)
            raise MLSDMServerError(
                f"Provider error: {e}",
                error_type="provider_error",
                details={"provider_id": getattr(e, "provider_id", None)},
            ) from e
        except ValueError as e:
            _logger.error("Validation error: %s", e)
            raise MLSDMValidationError(
                f"Validation error: {e}",
                details={"prompt_length": len(prompt)},
            ) from e
        except Exception as e:
            # Re-raise unknown exceptions as-is for backward compatibility
            # This allows existing error handling to continue working
            _logger.error("Unexpected error: %s", e)
            raise

    def generate_raw(
        self,
        prompt: str,
        max_tokens: int | None = None,
        moral_value: float | None = None,
        user_intent: str | None = None,
        cognitive_load: float | None = None,
        context_top_k: int | None = None,
    ) -> dict[str, Any]:
        """Generate a response and return raw dictionary (backward compatibility).

        This method is provided for backward compatibility with code that expects
        the raw dictionary response format.

        Args:
            prompt: Input text prompt to process.
            max_tokens: Maximum number of tokens to generate (default: 512).
            moral_value: Moral threshold value between 0.0 and 1.0 (default: 0.5).
            user_intent: User intent category (default: "conversational").
            cognitive_load: Cognitive load value between 0.0 and 1.0 (default: 0.5).
            context_top_k: Number of top context items to retrieve (default: 5).

        Returns:
            Dictionary containing raw engine response.
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

        return self._engine.generate(**kwargs)

    @property
    def backend(self) -> str:
        """Get the current backend name."""
        return self._backend

    @property
    def config(self) -> NeuroEngineConfig | None:
        """Get the engine configuration."""
        return self._config

    @property
    def timeout(self) -> float:
        """Get the configured timeout in seconds."""
        return self._timeout
