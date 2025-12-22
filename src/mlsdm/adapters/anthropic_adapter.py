"""
Anthropic LLM adapter for NeuroCognitiveEngine.

Provides integration with Anthropic's Claude API for text generation.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from mlsdm.utils.env import get_env_float, get_env_int

if TYPE_CHECKING:
    from collections.abc import Callable


def build_anthropic_llm_adapter() -> Callable[[str, int], str]:
    """
    Build an LLM adapter that uses Anthropic API.

    Returns:
        A function (prompt: str, max_tokens: int) -> str that calls Anthropic API.

    Environment Variables:
        ANTHROPIC_API_KEY: Required. Your Anthropic API key.
        ANTHROPIC_MODEL: Optional. Model to use (default: "claude-3-sonnet-20240229").

    Raises:
        ValueError: If ANTHROPIC_API_KEY is not set.
        ImportError: If anthropic package is not installed.

    Example:
        >>> os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
        >>> llm_fn = build_anthropic_llm_adapter()
        >>> response = llm_fn("Hello, world!", max_tokens=100)
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required for Anthropic adapter")

    model = os.environ.get("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")

    import anthropic

    # Initialize Anthropic client
    timeout_seconds = get_env_float(
        "ANTHROPIC_TIMEOUT_SECONDS",
        "LLM_REQUEST_TIMEOUT_SECONDS",
        "LLM_TIMEOUT_SECONDS",
    )
    max_retries = get_env_int("ANTHROPIC_MAX_RETRIES", "LLM_MAX_RETRIES")

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if timeout_seconds is not None:
        client_kwargs["timeout"] = timeout_seconds
    if max_retries is not None:
        client_kwargs["max_retries"] = max_retries
    client = anthropic.Anthropic(**client_kwargs)

    def llm_generate_fn(prompt: str, max_tokens: int) -> str:
        """
        Generate text using Anthropic API.

        Args:
            prompt: The input prompt text.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            The generated text response.

        Raises:
            Exception: If the API call fails.
        """
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

            # Extract the text from the response
            if response.content and len(response.content) > 0:
                content_block = response.content[0]
                # Anthropic returns ContentBlock objects with a text attribute
                return getattr(content_block, "text", "")
            return ""

        except anthropic.APITimeoutError as e:
            raise TimeoutError(f"Anthropic API call timed out: {e}") from e
        except anthropic.APIConnectionError as e:
            raise ConnectionError(f"Anthropic API connection failed: {e}") from e
        except anthropic.RateLimitError as e:
            raise RuntimeError(f"Anthropic API rate limit exceeded: {e}") from e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                raise RuntimeError(f"Anthropic API error (status {e.status_code}): {e}") from e
            raise Exception(f"Anthropic API error (status {e.status_code}): {e}") from e
        except Exception as e:
            # Re-raise with more context
            raise Exception(f"Anthropic API call failed: {e}") from e

    return llm_generate_fn
