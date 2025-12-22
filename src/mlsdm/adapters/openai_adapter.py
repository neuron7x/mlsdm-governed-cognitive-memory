"""
OpenAI LLM adapter for NeuroCognitiveEngine.

Provides integration with OpenAI's API for text generation.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from mlsdm.utils.env import get_env_float, get_env_int

if TYPE_CHECKING:
    from collections.abc import Callable


def build_openai_llm_adapter() -> Callable[[str, int], str]:
    """
    Build an LLM adapter that uses OpenAI API.

    Returns:
        A function (prompt: str, max_tokens: int) -> str that calls OpenAI API.

    Environment Variables:
        OPENAI_API_KEY: Required. Your OpenAI API key.
        OPENAI_MODEL: Optional. Model to use (default: "gpt-3.5-turbo").

    Raises:
        ValueError: If OPENAI_API_KEY is not set.
        ImportError: If openai package is not installed.

    Example:
        >>> os.environ["OPENAI_API_KEY"] = "sk-..."
        >>> llm_fn = build_openai_llm_adapter()
        >>> response = llm_fn("Hello, world!", max_tokens=100)
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI adapter")

    model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")

    import openai

    # Initialize OpenAI client
    timeout_seconds = get_env_float(
        "OPENAI_TIMEOUT_SECONDS",
        "LLM_REQUEST_TIMEOUT_SECONDS",
        "LLM_TIMEOUT_SECONDS",
    )
    max_retries = get_env_int("OPENAI_MAX_RETRIES", "LLM_MAX_RETRIES")

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if timeout_seconds is not None:
        client_kwargs["timeout"] = timeout_seconds
    if max_retries is not None:
        client_kwargs["max_retries"] = max_retries
    client = openai.OpenAI(**client_kwargs)

    def llm_generate_fn(prompt: str, max_tokens: int) -> str:
        """
        Generate text using OpenAI API.

        Args:
            prompt: The input prompt text.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            The generated text response.

        Raises:
            Exception: If the API call fails.
        """
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7,
            )

            # Extract the text from the response
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content or ""
            return ""

        except openai.APITimeoutError as e:
            raise TimeoutError(f"OpenAI API call timed out: {e}") from e
        except openai.APIConnectionError as e:
            raise ConnectionError(f"OpenAI API connection failed: {e}") from e
        except openai.RateLimitError as e:
            raise RuntimeError(f"OpenAI API rate limit exceeded: {e}") from e
        except openai.APIStatusError as e:
            if e.status_code >= 500:
                raise RuntimeError(f"OpenAI API error (status {e.status_code}): {e}") from e
            raise Exception(f"OpenAI API error (status {e.status_code}): {e}") from e
        except Exception as e:
            # Re-raise with more context
            raise Exception(f"OpenAI API call failed: {e}") from e

    return llm_generate_fn
