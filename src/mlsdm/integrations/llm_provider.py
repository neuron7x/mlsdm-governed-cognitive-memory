"""
LLM Provider Abstraction Layer

Unified interface for multiple LLM providers including OpenAI, Anthropic,
Cohere, HuggingFace, and local models.
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

import requests


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class LLMProviderClient:
    """
    Universal LLM provider client.

    Provides unified interface for multiple LLM providers with
    automatic fallback and standardized request/response format.

    Example:
        >>> client = LLMProviderClient(
        ...     provider=LLMProvider.OPENAI,
        ...     api_key="sk-...",
        ...     model="gpt-4"
        ... )
        >>> response = client.generate("Hello, world!", max_tokens=100)
        >>> print(response)
    """

    def __init__(
        self,
        provider: LLMProvider = LLMProvider.LOCAL,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        timeout: int = 60,
    ) -> None:
        """
        Initialize LLM provider client.

        Args:
            provider: LLM provider to use
            api_key: API key for provider
            model: Model name or identifier
            temperature: Sampling temperature
            timeout: Request timeout in seconds
        """
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

        # Provider-specific endpoints
        self._endpoints = {
            LLMProvider.OPENAI: "https://api.openai.com/v1/chat/completions",
            LLMProvider.ANTHROPIC: "https://api.anthropic.com/v1/messages",
            LLMProvider.COHERE: "https://api.cohere.ai/v1/generate",
            LLMProvider.HUGGINGFACE: "https://api-inference.huggingface.co/models",
        }

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text completion.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Optional temperature override
            **kwargs: Provider-specific parameters

        Returns:
            Generated text

        Raises:
            ValueError: If provider is not supported or API call fails
        """
        temp = temperature if temperature is not None else self.temperature

        if self.provider == LLMProvider.LOCAL:
            return self._generate_local(prompt, max_tokens)
        elif self.provider == LLMProvider.OPENAI:
            return self._generate_openai(prompt, max_tokens, temp, **kwargs)
        elif self.provider == LLMProvider.ANTHROPIC:
            return self._generate_anthropic(prompt, max_tokens, temp, **kwargs)
        elif self.provider == LLMProvider.COHERE:
            return self._generate_cohere(prompt, max_tokens, temp, **kwargs)
        elif self.provider == LLMProvider.HUGGINGFACE:
            return self._generate_huggingface(prompt, max_tokens, temp, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _generate_local(self, prompt: str, max_tokens: int) -> str:
        """Generate with local stub (placeholder)."""
        self.logger.debug(f"Generating local response for prompt of length {len(prompt)}")
        return f"[Local stub response to: {prompt[:50]}...]"

    def _generate_openai(
        self, prompt: str, max_tokens: int, temperature: float, **kwargs: Any
    ) -> str:
        """Generate using OpenAI API."""
        if not self.api_key:
            raise ValueError("OpenAI API key required")

        url = self._endpoints[LLMProvider.OPENAI]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            self.logger.error(f"OpenAI request failed: {e}")
            raise

    def _generate_anthropic(
        self, prompt: str, max_tokens: int, temperature: float, **kwargs: Any
    ) -> str:
        """Generate using Anthropic API."""
        if not self.api_key:
            raise ValueError("Anthropic API key required")

        url = self._endpoints[LLMProvider.ANTHROPIC]
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data["content"][0]["text"]
        except requests.RequestException as e:
            self.logger.error(f"Anthropic request failed: {e}")
            raise

    def _generate_cohere(
        self, prompt: str, max_tokens: int, temperature: float, **kwargs: Any
    ) -> str:
        """Generate using Cohere API."""
        if not self.api_key:
            raise ValueError("Cohere API key required")

        url = self._endpoints[LLMProvider.COHERE]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data["generations"][0]["text"]
        except requests.RequestException as e:
            self.logger.error(f"Cohere request failed: {e}")
            raise

    def _generate_huggingface(
        self, prompt: str, max_tokens: int, temperature: float, **kwargs: Any
    ) -> str:
        """Generate using HuggingFace Inference API."""
        if not self.api_key:
            raise ValueError("HuggingFace API key required")

        url = f"{self._endpoints[LLMProvider.HUGGINGFACE]}/{self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                **kwargs,
            },
        }

        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            # HuggingFace response format varies
            if isinstance(data, list) and len(data) > 0:
                return data[0].get("generated_text", "")
            elif isinstance(data, dict):
                return data.get("generated_text", "")
            return str(data)

        except requests.RequestException as e:
            self.logger.error(f"HuggingFace request failed: {e}")
            raise

    def get_provider_info(self) -> Dict[str, Any]:
        """Get provider information."""
        return {
            "provider": self.provider.value,
            "model": self.model,
            "temperature": self.temperature,
            "timeout": self.timeout,
        }
