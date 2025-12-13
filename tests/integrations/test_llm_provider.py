"""
Tests for LLM provider integration.
"""

from unittest.mock import MagicMock, patch

import pytest

from mlsdm.integrations import LLMProvider, LLMProviderClient


class TestLLMProviderClient:
    """Test LLM provider client integration."""

    def test_initialization(self) -> None:
        """Test client initialization."""
        client = LLMProviderClient(
            provider=LLMProvider.OPENAI,
            api_key="sk-test",
            model="gpt-4",
            temperature=0.7,
        )

        assert client.provider == LLMProvider.OPENAI
        assert client.api_key == "sk-test"
        assert client.model == "gpt-4"
        assert client.temperature == 0.7

    def test_generate_local(self) -> None:
        """Test local generation (stub)."""
        client = LLMProviderClient(provider=LLMProvider.LOCAL)

        response = client.generate("Hello, world!", max_tokens=100)

        assert isinstance(response, str)
        assert len(response) > 0

    def test_generate_openai_success(self) -> None:
        """Test OpenAI generation with mocked API."""
        client = LLMProviderClient(
            provider=LLMProvider.OPENAI, api_key="sk-test", model="gpt-4"
        )

        mock_response = {
            "choices": [{"message": {"content": "Generated response from OpenAI"}}]
        }

        with patch("requests.post") as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status = MagicMock()

            response = client.generate("Test prompt", max_tokens=100)

            assert response == "Generated response from OpenAI"

    def test_generate_openai_missing_key(self) -> None:
        """Test OpenAI generation without API key."""
        client = LLMProviderClient(provider=LLMProvider.OPENAI)

        with pytest.raises(ValueError, match="OpenAI API key required"):
            client.generate("Test", max_tokens=100)

    def test_temperature_override(self) -> None:
        """Test temperature override in generate call."""
        client = LLMProviderClient(
            provider=LLMProvider.LOCAL, temperature=0.7
        )

        # Should use overridden temperature
        response = client.generate("Test", max_tokens=100, temperature=0.9)
        assert isinstance(response, str)

    def test_get_provider_info(self) -> None:
        """Test provider info retrieval."""
        client = LLMProviderClient(
            provider=LLMProvider.ANTHROPIC,
            model="claude-2",
            temperature=0.8,
            timeout=30,
        )

        info = client.get_provider_info()

        assert info["provider"] == "anthropic"
        assert info["model"] == "claude-2"
        assert info["temperature"] == 0.8
        assert info["timeout"] == 30
