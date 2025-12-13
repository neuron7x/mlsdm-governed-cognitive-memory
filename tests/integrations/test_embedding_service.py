"""
Tests for embedding service integration.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from mlsdm.integrations import EmbeddingServiceClient, EmbeddingProvider


class TestEmbeddingServiceClient:
    """Test embedding service client integration."""

    def test_initialization(self) -> None:
        """Test client initialization."""
        client = EmbeddingServiceClient(
            provider=EmbeddingProvider.OPENAI,
            api_key="sk-test",
            model="text-embedding-ada-002",
            dimension=1536,
        )

        assert client.provider == EmbeddingProvider.OPENAI
        assert client.api_key == "sk-test"
        assert client.model == "text-embedding-ada-002"
        assert client.dimension == 1536

    def test_embed_local(self) -> None:
        """Test local embedding generation."""
        client = EmbeddingServiceClient(provider=EmbeddingProvider.LOCAL, dimension=384)

        embedding = client.embed("Hello, world!")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

    def test_embed_batch(self) -> None:
        """Test batch embedding generation."""
        client = EmbeddingServiceClient(provider=EmbeddingProvider.LOCAL, dimension=384)

        texts = ["Hello", "World", "Test"]
        embeddings = client.embed_batch(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)

    def test_embed_openai_success(self) -> None:
        """Test OpenAI embedding with mocked API."""
        client = EmbeddingServiceClient(
            provider=EmbeddingProvider.OPENAI, api_key="sk-test", dimension=1536
        )

        mock_response = {
            "data": [{"embedding": list(np.random.randn(1536))}]
        }

        with patch("requests.post") as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status = MagicMock()

            embedding = client.embed("Test text")

            assert isinstance(embedding, np.ndarray)
            assert len(embedding) == 1536

    def test_embed_openai_missing_key(self) -> None:
        """Test OpenAI embedding without API key."""
        client = EmbeddingServiceClient(provider=EmbeddingProvider.OPENAI)

        with pytest.raises(ValueError, match="OpenAI API key required"):
            client.embed("Test")

    def test_unsupported_provider(self) -> None:
        """Test error handling for unsupported provider."""
        # Can't directly test since Enum prevents invalid values
        # but test that the provider types are properly defined
        assert EmbeddingProvider.OPENAI.value == "openai"
        assert EmbeddingProvider.COHERE.value == "cohere"
        assert EmbeddingProvider.HUGGINGFACE.value == "huggingface"
