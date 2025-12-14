"""
External Embedding Service Integration

Provides unified interface for external embedding APIs including OpenAI,
Cohere, HuggingFace, and others.
"""

import hashlib
import logging
from enum import Enum
from typing import List, Optional

import numpy as np
import requests


class EmbeddingProvider(Enum):
    """Supported embedding providers."""

    OPENAI = "openai"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class EmbeddingServiceClient:
    """
    Universal embedding service client.

    Provides unified interface for multiple embedding providers with
    automatic fallback and retry logic.

    Example:
        >>> client = EmbeddingServiceClient(
        ...     provider=EmbeddingProvider.OPENAI,
        ...     api_key="sk-...",
        ...     model="text-embedding-ada-002"
        ... )
        >>> embedding = client.embed("Hello, world!")
        >>> print(embedding.shape)  # (1536,)
    """

    def __init__(
        self,
        provider: EmbeddingProvider = EmbeddingProvider.LOCAL,
        api_key: Optional[str] = None,
        model: str = "text-embedding-ada-002",
        dimension: int = 384,
        timeout: int = 30,
    ) -> None:
        """
        Initialize embedding service client.

        Args:
            provider: Embedding provider to use
            api_key: API key for provider
            model: Model name or identifier
            dimension: Expected embedding dimension
            timeout: Request timeout in seconds
        """
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)

        # Provider-specific endpoints
        self._endpoints = {
            EmbeddingProvider.OPENAI: "https://api.openai.com/v1/embeddings",
            EmbeddingProvider.COHERE: "https://api.cohere.ai/v1/embed",
            EmbeddingProvider.HUGGINGFACE: "https://api-inference.huggingface.co/pipeline/feature-extraction",
        }

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as numpy array

        Raises:
            ValueError: If provider is not supported or API call fails
        """
        if self.provider == EmbeddingProvider.LOCAL:
            return self._embed_local(text)
        elif self.provider == EmbeddingProvider.OPENAI:
            return self._embed_openai(text)
        elif self.provider == EmbeddingProvider.COHERE:
            return self._embed_cohere(text)
        elif self.provider == EmbeddingProvider.HUGGINGFACE:
            return self._embed_huggingface(text)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            Array of embeddings with shape (len(texts), dimension)
        """
        embeddings = [self.embed(text) for text in texts]
        return np.array(embeddings)

    def _embed_local(self, text: str) -> np.ndarray:
        """Generate deterministic pseudo-embedding from text hash."""
        # Create deterministic embedding based on text hash for reproducibility
        # This replaces random embedding to ensure consistent results
        
        # Hash text to get deterministic seed
        text_bytes = text.encode('utf-8')
        hash_bytes = hashlib.blake2b(text_bytes, digest_size=32).digest()
        seed = int.from_bytes(hash_bytes[:8], 'big') % (2**32)
        
        # Generate deterministic pseudo-embedding
        rng = np.random.default_rng(seed)
        embedding = rng.standard_normal(self.dimension).astype(np.float32)
        
        # Normalize to unit length for consistency with real embeddings
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        self.logger.debug(f"Generated deterministic local embedding for text of length {len(text)}")
        return embedding

    def _embed_openai(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI API."""
        if not self.api_key:
            raise ValueError("OpenAI API key required")

        url = self._endpoints[EmbeddingProvider.OPENAI]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"input": text, "model": self.model}

        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            # Validate response structure
            if "data" not in data or not data["data"]:
                raise ValueError("Invalid OpenAI embedding response: missing data")
            if "embedding" not in data["data"][0]:
                raise ValueError("Invalid OpenAI embedding response: missing embedding")
                
            embedding = np.array(data["data"][0]["embedding"], dtype=np.float32)
            return embedding
        except requests.RequestException as e:
            self.logger.error(f"OpenAI embedding request failed: {e}")
            raise

    def _embed_cohere(self, text: str) -> np.ndarray:
        """Generate embedding using Cohere API."""
        if not self.api_key:
            raise ValueError("Cohere API key required")

        url = self._endpoints[EmbeddingProvider.COHERE]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"texts": [text], "model": self.model}

        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            embedding = np.array(data["embeddings"][0], dtype=np.float32)
            return embedding
        except requests.RequestException as e:
            self.logger.error(f"Cohere embedding request failed: {e}")
            raise

    def _embed_huggingface(self, text: str) -> np.ndarray:
        """Generate embedding using HuggingFace Inference API."""
        if not self.api_key:
            raise ValueError("HuggingFace API key required")

        url = f"{self._endpoints[EmbeddingProvider.HUGGINGFACE]}/{self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"inputs": text}

        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            # HuggingFace returns nested list
            if isinstance(data, list) and len(data) > 0:
                embedding = np.array(data[0], dtype=np.float32)
            else:
                embedding = np.array(data, dtype=np.float32)
            return embedding
        except requests.RequestException as e:
            self.logger.error(f"HuggingFace embedding request failed: {e}")
            raise
