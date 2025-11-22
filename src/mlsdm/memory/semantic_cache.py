"""
Semantic response cache for NeuroCognitiveEngine.

This module provides a semantic caching layer that stores and retrieves responses
based on embedding similarity, moral values, and user intent. This reduces
unnecessary LLM calls for semantically similar queries.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CacheEntry:
    """A single cache entry with query context and response.

    Attributes:
        query_embedding: Embedding vector for the query
        moral_value: Moral threshold used for this query
        user_intent: User intent category
        response: Cached response text
        hit_count: Number of times this entry was retrieved
    """
    query_embedding: np.ndarray
    moral_value: float
    user_intent: str
    response: str
    hit_count: int = 0


class SemanticResponseCache:
    """Semantic cache for LLM responses.

    Stores responses indexed by (query_embedding, moral_value, user_intent) tuples
    and retrieves them based on similarity matching. Uses LRU eviction when full.

    Args:
        max_entries: Maximum number of cache entries
        similarity_threshold: Minimum cosine similarity for cache hit (0.0-1.0)
        moral_tolerance: Maximum difference in moral_value for cache hit

    Example:
        >>> cache = SemanticResponseCache(max_entries=100, similarity_threshold=0.9)
        >>> embedding = np.random.randn(384)
        >>> cache.store(embedding, 0.5, "conversational", "Hello!")
        >>> result = cache.lookup(embedding, 0.5, "conversational")
        >>> result is not None
        True
    """

    def __init__(
        self,
        max_entries: int = 1000,
        similarity_threshold: float = 0.85,
        moral_tolerance: float = 0.1,
    ) -> None:
        """Initialize semantic cache.

        Args:
            max_entries: Maximum number of entries (LRU eviction when exceeded)
            similarity_threshold: Cosine similarity threshold for matches (0.0-1.0)
            moral_tolerance: Maximum absolute difference in moral_value for matches
        """
        if max_entries <= 0:
            raise ValueError(f"max_entries must be positive, got {max_entries}")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be in [0, 1], got {similarity_threshold}"
            )
        if moral_tolerance < 0.0:
            raise ValueError(f"moral_tolerance must be non-negative, got {moral_tolerance}")

        self.max_entries = max_entries
        self.similarity_threshold = similarity_threshold
        self.moral_tolerance = moral_tolerance

        # Use OrderedDict for LRU behavior
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Statistics
        self._hits = 0
        self._misses = 0

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            emb1: First embedding vector
            emb2: Second embedding vector

        Returns:
            Cosine similarity in range [-1, 1]
        """
        # Normalize vectors
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Compute cosine similarity
        similarity = np.dot(emb1, emb2) / (norm1 * norm2)

        return float(similarity)

    def _make_key(self, query_embedding: np.ndarray, moral_value: float, user_intent: str) -> str:
        """Create a unique key for cache storage.

        Note: This is used for exact matching in the OrderedDict.
        Semantic matching is done via similarity computation.
        """
        # Use hash of embedding + moral + intent as key
        emb_hash = hash(query_embedding.tobytes())
        return f"{emb_hash}_{moral_value:.3f}_{user_intent}"

    def lookup(
        self,
        query_embedding: np.ndarray,
        moral_value: float,
        user_intent: str
    ) -> str | None:
        """Look up a cached response for semantically similar query.

        Args:
            query_embedding: Query embedding vector
            moral_value: Moral threshold for query
            user_intent: User intent category

        Returns:
            Cached response if found, None otherwise
        """
        if not isinstance(query_embedding, np.ndarray):
            return None

        # Search for best matching entry
        best_match: CacheEntry | None = None
        best_similarity = -1.0
        best_key: str | None = None

        for key, entry in self._cache.items():
            # Check user intent match (must be exact)
            if entry.user_intent != user_intent:
                continue

            # Check moral value tolerance
            if abs(entry.moral_value - moral_value) > self.moral_tolerance:
                continue

            # Compute semantic similarity
            similarity = self._compute_similarity(query_embedding, entry.query_embedding)

            if similarity >= self.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = entry
                best_key = key

        if best_match is not None and best_key is not None:
            # Cache hit - update statistics and move to end (LRU)
            self._hits += 1
            best_match.hit_count += 1
            self._cache.move_to_end(best_key)
            return best_match.response

        # Cache miss
        self._misses += 1
        return None

    def store(
        self,
        query_embedding: np.ndarray,
        moral_value: float,
        user_intent: str,
        response: str
    ) -> None:
        """Store a response in the cache.

        Args:
            query_embedding: Query embedding vector
            moral_value: Moral threshold used
            user_intent: User intent category
            response: Response text to cache
        """
        if not isinstance(query_embedding, np.ndarray):
            raise TypeError("query_embedding must be a numpy array")
        if not isinstance(response, str) or not response.strip():
            # Don't cache empty responses
            return

        # Create cache key
        key = self._make_key(query_embedding, moral_value, user_intent)

        # Check if entry already exists
        if key in self._cache:
            # Update existing entry and move to end
            self._cache[key].response = response
            self._cache.move_to_end(key)
            return

        # Evict oldest entry if at capacity
        if len(self._cache) >= self.max_entries:
            self._cache.popitem(last=False)  # Remove oldest (FIFO/LRU)

        # Add new entry
        entry = CacheEntry(
            query_embedding=query_embedding.copy(),
            moral_value=moral_value,
            user_intent=user_intent,
            response=response,
            hit_count=0
        )
        self._cache[key] = entry

    def clear(self) -> None:
        """Clear all cache entries and reset statistics."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache metrics
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            'size': len(self._cache),
            'max_entries': self.max_entries,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests,
        }

    def __len__(self) -> int:
        """Return number of entries in cache."""
        return len(self._cache)
