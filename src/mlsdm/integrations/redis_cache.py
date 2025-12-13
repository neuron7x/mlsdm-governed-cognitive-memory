"""
Distributed Redis Cache Integration

Provides Redis-based distributed caching for multi-instance MLSDM deployments.
"""

import json
import logging
import pickle
from typing import Any, Optional

import numpy as np


class RedisCache:
    """
    Redis-based distributed cache for MLSDM.

    Supports caching embeddings, LLM responses, and other data structures
    with automatic serialization and TTL management.

    Example:
        >>> cache = RedisCache(
        ...     host="localhost",
        ...     port=6379,
        ...     password="secret",
        ...     db=0,
        ...     ttl=3600
        ... )
        >>> cache.set("key", "value")
        >>> value = cache.get("key")
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        db: int = 0,
        ttl: int = 3600,
        prefix: str = "mlsdm:",
    ) -> None:
        """
        Initialize Redis cache.

        Args:
            host: Redis host
            port: Redis port
            password: Redis password (optional)
            db: Redis database number
            ttl: Default TTL in seconds
            prefix: Key prefix for namespacing
        """
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.ttl = ttl
        self.prefix = prefix
        self.logger = logging.getLogger(__name__)

        self._client: Any = None
        self._connect()

    def _connect(self) -> None:
        """Establish Redis connection."""
        try:
            import redis  # type: ignore[import]

            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                decode_responses=False,
            )
            # Test connection
            self._client.ping()
            self.logger.info(f"Connected to Redis at {self.host}:{self.port}")

        except ImportError:
            self.logger.error("redis package not installed. Install with: pip install redis")
            raise
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if self._client is None:
            return None

        try:
            data = self._client.get(self._make_key(key))
            if data is None:
                return None
            return pickle.loads(data)
        except Exception as e:
            self.logger.error(f"Failed to get key {key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Optional TTL override

        Returns:
            True if successful
        """
        if self._client is None:
            return False

        try:
            data = pickle.dumps(value)
            self._client.setex(self._make_key(key), ttl or self.ttl, data)
            return True
        except Exception as e:
            self.logger.error(f"Failed to set key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if self._client is None:
            return False

        try:
            self._client.delete(self._make_key(key))
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete key {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if self._client is None:
            return False

        try:
            return bool(self._client.exists(self._make_key(key)))
        except Exception as e:
            self.logger.error(f"Failed to check key {key}: {e}")
            return False

    def get_embedding(self, text_hash: str) -> Optional[np.ndarray]:
        """
        Get cached embedding vector.

        Args:
            text_hash: Hash of the text

        Returns:
            Cached embedding or None
        """
        data = self.get(f"embedding:{text_hash}")
        if data is not None and isinstance(data, np.ndarray):
            return data
        return None

    def set_embedding(
        self, text_hash: str, embedding: np.ndarray, ttl: Optional[int] = None
    ) -> bool:
        """Cache embedding vector."""
        return self.set(f"embedding:{text_hash}", embedding, ttl)

    def clear_pattern(self, pattern: str) -> int:
        """
        Clear keys matching pattern.

        Args:
            pattern: Key pattern (e.g., "embedding:*")

        Returns:
            Number of keys deleted
        """
        if self._client is None:
            return 0

        try:
            keys = self._client.keys(self._make_key(pattern))
            if keys:
                return self._client.delete(*keys)
            return 0
        except Exception as e:
            self.logger.error(f"Failed to clear pattern {pattern}: {e}")
            return 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        if self._client is None:
            return {}

        try:
            info = self._client.info("stats")
            return {
                "total_connections": info.get("total_connections_received", 0),
                "total_commands": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": (
                    info.get("keyspace_hits", 0)
                    / max(
                        info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1
                    )
                ),
            }
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {}
