"""
Distributed Redis Cache Integration

Provides Redis-based distributed caching for multi-instance MLSDM deployments.
"""

import base64
import json
import logging
import warnings
from typing import Any, Dict, Optional

import numpy as np


class RedisCache:
    """
    Redis-based distributed cache for MLSDM.

    Supports caching embeddings, LLM responses, and other data structures
    with safe serialization by default (no pickle).

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
        dangerously_allow_pickle: bool = False,
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
            dangerously_allow_pickle: UNSAFE - Allow pickle serialization.
                Only enable if you fully control the data source and understand
                the security risks of arbitrary code execution.
        """
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.ttl = ttl
        self.prefix = prefix
        self.dangerously_allow_pickle = dangerously_allow_pickle
        self.logger = logging.getLogger(__name__)

        if self.dangerously_allow_pickle:
            warnings.warn(
                "RedisCache: pickle serialization enabled. This is UNSAFE for untrusted data "
                "and can lead to arbitrary code execution. Use only if you fully control the data source.",
                SecurityWarning,
                stacklevel=2,
            )

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

    def _serialize_safe(self, value: Any) -> bytes:
        """
        Safely serialize value without pickle.

        Supports: None, bool, int, float, str, bytes, list, dict, numpy.ndarray

        Args:
            value: Value to serialize

        Returns:
            Serialized bytes

        Raises:
            TypeError: If value type is not supported
        """
        if value is None:
            return json.dumps({"_type": "none"}).encode("utf-8")
        elif isinstance(value, bool):
            return json.dumps({"_type": "bool", "value": value}).encode("utf-8")
        elif isinstance(value, int):
            return json.dumps({"_type": "int", "value": value}).encode("utf-8")
        elif isinstance(value, float):
            return json.dumps({"_type": "float", "value": value}).encode("utf-8")
        elif isinstance(value, str):
            return json.dumps({"_type": "str", "value": value}).encode("utf-8")
        elif isinstance(value, bytes):
            return json.dumps({"_type": "bytes", "value": base64.b64encode(value).decode("ascii")}).encode("utf-8")
        elif isinstance(value, (list, tuple)):
            # Recursively serialize list elements (must be primitive types)
            return json.dumps({"_type": "list", "value": value}).encode("utf-8")
        elif isinstance(value, dict):
            # Dict with string keys and primitive values
            return json.dumps({"_type": "dict", "value": value}).encode("utf-8")
        elif isinstance(value, np.ndarray):
            # Serialize numpy array without pickle
            if not np.issubdtype(value.dtype, np.number):
                raise TypeError(f"Only numeric numpy arrays are supported, got dtype={value.dtype}")
            return json.dumps({
                "_type": "ndarray",
                "dtype": str(value.dtype),
                "shape": value.shape,
                "data": base64.b64encode(value.tobytes()).decode("ascii")
            }).encode("utf-8")
        else:
            raise TypeError(f"Unsupported type for safe serialization: {type(value).__name__}")

    def _deserialize_safe(self, data: bytes) -> Any:
        """
        Safely deserialize value without pickle.

        Args:
            data: Serialized bytes

        Returns:
            Deserialized value

        Raises:
            ValueError: If data is invalid or type is unknown
        """
        try:
            obj = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise ValueError(f"Failed to deserialize data: {e}")

        if not isinstance(obj, dict) or "_type" not in obj:
            raise ValueError("Invalid serialized data format")

        obj_type = obj["_type"]

        if obj_type == "none":
            return None
        elif obj_type == "bool":
            return obj["value"]
        elif obj_type == "int":
            return obj["value"]
        elif obj_type == "float":
            return obj["value"]
        elif obj_type == "str":
            return obj["value"]
        elif obj_type == "bytes":
            return base64.b64decode(obj["value"])
        elif obj_type == "list":
            return obj["value"]
        elif obj_type == "dict":
            return obj["value"]
        elif obj_type == "ndarray":
            dtype = np.dtype(obj["dtype"])
            shape = tuple(obj["shape"])
            data_bytes = base64.b64decode(obj["data"])
            arr = np.frombuffer(data_bytes, dtype=dtype)
            return arr.reshape(shape)
        else:
            raise ValueError(f"Unknown serialized type: {obj_type}")

    def _serialize(self, value: Any) -> bytes:
        """Serialize value using configured method."""
        if self.dangerously_allow_pickle:
            import pickle
            self.logger.warning(f"Using UNSAFE pickle serialization for {type(value).__name__}")
            return pickle.dumps(value)
        else:
            return self._serialize_safe(value)

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value using configured method."""
        if self.dangerously_allow_pickle:
            import pickle
            self.logger.warning("Using UNSAFE pickle deserialization")
            return pickle.loads(data)
        else:
            return self._deserialize_safe(data)

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
            return self._deserialize(data)
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
            data = self._serialize(value)
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
        Clear keys matching pattern using SCAN for safety.

        Args:
            pattern: Key pattern (e.g., "embedding:*")

        Returns:
            Number of keys deleted
        """
        if self._client is None:
            return 0

        try:
            # Use SCAN instead of KEYS to avoid blocking Redis
            count = 0
            cursor = 0
            full_pattern = self._make_key(pattern)

            while True:
                cursor, keys = self._client.scan(cursor, match=full_pattern, count=100)
                if keys:
                    count += self._client.delete(*keys)
                if cursor == 0:
                    break

            return count
        except Exception as e:
            self.logger.error(f"Failed to clear pattern {pattern}: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
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
