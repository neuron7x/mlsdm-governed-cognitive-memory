"""
MLSDM Utility modules.

Provides common utilities for the MLSDM framework including:
- Bulkhead pattern for fault isolation
- Rate limiting
- Embedding cache for performance optimization
- Configuration management
- Error handling
- Input validation
- Security logging
"""

from .bulkhead import (
    Bulkhead,
    BulkheadCompartment,
    BulkheadConfig,
    BulkheadFullError,
    BulkheadStats,
)
from .embedding_cache import (
    EmbeddingCache,
    EmbeddingCacheConfig,
    EmbeddingCacheStats,
    clear_default_cache,
    get_default_cache,
)
from .rate_limiter import RateLimiter

__all__ = [
    # Bulkhead pattern
    "Bulkhead",
    "BulkheadCompartment",
    "BulkheadConfig",
    "BulkheadFullError",
    "BulkheadStats",
    # Embedding cache
    "EmbeddingCache",
    "EmbeddingCacheConfig",
    "EmbeddingCacheStats",
    "get_default_cache",
    "clear_default_cache",
    # Rate limiting
    "RateLimiter",
]
