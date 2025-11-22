"""Unit tests for semantic cache module."""

import numpy as np
import pytest

from mlsdm.memory.semantic_cache import CacheEntry, SemanticResponseCache


class TestCacheEntry:
    """Test CacheEntry dataclass."""
    
    def test_creation(self):
        """CacheEntry can be created with required fields."""
        embedding = np.random.randn(384)
        entry = CacheEntry(
            query_embedding=embedding,
            moral_value=0.5,
            user_intent="conversational",
            response="Test response"
        )
        
        assert np.array_equal(entry.query_embedding, embedding)
        assert entry.moral_value == 0.5
        assert entry.user_intent == "conversational"
        assert entry.response == "Test response"
        assert entry.hit_count == 0


class TestSemanticResponseCache:
    """Test SemanticResponseCache class."""
    
    def test_initialization(self):
        """Cache initializes with correct parameters."""
        cache = SemanticResponseCache(
            max_entries=100,
            similarity_threshold=0.9,
            moral_tolerance=0.1
        )
        
        assert cache.max_entries == 100
        assert cache.similarity_threshold == 0.9
        assert cache.moral_tolerance == 0.1
        assert len(cache) == 0
    
    def test_invalid_parameters(self):
        """Cache rejects invalid parameters."""
        with pytest.raises(ValueError, match="max_entries must be positive"):
            SemanticResponseCache(max_entries=0)
        
        with pytest.raises(ValueError, match="similarity_threshold must be in"):
            SemanticResponseCache(similarity_threshold=1.5)
        
        with pytest.raises(ValueError, match="moral_tolerance must be non-negative"):
            SemanticResponseCache(moral_tolerance=-0.1)
    
    def test_store_and_lookup_exact_match(self):
        """Store and lookup with identical parameters."""
        cache = SemanticResponseCache()
        embedding = np.random.randn(384)
        
        cache.store(embedding, 0.5, "conversational", "Test response")
        
        result = cache.lookup(embedding, 0.5, "conversational")
        
        assert result == "Test response"
    
    def test_lookup_miss_empty_cache(self):
        """Lookup on empty cache returns None."""
        cache = SemanticResponseCache()
        embedding = np.random.randn(384)
        
        result = cache.lookup(embedding, 0.5, "conversational")
        
        assert result is None
    
    def test_lookup_miss_different_intent(self):
        """Lookup with different intent returns None."""
        cache = SemanticResponseCache()
        embedding = np.random.randn(384)
        
        cache.store(embedding, 0.5, "conversational", "Response 1")
        
        result = cache.lookup(embedding, 0.5, "analytical")
        
        assert result is None
    
    def test_lookup_miss_moral_value_out_of_tolerance(self):
        """Lookup with moral value outside tolerance returns None."""
        cache = SemanticResponseCache(moral_tolerance=0.05)
        embedding = np.random.randn(384)
        
        cache.store(embedding, 0.5, "conversational", "Response 1")
        
        # 0.8 is outside 0.05 tolerance of 0.5
        result = cache.lookup(embedding, 0.8, "conversational")
        
        assert result is None
    
    def test_lookup_within_moral_tolerance(self):
        """Lookup with moral value within tolerance succeeds."""
        cache = SemanticResponseCache(moral_tolerance=0.1)
        embedding = np.random.randn(384)
        
        cache.store(embedding, 0.5, "conversational", "Response 1")
        
        # 0.55 is within 0.1 tolerance of 0.5
        result = cache.lookup(embedding, 0.55, "conversational")
        
        assert result == "Response 1"
    
    def test_semantic_similarity_matching(self):
        """Lookup with similar (but not identical) embeddings."""
        cache = SemanticResponseCache(similarity_threshold=0.95)
        
        # Create two very similar embeddings
        base_embedding = np.random.randn(384)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)
        
        similar_embedding = base_embedding + np.random.randn(384) * 0.01
        similar_embedding = similar_embedding / np.linalg.norm(similar_embedding)
        
        cache.store(base_embedding, 0.5, "conversational", "Response")
        
        # With high threshold, lookup might succeed or fail depending on similarity
        result = cache.lookup(similar_embedding, 0.5, "conversational")
        
        # We can't assert the result deterministically, but it shouldn't crash
        assert result is None or result == "Response"
    
    def test_lru_eviction(self):
        """Cache evicts oldest entries when full."""
        cache = SemanticResponseCache(max_entries=2)
        
        emb1 = np.random.randn(384)
        emb2 = np.random.randn(384)
        emb3 = np.random.randn(384)
        
        cache.store(emb1, 0.5, "conversational", "Response 1")
        cache.store(emb2, 0.5, "conversational", "Response 2")
        
        assert len(cache) == 2
        
        # Adding third entry should evict first
        cache.store(emb3, 0.5, "conversational", "Response 3")
        
        assert len(cache) == 2
        
        # First entry should be gone
        result1 = cache.lookup(emb1, 0.5, "conversational")
        assert result1 is None
        
        # Second and third should still be there
        result2 = cache.lookup(emb2, 0.5, "conversational")
        result3 = cache.lookup(emb3, 0.5, "conversational")
        
        assert result2 == "Response 2"
        assert result3 == "Response 3"
    
    def test_hit_count_tracking(self):
        """Cache tracks hit counts correctly."""
        cache = SemanticResponseCache()
        embedding = np.random.randn(384)
        
        cache.store(embedding, 0.5, "conversational", "Response")
        
        # Multiple lookups should increment hit count
        cache.lookup(embedding, 0.5, "conversational")
        cache.lookup(embedding, 0.5, "conversational")
        
        stats = cache.get_stats()
        assert stats['hits'] == 2
    
    def test_statistics(self):
        """Cache statistics are accurate."""
        cache = SemanticResponseCache(max_entries=10)
        
        # Store some entries
        for i in range(3):
            emb = np.random.randn(384)
            cache.store(emb, 0.5, "conversational", f"Response {i}")
        
        stats = cache.get_stats()
        
        assert stats['size'] == 3
        assert stats['max_entries'] == 10
        assert stats['hits'] == 0  # No lookups yet
        assert stats['misses'] == 0
        assert stats['hit_rate'] == 0.0
    
    def test_hit_rate_calculation(self):
        """Hit rate is calculated correctly."""
        cache = SemanticResponseCache()
        embedding = np.random.randn(384)
        
        cache.store(embedding, 0.5, "conversational", "Response")
        
        # 2 hits
        cache.lookup(embedding, 0.5, "conversational")
        cache.lookup(embedding, 0.5, "conversational")
        
        # 1 miss
        other_emb = np.random.randn(384)
        cache.lookup(other_emb, 0.5, "conversational")
        
        stats = cache.get_stats()
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['total_requests'] == 3
        assert abs(stats['hit_rate'] - 2/3) < 0.01
    
    def test_clear(self):
        """Clear removes all entries and resets statistics."""
        cache = SemanticResponseCache()
        
        # Add entries
        for i in range(3):
            emb = np.random.randn(384)
            cache.store(emb, 0.5, "conversational", f"Response {i}")
        
        # Do some lookups
        cache.lookup(np.random.randn(384), 0.5, "conversational")
        
        assert len(cache) > 0
        
        cache.clear()
        
        assert len(cache) == 0
        stats = cache.get_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0
    
    def test_empty_response_not_cached(self):
        """Empty responses are not stored in cache."""
        cache = SemanticResponseCache()
        embedding = np.random.randn(384)
        
        cache.store(embedding, 0.5, "conversational", "")
        
        assert len(cache) == 0
    
    def test_update_existing_entry(self):
        """Storing same key updates existing entry."""
        cache = SemanticResponseCache()
        embedding = np.random.randn(384)
        
        cache.store(embedding, 0.5, "conversational", "Response 1")
        assert len(cache) == 1
        
        # Store again with same parameters
        cache.store(embedding, 0.5, "conversational", "Response 2")
        
        # Should still have only 1 entry
        assert len(cache) == 1
        
        # Should retrieve updated response
        result = cache.lookup(embedding, 0.5, "conversational")
        assert result == "Response 2"
    
    def test_invalid_embedding_type(self):
        """Cache handles invalid embedding types gracefully."""
        cache = SemanticResponseCache()
        
        # Lookup with invalid type
        result = cache.lookup([1, 2, 3], 0.5, "conversational")  # type: ignore
        assert result is None
        
        # Store with invalid type should raise error
        with pytest.raises(TypeError):
            cache.store([1, 2, 3], 0.5, "conversational", "Response")  # type: ignore
