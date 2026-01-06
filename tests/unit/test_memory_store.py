"""
Unit Tests for Memory Store

Tests for memory store protocols and utilities.
"""

from mlsdm.memory.store import compute_content_hash


class TestComputeContentHash:
    """Test compute_content_hash function."""

    def test_compute_content_hash_basic(self):
        """Test that compute_content_hash returns a valid SHA256 hash."""
        content = "test content"
        hash_result = compute_content_hash(content)

        # SHA256 hash should be 64 characters (hex)
        assert isinstance(hash_result, str)
        assert len(hash_result) == 64
        # Should be all hex characters
        assert all(c in "0123456789abcdef" for c in hash_result)

    def test_compute_content_hash_deterministic(self):
        """Test that compute_content_hash is deterministic."""
        content = "test content"
        hash1 = compute_content_hash(content)
        hash2 = compute_content_hash(content)

        assert hash1 == hash2

    def test_compute_content_hash_different_content(self):
        """Test that different content produces different hashes."""
        hash1 = compute_content_hash("content1")
        hash2 = compute_content_hash("content2")

        assert hash1 != hash2

    def test_compute_content_hash_empty_string(self):
        """Test compute_content_hash with empty string."""
        hash_result = compute_content_hash("")

        assert isinstance(hash_result, str)
        assert len(hash_result) == 64
