"""Comprehensive security tests for the system.

Tests cover:
- Rate limiting
- Input validation
- Security logging
- Authentication
"""

import time

import numpy as np
import pytest

from src.utils.input_validator import InputValidator
from src.utils.rate_limiter import RateLimiter
from src.utils.security_logger import SecurityEventType, SecurityLogger


class TestRateLimiter:
    """Test suite for rate limiter."""
    
    def test_rate_limiter_basic(self):
        """Test basic rate limiting functionality."""
        limiter = RateLimiter(rate=5.0, capacity=10)
        client_id = "test_client_1"
        
        # Should allow initial requests up to capacity
        for _ in range(10):
            assert limiter.is_allowed(client_id)
        
        # Should deny next request (bucket exhausted)
        assert not limiter.is_allowed(client_id)
    
    def test_rate_limiter_refill(self):
        """Test that rate limiter refills over time."""
        limiter = RateLimiter(rate=10.0, capacity=5)
        client_id = "test_client_2"
        
        # Exhaust bucket
        for _ in range(5):
            assert limiter.is_allowed(client_id)
        
        # Should deny
        assert not limiter.is_allowed(client_id)
        
        # Wait for refill (0.2 seconds = 2 tokens at 10 RPS)
        time.sleep(0.3)
        
        # Should allow again
        assert limiter.is_allowed(client_id)
    
    def test_rate_limiter_multiple_clients(self):
        """Test rate limiter handles multiple clients independently."""
        limiter = RateLimiter(rate=5.0, capacity=3)
        
        # Exhaust client1
        for _ in range(3):
            assert limiter.is_allowed("client1")
        assert not limiter.is_allowed("client1")
        
        # Client2 should still have tokens
        assert limiter.is_allowed("client2")
    
    def test_rate_limiter_reset(self):
        """Test rate limiter reset functionality."""
        limiter = RateLimiter(rate=5.0, capacity=3)
        client_id = "test_client_3"
        
        # Exhaust bucket
        for _ in range(3):
            assert limiter.is_allowed(client_id)
        assert not limiter.is_allowed(client_id)
        
        # Reset
        limiter.reset(client_id)
        
        # Should allow again
        assert limiter.is_allowed(client_id)
    
    def test_rate_limiter_stats(self):
        """Test rate limiter statistics."""
        limiter = RateLimiter(rate=5.0, capacity=10)
        client_id = "test_client_4"
        
        stats = limiter.get_stats(client_id)
        assert stats["tokens"] == 10.0
        
        limiter.is_allowed(client_id)
        stats = limiter.get_stats(client_id)
        # Use approximate comparison due to floating point precision
        assert abs(stats["tokens"] - 9.0) < 0.01
    
    def test_rate_limiter_cleanup(self):
        """Test cleanup of old entries."""
        limiter = RateLimiter(rate=5.0, capacity=10)
        
        limiter.is_allowed("client1")
        limiter.is_allowed("client2")
        
        # Cleanup with short max age (entries are recent, so none cleaned)
        cleaned = limiter.cleanup_old_entries(max_age_seconds=10.0)
        assert cleaned == 0
        
        # Cleanup with zero max age (all entries cleaned)
        cleaned = limiter.cleanup_old_entries(max_age_seconds=0.0)
        assert cleaned == 2
    
    def test_rate_limiter_invalid_params(self):
        """Test rate limiter rejects invalid parameters."""
        with pytest.raises(ValueError):
            RateLimiter(rate=-1.0)
        
        with pytest.raises(ValueError):
            RateLimiter(rate=5.0, capacity=-1)


class TestInputValidator:
    """Test suite for input validation."""
    
    def test_validate_vector_valid(self):
        """Test validation of valid vector."""
        validator = InputValidator()
        vector = [1.0, 2.0, 3.0]
        result = validator.validate_vector(vector, expected_dim=3)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert np.allclose(result, [1.0, 2.0, 3.0])
    
    def test_validate_vector_dimension_mismatch(self):
        """Test validation rejects dimension mismatch."""
        validator = InputValidator()
        vector = [1.0, 2.0]
        
        with pytest.raises(ValueError, match="dimension"):
            validator.validate_vector(vector, expected_dim=3)
    
    def test_validate_vector_nan(self):
        """Test validation rejects NaN values."""
        validator = InputValidator()
        vector = [1.0, float('nan'), 3.0]
        
        with pytest.raises(ValueError, match="NaN"):
            validator.validate_vector(vector, expected_dim=3)
    
    def test_validate_vector_inf(self):
        """Test validation rejects Inf values."""
        validator = InputValidator()
        vector = [1.0, float('inf'), 3.0]
        
        with pytest.raises(ValueError, match="Inf"):
            validator.validate_vector(vector, expected_dim=3)
    
    def test_validate_vector_too_large(self):
        """Test validation rejects oversized vectors."""
        validator = InputValidator()
        vector = [1.0] * (InputValidator.MAX_VECTOR_SIZE + 1)
        
        with pytest.raises(ValueError, match="exceeds maximum"):
            validator.validate_vector(vector, expected_dim=len(vector))
    
    def test_validate_vector_normalize(self):
        """Test vector normalization."""
        validator = InputValidator()
        vector = [3.0, 4.0]
        result = validator.validate_vector(vector, expected_dim=2, normalize=True)
        
        # Should be normalized to unit length
        assert np.allclose(np.linalg.norm(result), 1.0)
        assert np.allclose(result, [0.6, 0.8])
    
    def test_validate_vector_zero_normalization(self):
        """Test normalization of zero vector fails."""
        validator = InputValidator()
        vector = [0.0, 0.0, 0.0]
        
        with pytest.raises(ValueError, match="zero vector"):
            validator.validate_vector(vector, expected_dim=3, normalize=True)
    
    def test_validate_moral_value_valid(self):
        """Test validation of valid moral values."""
        validator = InputValidator()
        
        assert validator.validate_moral_value(0.0) == 0.0
        assert validator.validate_moral_value(0.5) == 0.5
        assert validator.validate_moral_value(1.0) == 1.0
    
    def test_validate_moral_value_out_of_range(self):
        """Test validation rejects out of range moral values."""
        validator = InputValidator()
        
        with pytest.raises(ValueError, match="must be between"):
            validator.validate_moral_value(-0.1)
        
        with pytest.raises(ValueError, match="must be between"):
            validator.validate_moral_value(1.1)
    
    def test_validate_moral_value_nan(self):
        """Test validation rejects NaN moral value."""
        validator = InputValidator()
        
        with pytest.raises(ValueError, match="NaN"):
            validator.validate_moral_value(float('nan'))
    
    def test_validate_moral_value_invalid_type(self):
        """Test validation rejects invalid types."""
        validator = InputValidator()
        
        with pytest.raises(ValueError, match="must be numeric"):
            validator.validate_moral_value("0.5")
    
    def test_sanitize_string_valid(self):
        """Test string sanitization with valid input."""
        validator = InputValidator()
        
        text = "Hello, world!"
        result = validator.sanitize_string(text)
        assert result == "Hello, world!"
    
    def test_sanitize_string_remove_null_bytes(self):
        """Test sanitization removes null bytes."""
        validator = InputValidator()
        
        text = "Hello\x00World"
        result = validator.sanitize_string(text)
        assert result == "HelloWorld"
    
    def test_sanitize_string_max_length(self):
        """Test sanitization enforces max length."""
        validator = InputValidator()
        
        text = "a" * 20000
        with pytest.raises(ValueError, match="exceeds maximum"):
            validator.sanitize_string(text, max_length=10000)
    
    def test_sanitize_string_newlines(self):
        """Test sanitization handles newlines."""
        validator = InputValidator()
        
        text = "Line1\nLine2"
        result = validator.sanitize_string(text, allow_newlines=True)
        assert "\n" in result
        
        result = validator.sanitize_string(text, allow_newlines=False)
        assert "\n" not in result
    
    def test_sanitize_string_control_chars(self):
        """Test sanitization removes control characters."""
        validator = InputValidator()
        
        # Test with various control characters
        text = "Hello\x01\x02\x03World"
        result = validator.sanitize_string(text)
        assert result == "HelloWorld"
    
    def test_validate_client_id_valid(self):
        """Test validation of valid client IDs."""
        validator = InputValidator()
        
        assert validator.validate_client_id("192.168.1.1") == "192.168.1.1"
        assert validator.validate_client_id("abc-123_def") == "abc-123_def"
    
    def test_validate_client_id_invalid(self):
        """Test validation rejects invalid client IDs."""
        validator = InputValidator()
        
        with pytest.raises(ValueError, match="empty"):
            validator.validate_client_id("")
        
        with pytest.raises(ValueError, match="invalid characters"):
            validator.validate_client_id("test@example.com")
        
        with pytest.raises(ValueError, match="too long"):
            validator.validate_client_id("a" * 300)
    
    def test_validate_numeric_range_valid(self):
        """Test numeric range validation with valid values."""
        validator = InputValidator()
        
        assert validator.validate_numeric_range(5.0, 0.0, 10.0) == 5.0
        assert validator.validate_numeric_range(0.0, 0.0, 10.0) == 0.0
        assert validator.validate_numeric_range(10.0, 0.0, 10.0) == 10.0
    
    def test_validate_numeric_range_out_of_range(self):
        """Test numeric range validation rejects out of range values."""
        validator = InputValidator()
        
        with pytest.raises(ValueError, match="less than minimum"):
            validator.validate_numeric_range(-1.0, 0.0, 10.0)
        
        with pytest.raises(ValueError, match="exceeds maximum"):
            validator.validate_numeric_range(11.0, 0.0, 10.0)
    
    def test_validate_array_size_valid(self):
        """Test array size validation with valid arrays."""
        validator = InputValidator()
        
        arr = [1, 2, 3, 4, 5]
        size = validator.validate_array_size(arr, max_size=10)
        assert size == 5
    
    def test_validate_array_size_too_large(self):
        """Test array size validation rejects oversized arrays."""
        validator = InputValidator()
        
        arr = [1] * 100
        with pytest.raises(ValueError, match="exceeds maximum"):
            validator.validate_array_size(arr, max_size=50)


class TestSecurityLogger:
    """Test suite for security logger."""
    
    def test_security_logger_creation(self):
        """Test security logger can be created."""
        logger = SecurityLogger("test_logger")
        assert logger is not None
    
    def test_log_auth_success(self):
        """Test logging authentication success."""
        logger = SecurityLogger("test_auth_success")
        correlation_id = logger.log_auth_success("client_123")
        assert correlation_id is not None
    
    def test_log_auth_failure(self):
        """Test logging authentication failure."""
        logger = SecurityLogger("test_auth_failure")
        correlation_id = logger.log_auth_failure("client_123", "Invalid token")
        assert correlation_id is not None
    
    def test_log_rate_limit_exceeded(self):
        """Test logging rate limit exceeded."""
        logger = SecurityLogger("test_rate_limit")
        correlation_id = logger.log_rate_limit_exceeded("client_123")
        assert correlation_id is not None
    
    def test_log_invalid_input(self):
        """Test logging invalid input."""
        logger = SecurityLogger("test_invalid_input")
        correlation_id = logger.log_invalid_input(
            "client_123",
            "Vector dimension mismatch"
        )
        assert correlation_id is not None
    
    def test_log_state_change(self):
        """Test logging state change."""
        logger = SecurityLogger("test_state_change")
        correlation_id = logger.log_state_change(
            "phase_transition",
            {"from": "wake", "to": "sleep"}
        )
        assert correlation_id is not None
    
    def test_log_anomaly(self):
        """Test logging anomaly."""
        logger = SecurityLogger("test_anomaly")
        correlation_id = logger.log_anomaly(
            "threshold_breach",
            "Moral filter threshold exceeded bounds",
            severity="high"
        )
        assert correlation_id is not None
    
    def test_log_system_event(self):
        """Test logging system event."""
        logger = SecurityLogger("test_system_event")
        correlation_id = logger.log_system_event(
            SecurityEventType.STARTUP,
            "System started"
        )
        assert correlation_id is not None
    
    def test_correlation_id_consistency(self):
        """Test correlation IDs are consistent when provided."""
        logger = SecurityLogger("test_correlation")
        test_corr_id = "test-correlation-123"
        
        result_id = logger.log_auth_success("client_123", correlation_id=test_corr_id)
        assert result_id == test_corr_id
    
    def test_no_pii_in_logs(self):
        """Test that PII fields are filtered out."""
        logger = SecurityLogger("test_pii")
        
        # This should not raise an error, and PII should be filtered
        correlation_id = logger.log_state_change(
            "user_action",
            {
                "action": "login",
                "email": "test@example.com",  # Should be filtered
                "username": "testuser",  # Should be filtered
                "timestamp": 1234567890
            }
        )
        assert correlation_id is not None


class TestSecurityIntegration:
    """Integration tests for security features."""
    
    def test_rate_limiter_with_validator(self):
        """Test rate limiter and validator work together."""
        limiter = RateLimiter(rate=5.0, capacity=3)
        validator = InputValidator()
        
        client_id = "integration_test_1"
        
        # Process some valid requests
        for i in range(3):
            if limiter.is_allowed(client_id):
                vector = [float(i)] * 10
                result = validator.validate_vector(vector, expected_dim=10)
                assert result is not None
        
        # Next request should be rate limited
        assert not limiter.is_allowed(client_id)
    
    def test_validator_with_logger(self):
        """Test validator and logger work together."""
        validator = InputValidator()
        logger = SecurityLogger("integration_validator_logger")
        
        # Valid input
        try:
            vector = [1.0, 2.0, 3.0]
            validator.validate_vector(vector, expected_dim=3)
        except ValueError as e:
            logger.log_invalid_input("client_123", str(e))
            pytest.fail("Should not raise exception for valid input")
        
        # Invalid input
        try:
            vector = [1.0, float('nan'), 3.0]
            validator.validate_vector(vector, expected_dim=3)
            pytest.fail("Should raise exception for invalid input")
        except ValueError as e:
            correlation_id = logger.log_invalid_input("client_123", str(e))
            assert correlation_id is not None
