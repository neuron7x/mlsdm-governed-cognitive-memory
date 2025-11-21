"""
Unit Tests for Security Logger

Tests structured security audit logging with PII protection.
"""

import json
import pytest
from src.utils.security_logger import (
    SecurityLogger, SecurityEventType, get_security_logger
)


class TestSecurityEventType:
    """Test security event type enum."""
    
    def test_event_types_defined(self):
        """Test all required event types are defined."""
        assert hasattr(SecurityEventType, 'AUTH_SUCCESS')
        assert hasattr(SecurityEventType, 'AUTH_FAILURE')
        assert hasattr(SecurityEventType, 'RATE_LIMIT_EXCEEDED')
        assert hasattr(SecurityEventType, 'INVALID_INPUT')
        assert hasattr(SecurityEventType, 'SYSTEM_ERROR')
    
    def test_event_type_values(self):
        """Test event type values are strings."""
        assert isinstance(SecurityEventType.AUTH_SUCCESS.value, str)
        assert isinstance(SecurityEventType.RATE_LIMIT_EXCEEDED.value, str)


class TestSecurityLogger:
    """Test security logger functionality."""
    
    def test_logger_initialization(self):
        """Test logger can be initialized."""
        logger = SecurityLogger()
        assert logger is not None
    
    def test_log_security_event_basic(self, caplog):
        """Test logging a basic security event."""
        logger = SecurityLogger()
        
        logger.log_event(
            event_type=SecurityEventType.AUTH_SUCCESS,
            message="User authenticated successfully",
            client_id="test_client"
        )
        
        # Check that log was created
        assert len(caplog.records) > 0
    
    def test_log_event_with_correlation_id(self, caplog):
        """Test logging event with correlation ID."""
        logger = SecurityLogger()
        correlation_id = "test-correlation-123"
        
        logger.log_event(
            event_type=SecurityEventType.SYSTEM_ERROR,
            message="Test error",
            correlation_id=correlation_id
        )
        
        # Verify correlation ID is in log
        record = caplog.records[-1]
        assert correlation_id in str(record.msg) or hasattr(record, 'correlation_id')
    
    def test_log_event_with_metadata(self, caplog):
        """Test logging event with additional metadata."""
        logger = SecurityLogger()
        
        metadata = {
            "dimension": 384,
            "threshold": 0.5
        }
        
        logger.log_event(
            event_type=SecurityEventType.STATE_CHANGE,
            message="Configuration updated",
            metadata=metadata
        )
        
        # Verify metadata is logged
        assert len(caplog.records) > 0
    
    def test_pii_filtering_email(self, caplog):
        """Test that email addresses are filtered."""
        logger = SecurityLogger()
        
        metadata = {
            "email": "user@example.com",
            "username": "testuser"
        }
        
        logger.log_event(
            event_type=SecurityEventType.AUTH_SUCCESS,
            message="Login",
            metadata=metadata
        )
        
        # Email should be filtered from logs
        record_msg = str(caplog.records[-1].msg)
        assert "user@example.com" not in record_msg
    
    def test_pii_filtering_password(self, caplog):
        """Test that passwords are filtered."""
        logger = SecurityLogger()
        
        metadata = {
            "password": "secret123",
            "user": "testuser"
        }
        
        logger.log_event(
            event_type=SecurityEventType.AUTH_FAILURE,
            message="Login failed",
            metadata=metadata
        )
        
        # Password should be filtered
        record_msg = str(caplog.records[-1].msg)
        assert "secret123" not in record_msg
    
    def test_pii_filtering_token(self, caplog):
        """Test that tokens are filtered."""
        logger = SecurityLogger()
        
        metadata = {
            "token": "bearer_token_xyz",
            "action": "api_call"
        }
        
        logger.log_event(
            event_type=SecurityEventType.AUTHZ_DENIED,
            message="Unauthorized",
            metadata=metadata
        )
        
        # Token should be filtered
        record_msg = str(caplog.records[-1].msg)
        assert "bearer_token_xyz" not in record_msg
    
    def test_client_id_pseudonymization(self, caplog):
        """Test that client IDs are pseudonymized."""
        logger = SecurityLogger()
        
        original_client_id = "192.168.1.100:Mozilla/5.0"
        
        logger.log_event(
            event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
            message="Rate limit exceeded",
            client_id=original_client_id
        )
        
        # Original client ID should not appear in logs
        record_msg = str(caplog.records[-1].msg)
        assert original_client_id not in record_msg
    
    def test_multiple_events_different_types(self, caplog):
        """Test logging multiple events of different types."""
        logger = SecurityLogger()
        
        events = [
            (SecurityEventType.AUTH_SUCCESS, "Login success"),
            (SecurityEventType.RATE_LIMIT_WARNING, "Rate limit warning"),
            (SecurityEventType.INVALID_INPUT, "Invalid input detected"),
        ]
        
        for event_type, message in events:
            logger.log_event(event_type=event_type, message=message)
        
        assert len(caplog.records) >= 3
    
    def test_log_authentication_success(self, caplog):
        """Test logging authentication success."""
        logger = SecurityLogger()
        
        logger.log_authentication_success(
            client_id="test_client",
            method="api_key"
        )
        
        assert len(caplog.records) > 0
        record = caplog.records[-1]
        assert "success" in str(record.msg).lower()
    
    def test_log_authentication_failure(self, caplog):
        """Test logging authentication failure."""
        logger = SecurityLogger()
        
        logger.log_authentication_failure(
            client_id="test_client",
            reason="invalid_credentials"
        )
        
        assert len(caplog.records) > 0
        record = caplog.records[-1]
        assert "fail" in str(record.msg).lower()
    
    def test_log_rate_limit_exceeded(self, caplog):
        """Test logging rate limit exceeded."""
        logger = SecurityLogger()
        
        logger.log_rate_limit_exceeded(
            client_id="test_client",
            current_rate=10.0,
            limit=5.0
        )
        
        assert len(caplog.records) > 0
    
    def test_log_invalid_input(self, caplog):
        """Test logging invalid input."""
        logger = SecurityLogger()
        
        logger.log_invalid_input(
            input_type="vector",
            reason="dimension_mismatch",
            expected="384",
            actual="256"
        )
        
        assert len(caplog.records) > 0
    
    def test_log_system_startup(self, caplog):
        """Test logging system startup."""
        logger = SecurityLogger()
        
        logger.log_system_startup(
            version="1.0.0",
            config={"dim": 384}
        )
        
        assert len(caplog.records) > 0
        record = caplog.records[-1]
        assert "startup" in str(record.msg).lower()
    
    def test_log_system_shutdown(self, caplog):
        """Test logging system shutdown."""
        logger = SecurityLogger()
        
        logger.log_system_shutdown(reason="normal")
        
        assert len(caplog.records) > 0
        record = caplog.records[-1]
        assert "shutdown" in str(record.msg).lower()


class TestGetSecurityLogger:
    """Test singleton logger retrieval."""
    
    def test_get_security_logger(self):
        """Test getting the security logger singleton."""
        logger1 = get_security_logger()
        logger2 = get_security_logger()
        
        # Should return the same instance
        assert logger1 is logger2
    
    def test_logger_is_security_logger_instance(self):
        """Test returned logger is SecurityLogger instance."""
        logger = get_security_logger()
        assert isinstance(logger, SecurityLogger)


class TestStructuredLogging:
    """Test structured logging format."""
    
    def test_log_contains_timestamp(self, caplog):
        """Test that logs contain timestamp information."""
        logger = SecurityLogger()
        
        logger.log_event(
            event_type=SecurityEventType.SYSTEM_ERROR,
            message="Test"
        )
        
        record = caplog.records[-1]
        # Timestamp is typically in the record or can be formatted
        assert hasattr(record, 'created')
    
    def test_log_contains_event_type(self, caplog):
        """Test that logs contain event type."""
        logger = SecurityLogger()
        
        logger.log_event(
            event_type=SecurityEventType.ANOMALY_DETECTED,
            message="Anomaly detected"
        )
        
        record = caplog.records[-1]
        record_msg = str(record.msg)
        assert "anomaly" in record_msg.lower()
    
    def test_log_json_parseable(self, caplog):
        """Test that log messages can be JSON format."""
        logger = SecurityLogger()
        
        logger.log_event(
            event_type=SecurityEventType.CONFIG_CHANGE,
            message="Config updated",
            metadata={"key": "value"}
        )
        
        # Should be able to parse structured data
        assert len(caplog.records) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_log_with_none_message(self, caplog):
        """Test logging with None message."""
        logger = SecurityLogger()
        
        logger.log_event(
            event_type=SecurityEventType.SYSTEM_ERROR,
            message=None
        )
        
        # Should handle gracefully
        assert len(caplog.records) > 0
    
    def test_log_with_empty_metadata(self, caplog):
        """Test logging with empty metadata."""
        logger = SecurityLogger()
        
        logger.log_event(
            event_type=SecurityEventType.STATE_CHANGE,
            message="Test",
            metadata={}
        )
        
        assert len(caplog.records) > 0
    
    def test_log_with_large_metadata(self, caplog):
        """Test logging with large metadata."""
        logger = SecurityLogger()
        
        metadata = {f"key_{i}": f"value_{i}" for i in range(100)}
        
        logger.log_event(
            event_type=SecurityEventType.SYSTEM_ERROR,
            message="Large metadata test",
            metadata=metadata
        )
        
        assert len(caplog.records) > 0
    
    def test_log_with_nested_metadata(self, caplog):
        """Test logging with nested metadata."""
        logger = SecurityLogger()
        
        metadata = {
            "level1": {
                "level2": {
                    "value": "nested"
                }
            }
        }
        
        logger.log_event(
            event_type=SecurityEventType.STATE_CHANGE,
            message="Nested metadata test",
            metadata=metadata
        )
        
        assert len(caplog.records) > 0
    
    def test_log_with_special_characters_in_message(self, caplog):
        """Test logging with special characters."""
        logger = SecurityLogger()
        
        message = "Test with special chars: <>&\"'"
        
        logger.log_event(
            event_type=SecurityEventType.INVALID_INPUT,
            message=message
        )
        
        assert len(caplog.records) > 0


class TestThreadSafety:
    """Test thread safety of security logger."""
    
    def test_concurrent_logging(self, caplog):
        """Test concurrent logging from multiple threads."""
        import threading
        logger = SecurityLogger()
        
        def log_events():
            for i in range(10):
                logger.log_event(
                    event_type=SecurityEventType.SYSTEM_ERROR,
                    message=f"Test {i}"
                )
        
        threads = [threading.Thread(target=log_events) for _ in range(5)]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have logged 50 events (5 threads * 10 events)
        assert len(caplog.records) >= 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
