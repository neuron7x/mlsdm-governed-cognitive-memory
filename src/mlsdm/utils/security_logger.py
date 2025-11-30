"""Security audit logging system.

This module provides structured logging for security events with correlation IDs
as specified in SECURITY_POLICY.md. No PII is logged.
"""

import json
import logging
import time
import uuid
from enum import Enum
from typing import Any


class SecurityEventType(Enum):
    """Types of security events to log."""

    # Authentication events
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    AUTH_MISSING = "auth_missing"

    # Authorization events
    AUTHZ_DENIED = "authorization_denied"

    # Rate limiting events
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    RATE_LIMIT_WARNING = "rate_limit_warning"

    # Input validation events
    INVALID_INPUT = "invalid_input"
    DIMENSION_MISMATCH = "dimension_mismatch"
    MORAL_VALUE_OUT_OF_RANGE = "moral_value_out_of_range"

    # State changes
    STATE_CHANGE = "state_change"
    CONFIG_CHANGE = "config_change"

    # Anomalies
    ANOMALY_DETECTED = "anomaly_detected"
    THRESHOLD_BREACH = "threshold_breach"

    # System events
    SYSTEM_ERROR = "system_error"
    STARTUP = "system_startup"
    SHUTDOWN = "system_shutdown"


class SecurityLogger:
    """Thread-safe security audit logger with structured JSON output."""

    def __init__(self, logger_name: str = "security_audit"):
        """Initialize security logger.

        Args:
            logger_name: Name for the logger instance
        """
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)

        # Ensure we have a handler
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(handler)

    def _log_event(
        self,
        event_type: SecurityEventType,
        level: int,
        message: str,
        correlation_id: str | None = None,
        client_id: str | None = None,
        additional_data: dict[str, Any] | None = None,
    ) -> str:
        """Internal method to log security event.

        Args:
            event_type: Type of security event
            level: Logging level (INFO, WARNING, ERROR)
            message: Human-readable message
            correlation_id: Optional correlation ID for request tracking
            client_id: Optional pseudonymized client identifier (no PII)
            additional_data: Additional structured data (no PII)

        Returns:
            Correlation ID used for this event
        """
        # Generate correlation ID if not provided
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())

        # Build structured log entry
        log_entry = {
            "timestamp": time.time(),
            "correlation_id": correlation_id,
            "event_type": event_type.value,
            "message": message,
        }

        # Add client ID if provided (pseudonymized, no PII)
        if client_id:
            log_entry["client_id"] = client_id

        # Add additional data if provided
        if additional_data:
            # Filter out any potential PII
            filtered_data = {
                k: v
                for k, v in additional_data.items()
                if k not in ["email", "username", "password", "token"]
            }
            log_entry["data"] = filtered_data

        # Log as structured JSON
        json_log = json.dumps(log_entry)
        self.logger.log(level, json_log)

        return correlation_id

    def log_auth_success(self, client_id: str, correlation_id: str | None = None) -> str:
        """Log successful authentication.

        Args:
            client_id: Pseudonymized client identifier
            correlation_id: Optional correlation ID

        Returns:
            Correlation ID
        """
        return self._log_event(
            SecurityEventType.AUTH_SUCCESS,
            logging.INFO,
            "Authentication successful",
            correlation_id=correlation_id,
            client_id=client_id,
        )

    def log_auth_failure(
        self, client_id: str, reason: str = "Invalid credentials", correlation_id: str | None = None
    ) -> str:
        """Log failed authentication attempt.

        Args:
            client_id: Pseudonymized client identifier
            reason: Reason for failure
            correlation_id: Optional correlation ID

        Returns:
            Correlation ID
        """
        return self._log_event(
            SecurityEventType.AUTH_FAILURE,
            logging.WARNING,
            f"Authentication failed: {reason}",
            correlation_id=correlation_id,
            client_id=client_id,
            additional_data={"reason": reason},
        )

    def log_rate_limit_exceeded(self, client_id: str, correlation_id: str | None = None) -> str:
        """Log rate limit exceeded event.

        Args:
            client_id: Pseudonymized client identifier
            correlation_id: Optional correlation ID

        Returns:
            Correlation ID
        """
        return self._log_event(
            SecurityEventType.RATE_LIMIT_EXCEEDED,
            logging.WARNING,
            "Rate limit exceeded",
            correlation_id=correlation_id,
            client_id=client_id,
        )

    def log_invalid_input(
        self, client_id: str, error_message: str, correlation_id: str | None = None
    ) -> str:
        """Log invalid input validation error.

        Args:
            client_id: Pseudonymized client identifier
            error_message: Validation error message
            correlation_id: Optional correlation ID

        Returns:
            Correlation ID
        """
        return self._log_event(
            SecurityEventType.INVALID_INPUT,
            logging.WARNING,
            f"Invalid input: {error_message}",
            correlation_id=correlation_id,
            client_id=client_id,
            additional_data={"error": error_message},
        )

    def log_state_change(
        self, change_type: str, details: dict[str, Any], correlation_id: str | None = None
    ) -> str:
        """Log important state change.

        Args:
            change_type: Type of state change
            details: Details about the change (no PII)
            correlation_id: Optional correlation ID

        Returns:
            Correlation ID
        """
        return self._log_event(
            SecurityEventType.STATE_CHANGE,
            logging.INFO,
            f"State change: {change_type}",
            correlation_id=correlation_id,
            additional_data={"change_type": change_type, "details": details},
        )

    def log_anomaly(
        self,
        anomaly_type: str,
        description: str,
        severity: str = "medium",
        correlation_id: str | None = None,
    ) -> str:
        """Log anomaly detection event.

        Args:
            anomaly_type: Type of anomaly detected
            description: Description of the anomaly
            severity: Severity level (low, medium, high)
            correlation_id: Optional correlation ID

        Returns:
            Correlation ID
        """
        level = logging.WARNING if severity in ["medium", "high"] else logging.INFO

        return self._log_event(
            SecurityEventType.ANOMALY_DETECTED,
            level,
            f"Anomaly detected: {anomaly_type}",
            correlation_id=correlation_id,
            additional_data={
                "anomaly_type": anomaly_type,
                "description": description,
                "severity": severity,
            },
        )

    def log_system_event(
        self,
        event_type: SecurityEventType,
        message: str,
        additional_data: dict[str, Any] | None = None,
    ) -> str:
        """Log system-level event.

        Args:
            event_type: Type of system event
            message: Event message
            additional_data: Additional data (no PII)

        Returns:
            Correlation ID
        """
        return self._log_event(event_type, logging.INFO, message, additional_data=additional_data)


# Global instance for convenience
_security_logger = SecurityLogger()


def get_security_logger() -> SecurityLogger:
    """Get the global security logger instance.

    Returns:
        Global SecurityLogger instance
    """
    return _security_logger
