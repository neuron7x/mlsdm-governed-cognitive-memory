"""Production-grade JSON structured logging with rotation support.

This module implements a Principal System Architect-level logging system
with structured JSON logs, multiple log levels, and automatic rotation.
"""

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from threading import Lock
from typing import Any


class EventType(Enum):
    """Types of cognitive system events to log."""

    # Moral governance events
    MORAL_REJECTED = "moral_rejected"
    MORAL_ACCEPTED = "moral_accepted"
    MORAL_THRESHOLD_ADJUSTED = "moral_threshold_adjusted"

    # Rhythm/sleep phase events
    SLEEP_PHASE_ENTERED = "sleep_phase_entered"
    WAKE_PHASE_ENTERED = "wake_phase_entered"
    PHASE_TRANSITION = "phase_transition"

    # Memory events
    MEMORY_FULL = "memory_full"
    MEMORY_STORE = "memory_store"
    MEMORY_RETRIEVE = "memory_retrieve"
    MEMORY_CONSOLIDATION = "memory_consolidation"

    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    SYSTEM_ERROR = "system_error"
    SYSTEM_WARNING = "system_warning"

    # Performance events
    PERFORMANCE_DEGRADATION = "performance_degradation"
    PROCESSING_TIME_EXCEEDED = "processing_time_exceeded"

    # General events
    EVENT_PROCESSED = "event_processed"
    STATE_CHANGE = "state_change"


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.

        Args:
            record: Log record to format

        Returns:
            JSON-formatted log string
        """
        # Extract custom fields if present
        event_type = getattr(record, "event_type", "unknown")
        correlation_id = getattr(record, "correlation_id", str(uuid.uuid4()))
        metrics = getattr(record, "metrics", {})

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "timestamp_unix": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "event_type": event_type,
            "correlation_id": correlation_id,
            "message": record.getMessage(),
            "metrics": metrics,  # Always include metrics (even if empty)
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
                "event_type",
                "correlation_id",
                "metrics",
            ]:
                log_entry[key] = value

        return json.dumps(log_entry)


class ObservabilityLogger:
    """Thread-safe observability logger with structured JSON output and rotation.

    Features:
    - JSON structured logs with timestamp, event_type, and metrics
    - Log levels: DEBUG, INFO, WARNING, ERROR
    - Log rotation by size and age
    - Thread-safe operation
    - Correlation IDs for request tracking
    - Production-grade error handling
    """

    def __init__(
        self,
        logger_name: str = "mlsdm_observability",
        log_dir: Path | str | None = None,
        log_file: str = "mlsdm_observability.log",
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5,
        max_age_days: int = 7,
        console_output: bool = True,
        min_level: int = logging.INFO,
    ):
        """Initialize observability logger.

        Args:
            logger_name: Name for the logger instance
            log_dir: Directory for log files (None for current directory)
            log_file: Name of the log file
            max_bytes: Maximum size of a log file before rotation
            backup_count: Number of backup files to keep
            max_age_days: Maximum age of log files in days
            console_output: Whether to output logs to console
            min_level: Minimum logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)  # Set to DEBUG to allow all levels
        self._lock = Lock()

        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Create log directory if specified
        if log_dir:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.log_dir / log_file
        else:
            log_path = Path(log_file)

        # Set up JSON formatter
        json_formatter = JSONFormatter()

        # Add rotating file handler (size-based rotation)
        file_handler = RotatingFileHandler(
            filename=str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(min_level)
        file_handler.setFormatter(json_formatter)
        self.logger.addHandler(file_handler)

        # Add time-based rotation handler (age-based rotation)
        time_handler = TimedRotatingFileHandler(
            filename=str(log_path).replace(".log", "_daily.log"),
            when="midnight",
            interval=1,
            backupCount=max_age_days,
            encoding="utf-8",
        )
        time_handler.setLevel(min_level)
        time_handler.setFormatter(json_formatter)
        self.logger.addHandler(time_handler)

        # Add console handler if requested
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(min_level)
            console_handler.setFormatter(json_formatter)
            self.logger.addHandler(console_handler)

        # Store configuration
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.max_age_days = max_age_days
        self.min_level = min_level

    def _log_event(
        self,
        event_type: EventType,
        level: int,
        message: str,
        correlation_id: str | None = None,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Internal method to log structured event.

        Args:
            event_type: Type of event being logged
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
            message: Human-readable message
            correlation_id: Optional correlation ID for request tracking
            metrics: Optional metrics dictionary
            **kwargs: Additional fields to include in the log

        Returns:
            Correlation ID used for this event
        """
        with self._lock:
            # Generate correlation ID if not provided
            if correlation_id is None:
                correlation_id = str(uuid.uuid4())

            # Create log record with extra fields
            extra = {
                "event_type": event_type.value,
                "correlation_id": correlation_id,
                "metrics": metrics or {},
            }

            # Add any additional kwargs
            extra.update(kwargs)

            # Log the event
            self.logger.log(level, message, extra=extra)

            return correlation_id

    def debug(
        self,
        event_type: EventType,
        message: str,
        correlation_id: str | None = None,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Log debug-level event.

        Args:
            event_type: Type of event
            message: Log message
            correlation_id: Optional correlation ID
            metrics: Optional metrics
            **kwargs: Additional fields

        Returns:
            Correlation ID
        """
        return self._log_event(
            event_type, logging.DEBUG, message, correlation_id, metrics, **kwargs
        )

    def info(
        self,
        event_type: EventType,
        message: str,
        correlation_id: str | None = None,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Log info-level event.

        Args:
            event_type: Type of event
            message: Log message
            correlation_id: Optional correlation ID
            metrics: Optional metrics
            **kwargs: Additional fields

        Returns:
            Correlation ID
        """
        return self._log_event(
            event_type, logging.INFO, message, correlation_id, metrics, **kwargs
        )

    def warn(
        self,
        event_type: EventType,
        message: str,
        correlation_id: str | None = None,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Log warning-level event.

        Args:
            event_type: Type of event
            message: Log message
            correlation_id: Optional correlation ID
            metrics: Optional metrics
            **kwargs: Additional fields

        Returns:
            Correlation ID
        """
        return self._log_event(
            event_type, logging.WARNING, message, correlation_id, metrics, **kwargs
        )

    def warning(
        self,
        event_type: EventType,
        message: str,
        correlation_id: str | None = None,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Log warning-level event (alias for warn).

        Args:
            event_type: Type of event
            message: Log message
            correlation_id: Optional correlation ID
            metrics: Optional metrics
            **kwargs: Additional fields

        Returns:
            Correlation ID
        """
        return self.warn(event_type, message, correlation_id, metrics, **kwargs)

    def error(
        self,
        event_type: EventType,
        message: str,
        correlation_id: str | None = None,
        metrics: dict[str, Any] | None = None,
        exc_info: bool = False,
        **kwargs: Any,
    ) -> str:
        """Log error-level event.

        Args:
            event_type: Type of event
            message: Log message
            correlation_id: Optional correlation ID
            metrics: Optional metrics
            exc_info: Whether to include exception info
            **kwargs: Additional fields

        Returns:
            Correlation ID
        """
        with self._lock:
            if correlation_id is None:
                correlation_id = str(uuid.uuid4())

            extra = {
                "event_type": event_type.value,
                "correlation_id": correlation_id,
                "metrics": metrics or {},
            }
            extra.update(kwargs)

            self.logger.error(message, exc_info=exc_info, extra=extra)

            return correlation_id

    # Convenience methods for common events

    def log_moral_rejected(
        self,
        moral_value: float,
        threshold: float,
        correlation_id: str | None = None,
    ) -> str:
        """Log moral value rejection event.

        Args:
            moral_value: The moral value that was rejected
            threshold: The threshold used for rejection
            correlation_id: Optional correlation ID

        Returns:
            Correlation ID
        """
        return self.warn(
            EventType.MORAL_REJECTED,
            f"Input rejected due to low moral value: {moral_value:.3f} < {threshold:.3f}",
            correlation_id=correlation_id,
            metrics={"moral_value": moral_value, "threshold": threshold},
        )

    def log_moral_accepted(
        self,
        moral_value: float,
        threshold: float,
        correlation_id: str | None = None,
    ) -> str:
        """Log moral value acceptance event.

        Args:
            moral_value: The moral value that was accepted
            threshold: The threshold used for acceptance
            correlation_id: Optional correlation ID

        Returns:
            Correlation ID
        """
        return self.info(
            EventType.MORAL_ACCEPTED,
            f"Input accepted with moral value: {moral_value:.3f} >= {threshold:.3f}",
            correlation_id=correlation_id,
            metrics={"moral_value": moral_value, "threshold": threshold},
        )

    def log_sleep_phase_entered(
        self,
        previous_phase: str,
        correlation_id: str | None = None,
    ) -> str:
        """Log transition to sleep phase.

        Args:
            previous_phase: The previous phase (typically "wake")
            correlation_id: Optional correlation ID

        Returns:
            Correlation ID
        """
        return self.info(
            EventType.SLEEP_PHASE_ENTERED,
            f"Cognitive rhythm transitioned to sleep phase from {previous_phase}",
            correlation_id=correlation_id,
            metrics={"previous_phase": previous_phase, "new_phase": "sleep"},
        )

    def log_wake_phase_entered(
        self,
        previous_phase: str,
        correlation_id: str | None = None,
    ) -> str:
        """Log transition to wake phase.

        Args:
            previous_phase: The previous phase (typically "sleep")
            correlation_id: Optional correlation ID

        Returns:
            Correlation ID
        """
        return self.info(
            EventType.WAKE_PHASE_ENTERED,
            f"Cognitive rhythm transitioned to wake phase from {previous_phase}",
            correlation_id=correlation_id,
            metrics={"previous_phase": previous_phase, "new_phase": "wake"},
        )

    def log_memory_full(
        self,
        current_size: int,
        capacity: int,
        memory_mb: float,
        correlation_id: str | None = None,
    ) -> str:
        """Log memory capacity reached event.

        Args:
            current_size: Current number of items in memory
            capacity: Maximum capacity
            memory_mb: Memory usage in megabytes
            correlation_id: Optional correlation ID

        Returns:
            Correlation ID
        """
        return self.warn(
            EventType.MEMORY_FULL,
            f"Memory capacity reached: {current_size}/{capacity} items ({memory_mb:.2f} MB)",
            correlation_id=correlation_id,
            metrics={
                "current_size": current_size,
                "capacity": capacity,
                "memory_mb": memory_mb,
                "utilization_percent": (current_size / capacity) * 100,
            },
        )

    def log_memory_store(
        self,
        vector_dim: int,
        memory_size: int,
        correlation_id: str | None = None,
    ) -> str:
        """Log memory storage event.

        Args:
            vector_dim: Dimension of the stored vector
            memory_size: Current size of memory after storage
            correlation_id: Optional correlation ID

        Returns:
            Correlation ID
        """
        return self.debug(
            EventType.MEMORY_STORE,
            f"Stored vector of dimension {vector_dim}, memory size now {memory_size}",
            correlation_id=correlation_id,
            metrics={"vector_dim": vector_dim, "memory_size": memory_size},
        )

    def log_processing_time_exceeded(
        self,
        processing_time_ms: float,
        threshold_ms: float,
        correlation_id: str | None = None,
    ) -> str:
        """Log processing time threshold exceeded.

        Args:
            processing_time_ms: Actual processing time in milliseconds
            threshold_ms: Threshold that was exceeded
            correlation_id: Optional correlation ID

        Returns:
            Correlation ID
        """
        return self.warn(
            EventType.PROCESSING_TIME_EXCEEDED,
            f"Processing time exceeded threshold: {processing_time_ms:.2f}ms > {threshold_ms:.2f}ms",
            correlation_id=correlation_id,
            metrics={
                "processing_time_ms": processing_time_ms,
                "threshold_ms": threshold_ms,
                "overage_ms": processing_time_ms - threshold_ms,
            },
        )

    def log_system_startup(
        self,
        version: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> str:
        """Log system startup event.

        Args:
            version: Optional system version
            config: Optional configuration summary

        Returns:
            Correlation ID
        """
        metrics: dict[str, Any] = {}
        if version:
            metrics["version"] = version
        if config:
            metrics["config"] = config

        return self.info(
            EventType.SYSTEM_STARTUP,
            "MLSDM system starting up",
            metrics=metrics,
        )

    def log_system_shutdown(
        self,
        reason: str | None = None,
    ) -> str:
        """Log system shutdown event.

        Args:
            reason: Optional shutdown reason

        Returns:
            Correlation ID
        """
        metrics: dict[str, Any] = {}
        if reason:
            metrics["reason"] = reason

        return self.info(
            EventType.SYSTEM_SHUTDOWN,
            "MLSDM system shutting down",
            metrics=metrics,
        )

    def get_config(self) -> dict[str, Any]:
        """Get logger configuration.

        Returns:
            Dictionary with logger configuration
        """
        return {
            "logger_name": self.logger.name,
            "max_bytes": self.max_bytes,
            "backup_count": self.backup_count,
            "max_age_days": self.max_age_days,
            "min_level": logging.getLevelName(self.min_level),
            "handlers": len(self.logger.handlers),
        }


# Global instance for convenience
_observability_logger: ObservabilityLogger | None = None


def get_observability_logger(
    logger_name: str = "mlsdm_observability",
    **kwargs: Any,
) -> ObservabilityLogger:
    """Get or create the observability logger instance.

    Args:
        logger_name: Name for the logger instance
        **kwargs: Additional arguments passed to ObservabilityLogger constructor

    Returns:
        ObservabilityLogger instance
    """
    global _observability_logger

    if _observability_logger is None:
        _observability_logger = ObservabilityLogger(logger_name=logger_name, **kwargs)

    return _observability_logger
