"""Production-grade JSON structured logging with rotation support.

This module implements a Principal System Architect-level logging system
with structured JSON logs, multiple log levels, and automatic rotation.

Key features:
- Payload scrubbing to prevent PII/sensitive data in logs
- Mandatory fields for full observability (request_id, phase, etc.)
- Thread-safe singleton pattern
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from threading import Lock
from typing import Any

# ---------------------------------------------------------------------------
# Payload Scrubbing
# ---------------------------------------------------------------------------


def payload_scrubber(
    text: str,
    max_length: int = 50,
    mask_char: str = "*",
) -> str:
    """Scrub sensitive content from text for safe logging.

    This function masks user input and LLM responses to prevent
    PII or sensitive content from appearing in logs.

    Args:
        text: The text to scrub
        max_length: Maximum length of text to show (rest is masked)
        mask_char: Character to use for masking

    Returns:
        Scrubbed text safe for logging
    """
    if not text:
        return "[empty]"

    if not isinstance(text, str):
        return f"[non-string:{type(text).__name__}]"

    # Remove any newlines/tabs for log readability
    clean = re.sub(r"[\n\r\t]+", " ", text)

    # Truncate and mask if too long
    if len(clean) > max_length:
        visible_chars = max_length // 2
        return f"{clean[:visible_chars]}{mask_char * 3}[{len(clean)} chars]{mask_char * 3}{clean[-visible_chars:]}"

    return clean


def scrub_for_log(value: Any, max_length: int = 50) -> str:
    """Convert any value to a log-safe scrubbed string.

    Args:
        value: Any value to convert for logging
        max_length: Maximum length for string values

    Returns:
        Log-safe string representation
    """
    if value is None:
        return "[none]"
    if isinstance(value, str):
        return payload_scrubber(value, max_length)
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return f"[{type(value).__name__}:{len(value)} items]"
    if isinstance(value, dict):
        return f"[dict:{len(value)} keys]"
    return f"[{type(value).__name__}]"


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

    # Emergency shutdown events (CRITICAL for prod observability)
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    EMERGENCY_SHUTDOWN_MEMORY = "emergency_shutdown_memory"
    EMERGENCY_SHUTDOWN_TIMEOUT = "emergency_shutdown_timeout"
    EMERGENCY_SHUTDOWN_RESET = "emergency_shutdown_reset"

    # Performance events
    PERFORMANCE_DEGRADATION = "performance_degradation"
    PROCESSING_TIME_EXCEEDED = "processing_time_exceeded"

    # General events
    EVENT_PROCESSED = "event_processed"
    STATE_CHANGE = "state_change"

    # Request lifecycle events (Phase 7 additions)
    REQUEST_STARTED = "request_started"
    REQUEST_COMPLETED = "request_completed"
    REQUEST_REJECTED = "request_rejected"

    # Generation events
    GENERATION_STARTED = "generation_started"
    GENERATION_COMPLETED = "generation_completed"

    # Aphasia events (for unified tracking)
    APHASIA_DETECTED = "aphasia_detected"
    APHASIA_REPAIRED = "aphasia_repaired"


class RejectionReason(Enum):
    """Reason codes for request rejections."""

    NORMAL = "normal"
    MORAL_REJECT = "moral_reject"
    MORAL_PRECHECK = "moral_precheck"
    GRAMMAR_PRECHECK = "grammar_precheck"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    RATE_LIMIT = "rate_limit"
    SLEEP_PHASE = "sleep_phase"
    VALIDATION_ERROR = "validation_error"


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

        # Use the record's timestamp for consistency
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "timestamp_unix": record.created,
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
        # Create a robust daily log filename
        daily_log_path = log_path.parent / f"{log_path.stem}_daily{log_path.suffix}"
        time_handler = TimedRotatingFileHandler(
            filename=str(daily_log_path),
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
        # Calculate utilization, protecting against division by zero
        utilization_percent = (current_size / capacity) * 100 if capacity > 0 else 0.0

        return self.warn(
            EventType.MEMORY_FULL,
            f"Memory capacity reached: {current_size}/{capacity} items ({memory_mb:.2f} MB)",
            correlation_id=correlation_id,
            metrics={
                "current_size": current_size,
                "capacity": capacity,
                "memory_mb": memory_mb,
                "utilization_percent": utilization_percent,
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

    def log_emergency_shutdown(
        self,
        reason: str,
        memory_mb: float | None = None,
        processing_time_ms: float | None = None,
        correlation_id: str | None = None,
    ) -> str:
        """Log an emergency shutdown event (CRITICAL for prod observability).

        This method logs emergency shutdown events with relevant context.
        INVARIANT: Only metadata is logged, never PII or raw user content.

        Args:
            reason: Reason for shutdown (e.g., 'memory_exceeded', 'processing_timeout')
            memory_mb: Current memory usage in MB (if applicable)
            processing_time_ms: Processing time in ms (if applicable)
            correlation_id: Optional correlation ID

        Returns:
            Correlation ID
        """
        event_type = EventType.EMERGENCY_SHUTDOWN
        if reason == "memory_exceeded":
            event_type = EventType.EMERGENCY_SHUTDOWN_MEMORY
        elif reason == "processing_timeout":
            event_type = EventType.EMERGENCY_SHUTDOWN_TIMEOUT

        metrics: dict[str, Any] = {"reason": reason}
        if memory_mb is not None:
            metrics["memory_mb"] = memory_mb
        if processing_time_ms is not None:
            metrics["processing_time_ms"] = processing_time_ms

        return self.error(
            event_type,
            f"EMERGENCY SHUTDOWN triggered: {reason}",
            correlation_id=correlation_id,
            metrics=metrics,
        )

    def log_emergency_shutdown_reset(
        self,
        correlation_id: str | None = None,
    ) -> str:
        """Log an emergency shutdown reset event.

        Args:
            correlation_id: Optional correlation ID

        Returns:
            Correlation ID
        """
        return self.warn(
            EventType.EMERGENCY_SHUTDOWN_RESET,
            "Emergency shutdown flag has been reset",
            correlation_id=correlation_id,
        )

    # ------------------------------------------------------------------ #
    # Request lifecycle logging with mandatory fields (Phase 7)          #
    # ------------------------------------------------------------------ #

    def log_request_started(
        self,
        request_id: str,
        phase: str,
        step_counter: int,
        endpoint: str | None = None,
        prompt_length: int | None = None,
        correlation_id: str | None = None,
    ) -> str:
        """Log the start of a request with mandatory observability fields.

        Args:
            request_id: Unique identifier for this request
            phase: Current cognitive phase (wake/sleep)
            step_counter: Current step counter
            endpoint: Optional API endpoint (e.g., '/generate', '/infer')
            prompt_length: Length of prompt (NOT the prompt itself)
            correlation_id: Optional correlation ID

        Returns:
            Correlation ID
        """
        metrics: dict[str, Any] = {
            "request_id": request_id,
            "phase": phase,
            "step_counter": step_counter,
        }
        if endpoint:
            metrics["endpoint"] = endpoint
        if prompt_length is not None:
            metrics["prompt_length"] = prompt_length

        return self.info(
            EventType.REQUEST_STARTED,
            f"Request started: {request_id}",
            correlation_id=correlation_id or request_id,
            metrics=metrics,
        )

    def log_request_completed(
        self,
        request_id: str,
        phase: str,
        step_counter: int,
        moral_score_before: float | None = None,
        moral_score_after: float | None = None,
        accepted: bool = True,
        reason: str = "normal",
        latency_ms: float | None = None,
        response_length: int | None = None,
        correlation_id: str | None = None,
    ) -> str:
        """Log the completion of a request with mandatory observability fields.

        This is the main logging method for request lifecycle tracking.
        All mandatory fields are captured here for full observability.

        INVARIANT: No PII, raw user input, or LLM output is logged.

        Args:
            request_id: Unique identifier for this request
            phase: Current cognitive phase (wake/sleep)
            step_counter: Current step counter
            moral_score_before: Moral score before processing
            moral_score_after: Moral score after processing
            accepted: Whether the request was accepted
            reason: Reason code (normal/moral_reject/emergency_shutdown/etc.)
            latency_ms: Total processing latency in milliseconds
            response_length: Length of response (NOT the response itself)
            correlation_id: Optional correlation ID

        Returns:
            Correlation ID
        """
        metrics: dict[str, Any] = {
            "request_id": request_id,
            "phase": phase,
            "step_counter": step_counter,
            "accepted": accepted,
            "reason": reason,
        }
        if moral_score_before is not None:
            metrics["moral_score_before"] = round(moral_score_before, 4)
        if moral_score_after is not None:
            metrics["moral_score_after"] = round(moral_score_after, 4)
        if latency_ms is not None:
            metrics["latency_ms"] = round(latency_ms, 2)
        if response_length is not None:
            metrics["response_length"] = response_length

        status = "completed" if accepted else f"rejected ({reason})"
        return self.info(
            EventType.REQUEST_COMPLETED,
            f"Request {status}: {request_id}",
            correlation_id=correlation_id or request_id,
            metrics=metrics,
        )

    def log_request_rejected(
        self,
        request_id: str,
        phase: str,
        step_counter: int,
        reason: str,
        moral_score: float | None = None,
        threshold: float | None = None,
        correlation_id: str | None = None,
    ) -> str:
        """Log a rejected request with context.

        Args:
            request_id: Unique identifier for this request
            phase: Current cognitive phase (wake/sleep)
            step_counter: Current step counter
            reason: Reason for rejection
            moral_score: Moral score that caused rejection (if applicable)
            threshold: Moral threshold (if applicable)
            correlation_id: Optional correlation ID

        Returns:
            Correlation ID
        """
        metrics: dict[str, Any] = {
            "request_id": request_id,
            "phase": phase,
            "step_counter": step_counter,
            "accepted": False,
            "reason": reason,
        }
        if moral_score is not None:
            metrics["moral_score"] = round(moral_score, 4)
        if threshold is not None:
            metrics["threshold"] = round(threshold, 4)

        return self.warn(
            EventType.REQUEST_REJECTED,
            f"Request rejected ({reason}): {request_id}",
            correlation_id=correlation_id or request_id,
            metrics=metrics,
        )

    def log_generation_event(
        self,
        request_id: str,
        phase: str,
        step_counter: int,
        event: str,  # "started" or "completed"
        latency_ms: float | None = None,
        response_length: int | None = None,
        correlation_id: str | None = None,
    ) -> str:
        """Log a generation lifecycle event.

        Args:
            request_id: Unique identifier for this request
            phase: Current cognitive phase (wake/sleep)
            step_counter: Current step counter
            event: Event type ("started" or "completed")
            latency_ms: Generation latency in milliseconds (for completed)
            response_length: Length of response (NOT the response itself)
            correlation_id: Optional correlation ID

        Returns:
            Correlation ID
        """
        event_type = (
            EventType.GENERATION_STARTED if event == "started"
            else EventType.GENERATION_COMPLETED
        )
        metrics: dict[str, Any] = {
            "request_id": request_id,
            "phase": phase,
            "step_counter": step_counter,
        }
        if latency_ms is not None:
            metrics["latency_ms"] = round(latency_ms, 2)
        if response_length is not None:
            metrics["response_length"] = response_length

        return self.info(
            event_type,
            f"Generation {event}: {request_id}",
            correlation_id=correlation_id or request_id,
            metrics=metrics,
        )

    def log_aphasia_event(
        self,
        request_id: str,
        detected: bool,
        severity: float,
        repaired: bool,
        flags: list[str] | None = None,
        severity_bucket: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        """Log an aphasia detection/repair event.

        This provides unified aphasia logging alongside the aphasia_logging module.

        Args:
            request_id: Unique identifier for this request
            detected: Whether aphasia was detected
            severity: Aphasia severity score (0.0 to 1.0)
            repaired: Whether repair was applied
            flags: List of aphasia flags detected
            severity_bucket: Severity bucket label (e.g., "low", "medium", "high")
            correlation_id: Optional correlation ID

        Returns:
            Correlation ID
        """
        event_type = EventType.APHASIA_REPAIRED if repaired else EventType.APHASIA_DETECTED
        metrics: dict[str, Any] = {
            "request_id": request_id,
            "detected": detected,
            "severity": round(severity, 3),
            "repaired": repaired,
        }
        if flags:
            metrics["flags"] = flags
        if severity_bucket:
            metrics["severity_bucket"] = severity_bucket

        action = "repaired" if repaired else ("detected" if detected else "checked")
        return self.info(
            event_type,
            f"Aphasia {action}: severity={severity:.3f}",
            correlation_id=correlation_id or request_id,
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
_observability_logger_lock = Lock()


def get_observability_logger(
    logger_name: str = "mlsdm_observability",
    **kwargs: Any,
) -> ObservabilityLogger:
    """Get or create the observability logger instance.

    This function is thread-safe and implements the singleton pattern
    with double-checked locking for optimal performance.

    Args:
        logger_name: Name for the logger instance
        **kwargs: Additional arguments passed to ObservabilityLogger constructor

    Returns:
        ObservabilityLogger instance
    """
    global _observability_logger

    # Double-checked locking pattern for thread-safe singleton
    if _observability_logger is None:
        with _observability_logger_lock:
            # Check again inside the lock
            if _observability_logger is None:
                _observability_logger = ObservabilityLogger(
                    logger_name=logger_name, **kwargs
                )

    return _observability_logger
