"""
MLSDM Security: Security utilities for the NeuroCognitiveEngine.

This module provides security features including rate limiting,
payload scrubbing, and logging controls.
"""

from mlsdm.security.payload_scrubber import (
    DEFAULT_SECRET_KEYS,
    EMAIL_PATTERN,
    FORBIDDEN_FIELDS,
    PII_FIELDS,
    SECRET_PATTERNS,
    is_secure_mode,
    scrub_dict,
    scrub_log_record,
    scrub_request_payload,
    scrub_text,
    should_log_payload,
)
from mlsdm.security.rate_limit import RateLimiter, get_rate_limiter

__all__ = [
    "RateLimiter",
    "get_rate_limiter",
    "scrub_text",
    "scrub_dict",
    "scrub_request_payload",
    "scrub_log_record",
    "should_log_payload",
    "is_secure_mode",
    "SECRET_PATTERNS",
    "PII_FIELDS",
    "FORBIDDEN_FIELDS",
    "EMAIL_PATTERN",
    "DEFAULT_SECRET_KEYS",
]
