"""
Payload scrubber for removing secrets from logs.

This module provides utilities for scrubbing sensitive information from
log messages and payloads before they are logged or exported.
"""

import re
from typing import Any

# Patterns for common secrets
SECRET_PATTERNS = [
    # API keys (common patterns)
    (re.compile(r'(api[_-]?key["\']?\s*[:=]\s*["\']?)([a-zA-Z0-9_\-]{20,})(["\']?)', re.IGNORECASE), r'\1***REDACTED***\3'),
    (re.compile(r'(sk-[a-zA-Z0-9]{20,})'), r'sk-***REDACTED***'),
    (re.compile(r'(Bearer\s+)([a-zA-Z0-9_\-\.]{20,})', re.IGNORECASE), r'\1***REDACTED***'),

    # Passwords
    (re.compile(r'(password["\']?\s*[:=]\s*["\']?)([^\s"\']{8,})(["\']?)', re.IGNORECASE), r'\1***REDACTED***\3'),
    (re.compile(r'(passwd["\']?\s*[:=]\s*["\']?)([^\s"\']{8,})(["\']?)', re.IGNORECASE), r'\1***REDACTED***\3'),
    (re.compile(r'(pwd["\']?\s*[:=]\s*["\']?)([^\s"\']{8,})(["\']?)', re.IGNORECASE), r'\1***REDACTED***\3'),

    # Tokens
    (re.compile(r'(token["\']?\s*[:=]\s*["\']?)([a-zA-Z0-9_\-\.]{20,})(["\']?)', re.IGNORECASE), r'\1***REDACTED***\3'),
    (re.compile(r'(access[_-]?token["\']?\s*[:=]\s*["\']?)([a-zA-Z0-9_\-\.]{20,})(["\']?)', re.IGNORECASE), r'\1***REDACTED***\3'),
    (re.compile(r'(refresh[_-]?token["\']?\s*[:=]\s*["\']?)([a-zA-Z0-9_\-\.]{20,})(["\']?)', re.IGNORECASE), r'\1***REDACTED***\3'),

    # AWS keys
    (re.compile(r'(AKIA[0-9A-Z]{16})'), r'AKIA***REDACTED***'),
    (re.compile(r'(aws[_-]?secret[_-]?access[_-]?key["\']?\s*[:=]\s*["\']?)([a-zA-Z0-9/+=]{40})(["\']?)', re.IGNORECASE), r'\1***REDACTED***\3'),

    # Private keys
    (re.compile(r'(-----BEGIN.*PRIVATE KEY-----).*?(-----END.*PRIVATE KEY-----)', re.DOTALL), r'\1\n***REDACTED***\n\2'),

    # Credit card numbers (simple pattern)
    (re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'), r'****-****-****-****'),

    # Email addresses (optional, depending on privacy requirements)
    # (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), r'***@***.***'),
]


def scrub_text(text: str) -> str:
    """Scrub sensitive information from text.

    Args:
        text: Input text that may contain secrets.

    Returns:
        Text with secrets replaced by placeholders.

    Example:
        >>> scrub_text("api_key=sk-123456789abcdef")
        'api_key=sk-***REDACTED***'
    """
    if not text:
        return text

    scrubbed = text
    for pattern, replacement in SECRET_PATTERNS:
        scrubbed = pattern.sub(replacement, scrubbed)

    return scrubbed


def scrub_dict(data: dict[str, Any], keys_to_scrub: set[str] | None = None) -> dict[str, Any]:
    """Scrub sensitive information from a dictionary.

    This function recursively scrubs both values matching secret patterns
    and specific keys that are known to contain secrets.

    Args:
        data: Dictionary to scrub.
        keys_to_scrub: Set of keys that should always be scrubbed.
            Defaults to common secret key names.

    Returns:
        Dictionary with scrubbed values (creates a new dict, doesn't modify original).

    Example:
        >>> scrub_dict({"api_key": "secret123", "username": "john"})
        {'api_key': '***REDACTED***', 'username': 'john'}
    """
    if keys_to_scrub is None:
        keys_to_scrub = {
            'api_key', 'apikey', 'api-key',
            'password', 'passwd', 'pwd',
            'token', 'access_token', 'refresh_token',
            'secret', 'secret_key', 'private_key',
            'openai_api_key', 'openai_key',
            'authorization', 'auth',
        }

    def _scrub_value(key: str, value: Any) -> Any:
        # Check if key should always be scrubbed
        if key.lower() in keys_to_scrub:
            return '***REDACTED***'

        # Recursively scrub nested structures
        if isinstance(value, dict):
            return {k: _scrub_value(k, v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_scrub_value(key, item) for item in value]
        elif isinstance(value, str):
            # Scrub text for patterns
            return scrub_text(value)
        else:
            return value

    return {k: _scrub_value(k, v) for k, v in data.items()}


def should_log_payload() -> bool:
    """Check if payloads should be logged based on environment variable.

    Returns:
        True if LOG_PAYLOADS=true, False otherwise (default: False).
    """
    import os
    return os.environ.get("LOG_PAYLOADS", "false").lower() == "true"
