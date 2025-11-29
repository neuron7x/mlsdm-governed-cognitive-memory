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
]

# PII field names that should be scrubbed
PII_FIELDS = frozenset({
    'email', 'e-mail', 'email_address',
    'ssn', 'social_security', 'social_security_number',
    'phone', 'phone_number', 'telephone',
    'address', 'home_address', 'street_address',
    'date_of_birth', 'dob', 'birth_date',
    'credit_card', 'card_number', 'cc_number',
})

# Default keys to scrub (secrets)
DEFAULT_SECRET_KEYS = frozenset({
    'api_key', 'apikey', 'api-key',
    'password', 'passwd', 'pwd',
    'token', 'access_token', 'refresh_token',
    'secret', 'secret_key', 'private_key',
    'openai_api_key', 'openai_key',
    'authorization', 'auth',
})

# Pre-computed combined set for scrub_pii=True
_SECRET_AND_PII_KEYS = DEFAULT_SECRET_KEYS | PII_FIELDS

# Email pattern for scrubbing PII in text
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')


def scrub_text(text: str, scrub_emails: bool = False) -> str:
    """Scrub sensitive information from text.

    Args:
        text: Input text that may contain secrets.
        scrub_emails: Whether to scrub email addresses (default: False for backward compat).

    Returns:
        Text with secrets replaced by placeholders.

    Example:
        >>> scrub_text("api_key=sk-123456789abcdef")
        'api_key=sk-***REDACTED***'
        >>> scrub_text("contact: user@example.com", scrub_emails=True)
        'contact: ***@***.***'
    """
    if not text:
        return text

    scrubbed = text
    for pattern, replacement in SECRET_PATTERNS:
        scrubbed = pattern.sub(replacement, scrubbed)

    # Optionally scrub email addresses
    if scrub_emails:
        scrubbed = EMAIL_PATTERN.sub(r'***@***.***', scrubbed)

    return scrubbed


def scrub_dict(
    data: dict[str, Any],
    keys_to_scrub: frozenset[str] | set[str] | None = None,
    scrub_emails: bool = False,
    scrub_pii: bool = False,
) -> dict[str, Any]:
    """Scrub sensitive information from a dictionary.

    This function recursively scrubs both values matching secret patterns
    and specific keys that are known to contain secrets.

    Args:
        data: Dictionary to scrub.
        keys_to_scrub: Set of keys that should always be scrubbed.
            Defaults to common secret key names.
        scrub_emails: Whether to scrub email addresses in text values.
        scrub_pii: Whether to scrub PII fields (email, ssn, phone, etc.).

    Returns:
        Dictionary with scrubbed values (creates a new dict, doesn't modify original).

    Example:
        >>> scrub_dict({"api_key": "secret123", "username": "john"})
        {'api_key': '***REDACTED***', 'username': 'john'}
        >>> scrub_dict({"email": "user@example.com"}, scrub_pii=True)
        {'email': '***REDACTED***'}
    """
    # Determine effective keys to scrub
    if keys_to_scrub is None:
        # Use pre-computed sets for efficiency
        effective_keys: frozenset[str] | set[str] = (
            _SECRET_AND_PII_KEYS if scrub_pii else DEFAULT_SECRET_KEYS
        )
    else:
        # Custom keys provided - need to compute union if scrub_pii
        effective_keys = keys_to_scrub | PII_FIELDS if scrub_pii else keys_to_scrub

    def _scrub_value(key: str, value: Any) -> Any:
        # Check if key should always be scrubbed
        if key.lower() in effective_keys:
            return '***REDACTED***'

        # Recursively scrub nested structures
        if isinstance(value, dict):
            return {k: _scrub_value(k, v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_scrub_value(key, item) for item in value]
        elif isinstance(value, str):
            # Scrub text for patterns
            return scrub_text(value, scrub_emails=scrub_emails)
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
