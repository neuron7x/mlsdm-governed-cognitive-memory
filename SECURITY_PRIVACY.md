# Security and Privacy: Payload Scrubber and Secure Mode

**Document Version:** 1.0.0  
**Last Updated:** November 2025  
**Status:** Production

This document describes the payload scrubber and secure mode features for protecting sensitive data in logs and telemetry.

---

## Overview

The MLSDM system provides two complementary security features:

1. **Payload Scrubber** - Removes sensitive data from payloads before logging
2. **Secure Mode** - A runtime mode that enforces stricter security controls

These features provide **best-effort** protection against data leakage. They are not cryptographic guarantees.

---

## Payload Scrubber

### Location

`src/mlsdm/security/payload_scrubber.py`

### What Gets Scrubbed

The scrubber removes or masks the following categories of sensitive data:

#### 1. Secret Patterns (regex-based)

| Pattern Type | Example | Result |
|--------------|---------|--------|
| API keys | `api_key="sk-abc123..."` | `api_key="***REDACTED***"` |
| Bearer tokens | `Bearer eyJhbG...` | `Bearer ***REDACTED***` |
| AWS keys | `AKIAIOSFODNN7EXAMPLE` | `AKIA***REDACTED***` |
| Passwords | `password="secret123"` | `password="***REDACTED***"` |
| Private keys | `-----BEGIN PRIVATE KEY-----` | `***REDACTED***` |
| Credit cards | `4111-1111-1111-1111` | `****-****-****-****` |

#### 2. PII Fields (key-based, case-insensitive)

| Field Names |
|-------------|
| `email`, `e-mail`, `email_address` |
| `ssn`, `social_security`, `social_security_number` |
| `phone`, `phone_number`, `telephone` |
| `address`, `home_address`, `street_address` |
| `date_of_birth`, `dob`, `birth_date` |
| `credit_card`, `card_number`, `cc_number` |

#### 3. Secret Keys (key-based, case-insensitive)

| Field Names |
|-------------|
| `api_key`, `apikey`, `api-key` |
| `password`, `passwd`, `pwd` |
| `token`, `access_token`, `refresh_token` |
| `secret`, `secret_key`, `private_key` |
| `openai_api_key`, `openai_key` |
| `authorization`, `auth` |

#### 4. Forbidden Fields (for secure mode)

| Category | Field Names |
|----------|-------------|
| User IDs | `user_id`, `userid`, `username`, `account_id`, `session_id` |
| Network IDs | `ip`, `ip_address`, `client_ip`, `remote_addr` |
| Raw Content | `raw_input`, `raw_text`, `prompt`, `full_prompt`, `full_response` |
| Metadata | `metadata`, `user_metadata`, `context` |

### API Functions

#### `scrub_text(text, scrub_emails=False)`

Scrubs secret patterns from plain text.

```python
from mlsdm.security import scrub_text

text = "api_key=sk-secret123456789012345678"
result = scrub_text(text)
# Result: "api_key=sk-***REDACTED***"
```

#### `scrub_dict(data, keys_to_scrub=None, scrub_emails=False, scrub_pii=False)`

Scrubs sensitive data from a dictionary. Works recursively on nested structures.

```python
from mlsdm.security import scrub_dict

data = {"api_key": "secret", "model": "gpt-4"}
result = scrub_dict(data)
# Result: {"api_key": "***REDACTED***", "model": "gpt-4"}
```

#### `scrub_request_payload(payload)`

Scrubs all sensitive fields from an API request payload. This is the primary function for secure mode.

```python
from mlsdm.security import scrub_request_payload

payload = {
    "prompt": "Hello world",
    "user_id": "user123",
    "api_key": "sk-secret"
}
result = scrub_request_payload(payload)
# Result: {"prompt": "***REDACTED***", "user_id": "***REDACTED***", "api_key": "***REDACTED***"}
```

#### `scrub_log_record(record)`

Scrubs sensitive data from a log record before writing to logs.

```python
from mlsdm.security import scrub_log_record

record = {
    "message": "Processing request",
    "user_id": "user123",
    "raw_input": "sensitive data"
}
result = scrub_log_record(record)
# Result: {"message": "Processing request", "user_id": "***REDACTED***", "raw_input": "***REDACTED***"}
```

### Key Features

- **Case-insensitive matching** - `User_ID`, `USER_ID`, and `user_id` are all scrubbed
- **Recursive scrubbing** - Nested dicts and lists are processed
- **Never raises exceptions** - Returns partially scrubbed data on error
- **Original data unchanged** - Creates new dict, doesn't modify input

---

## Secure Mode

### Enabling Secure Mode

Set the environment variable:

```bash
export MLSDM_SECURE_MODE=1
# or
export MLSDM_SECURE_MODE=true
```

### What Secure Mode Does

When `MLSDM_SECURE_MODE=1`:

1. **Disables NeuroLang training** - `neurolang_mode` is forced to `"disabled"`
2. **Ignores checkpoint paths** - No checkpoint loading occurs
3. **Disables aphasia repair** - Detection still works, but repair is disabled
4. **Enforces payload scrubbing** - Use `scrub_request_payload()` and `scrub_log_record()`

### Checking Secure Mode Status

```python
from mlsdm.security import is_secure_mode
from mlsdm.extensions.neuro_lang_extension import is_secure_mode_enabled

# From security module
if is_secure_mode():
    payload = scrub_request_payload(payload)

# From extension module
if is_secure_mode_enabled():
    # Secure mode is active
    pass
```

### Behavior in Secure Mode

| Feature | Normal Mode | Secure Mode |
|---------|-------------|-------------|
| NeuroLang Training | Enabled (configurable) | Disabled |
| Checkpoint Loading | Enabled (configurable) | Disabled |
| Aphasia Detection | Enabled (configurable) | Enabled |
| Aphasia Repair | Enabled (configurable) | Disabled |
| `generate()` | Full features | Works, no training |

---

## Limitations

### What Is NOT Covered

1. **In-memory data** - Scrubbing only affects logged/exported data
2. **Binary data** - Only text and dict structures are scrubbed
3. **Custom field names** - Only predefined field names are matched
4. **Encrypted data** - This is not encryption, just masking
5. **Side-channel attacks** - Timing, size, etc. are not protected

### False Positives

Some non-sensitive data may be scrubbed if:
- Field names match PII/secret patterns
- Text contains patterns that look like secrets (e.g., long alphanumeric strings)

### False Negatives

Some sensitive data may not be scrubbed if:
- Custom/unusual field names are used
- Sensitive data is encoded or obfuscated
- Data is in unexpected format

---

## Testing

### Run Security Tests

```bash
# All security tests
pytest tests/security -q

# Payload scrubber tests
pytest tests/security/test_payload_scrubber.py -v

# Secure mode tests
pytest tests/security/test_secure_mode.py -v

# Aphasia privacy tests
pytest tests/security/test_aphasia_logging_privacy.py -v
```

### Test Coverage

| Test Suite | Tests | Coverage |
|------------|-------|----------|
| `test_payload_scrubber.py` | 73 | PII, secrets, nested data, edge cases |
| `test_secure_mode.py` | 19 | Mode detection, generation, scrubbing |
| `test_aphasia_logging_privacy.py` | 6 | No prompt/response in aphasia logs |

---

## Integration Example

```python
from mlsdm.security import (
    is_secure_mode,
    scrub_request_payload,
    scrub_log_record,
)
import logging

logger = logging.getLogger(__name__)

def handle_request(request: dict) -> dict:
    # Scrub request payload before processing in secure mode
    if is_secure_mode():
        safe_request = scrub_request_payload(request)
        logger.info("Request received", extra=scrub_log_record({
            "request_id": request.get("request_id"),
            "timestamp": request.get("timestamp"),
            # Sensitive fields are automatically scrubbed
        }))
    else:
        safe_request = request
        logger.info("Request received", extra=request)

    # Process request...
    return response
```

---

## Related Documentation

- [SECURITY_SUMMARY.md](SECURITY_SUMMARY.md) - Overall security implementation
- [SECURITY_README.md](SECURITY_README.md) - Security quick start guide
- [SECURITY_POLICY.md](SECURITY_POLICY.md) - Security policy and disclosure
- [CLAIMS_TRACEABILITY.md](CLAIMS_TRACEABILITY.md) - Payload scrubbing claims validation

---

## Changelog

### v1.0.0 (November 2025)

- Initial release with payload scrubber and secure mode
- 73 tests for payload scrubber
- 19 tests for secure mode
- Case-insensitive key matching
- Recursive nested structure handling
- Exception-safe scrubbing functions
