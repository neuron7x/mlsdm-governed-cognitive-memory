# Security Features Quick Start

## Overview

The MLSDM Governed Cognitive Memory system includes comprehensive security features:

✅ **Rate Limiting** - 5 RPS per client
✅ **Input Validation** - Comprehensive validation and sanitization
✅ **Security Logging** - Structured audit logs with correlation IDs
✅ **Dependency Scanning** - Automated vulnerability detection (pip-audit in CI)
✅ **SAST Scanning** - Bandit and Semgrep in CI/CD pipeline
✅ **Pre-commit Hooks** - Security checks before commit (bandit, detect-private-key)
✅ **41 Security Tests** - All passing

## Quick Start

### Run Security Tests

```bash
# All security tests
python -m pytest src/tests/unit/test_security.py -v --no-cov

# Integration test suite
python scripts/test_security_features.py
```

### Run Dependency Security Audit

```bash
# Scan for known vulnerabilities in dependencies
pip install pip-audit
pip-audit --strict

# Alternative: Use safety (if installed)
pip install safety
safety check
```

### Run SAST Scan Locally

```bash
# Install bandit
pip install bandit

# Run bandit scan
bandit -r src/mlsdm --severity-level medium --confidence-level medium

# Check for high severity only
bandit -r src/mlsdm --severity-level high --confidence-level high
```

### Configuration

```bash
# Set API key for authentication
export API_KEY="your-secure-key-here"

# Disable rate limiting for testing
export DISABLE_RATE_LIMIT=1
```

## Dependency Security

### Automated Scanning in CI

The following security scans run automatically:

| Scan Type | Workflow | Trigger |
|-----------|----------|---------|
| pip-audit | `ci-neuro-cognitive-engine.yml` | Every PR and push to main |
| Bandit SAST | `sast-scan.yml` | Every PR to main |
| Semgrep | `sast-scan.yml` | Every PR to main |
| Trivy (container) | `release.yml` | On release tags |

### Security Pinning

Critical indirect dependencies are pinned in `requirements.txt` to avoid known vulnerabilities:

- `certifi>=2024.7.4` - SSL certificate handling
- `cryptography>=43.0.1` - Cryptographic operations
- `jinja2>=3.1.6` - Template engine (used by sentence-transformers)
- `urllib3>=2.2.2` - HTTP client (used by requests)
- `setuptools>=78.1.1` - Build system security
- `idna>=3.7` - Domain name handling

### Pre-commit Hooks

Install pre-commit hooks to catch security issues before commit:

```bash
pip install pre-commit
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

Security-related hooks:
- `detect-private-key` - Prevents committing private keys
- `bandit` - Python security linter
- `check-merge-conflict` - Prevents incomplete merges

## Using Security Features

### Rate Limiting

```python
from mlsdm.utils.rate_limiter import RateLimiter

limiter = RateLimiter(rate=5.0, capacity=10)
if limiter.is_allowed(client_id):
    # Process request
    pass
else:
    # Return 429 Too Many Requests
    pass
```

### Input Validation

```python
from mlsdm.utils.input_validator import InputValidator

validator = InputValidator()

# Validate vector
vector = validator.validate_vector([1.0, 2.0, 3.0], expected_dim=3)

# Validate moral value
moral = validator.validate_moral_value(0.75)

# Sanitize string
safe_text = validator.sanitize_string(user_input, max_length=1000)
```

### Security Logging

```python
from mlsdm.utils.security_logger import get_security_logger

logger = get_security_logger()

# Log authentication
logger.log_auth_success(client_id="abc123")
logger.log_auth_failure(client_id="abc123", reason="Invalid token")

# Log rate limiting
logger.log_rate_limit_exceeded(client_id="abc123")

# Log validation errors
logger.log_invalid_input(client_id="abc123", error_message="Invalid input")
```

## API Endpoints

All endpoints include rate limiting and input validation:

```bash
# Health check (no auth required)
curl http://localhost:8000/health

# Get state (requires auth)
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8000/v1/state/

# Process event (requires auth, rate limited, validated)
curl -X POST -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"event_vector": [1.0, 2.0, 3.0], "moral_value": 0.8}' \
     http://localhost:8000/v1/process_event/
```

## Documentation

- **Full Implementation Guide:** [SECURITY_IMPLEMENTATION.md](SECURITY_IMPLEMENTATION.md)
- **Validation Report:** [SECURITY_SUMMARY.md](SECURITY_SUMMARY.md)
- **Security Policy:** [SECURITY_POLICY.md](SECURITY_POLICY.md)
- **Threat Model:** [THREAT_MODEL.md](THREAT_MODEL.md)

## Testing

```bash
# Security tests (41 tests)
pytest src/tests/unit/test_security.py -v

# API tests (15 tests)
pytest src/tests/unit/test_api.py -v

# All tests
pytest src/tests/unit/ -v
```

## Status

**Security Implementation:** ✅ Complete
**Tests:** ✅ 56/56 Passing
**CodeQL:** ✅ 0 Vulnerabilities
**Dependency Audit:** ✅ No known vulnerabilities (pip-audit)
**Production Ready:** ✅ Yes

## GitHub Actions Versions

All GitHub Actions are pinned to stable versions:

- `actions/checkout@v4`
- `actions/setup-python@v5`
- `actions/upload-artifact@v4`
- `github/codeql-action/upload-sarif@v3`
- `docker/setup-buildx-action@v3`
- `docker/login-action@v3`
- `docker/build-push-action@v5`
- `aquasecurity/trivy-action@0.31.0`
- `softprops/action-gh-release@v2`
- `semgrep/semgrep-action@v1`

## Support

For security issues, see [SECURITY_POLICY.md](SECURITY_POLICY.md) for responsible disclosure procedures.
