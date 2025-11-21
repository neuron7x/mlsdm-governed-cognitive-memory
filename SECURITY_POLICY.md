# Security Policy

**Document Version:** 1.0.0  
**Project Version:** 1.0.0  
**Last Updated:** November 2025  
**Security Contact:** Report vulnerabilities via GitHub Security Advisories

## Table of Contents

- [Security Overview](#security-overview)
- [Supported Versions](#supported-versions)
- [Reporting a Vulnerability](#reporting-a-vulnerability)
- [Security Architecture](#security-architecture)
- [Security Controls](#security-controls)
- [Data Protection](#data-protection)
- [Authentication and Authorization](#authentication-and-authorization)
- [Input Validation](#input-validation)
- [Rate Limiting and DDoS Protection](#rate-limiting-and-ddos-protection)
- [Logging and Monitoring](#logging-and-monitoring)
- [Dependency Management](#dependency-management)
- [Secure Deployment Guidelines](#secure-deployment-guidelines)
- [Security Testing](#security-testing)
- [Compliance and Standards](#compliance-and-standards)

---

## Security Overview

MLSDM Governed Cognitive Memory implements defense-in-depth security principles to protect against common vulnerabilities and ensure safe operation in production environments. This policy outlines security measures, best practices, and incident response procedures.

### Security Objectives

1. **Confidentiality**: Protect sensitive data from unauthorized access
2. **Integrity**: Ensure data accuracy and prevent unauthorized modifications
3. **Availability**: Maintain reliable service operation under attack
4. **Accountability**: Provide audit trails for security-relevant events
5. **Resilience**: Gracefully degrade under adverse conditions

---

## Supported Versions

Security updates are provided for the following versions:

| Version | Supported | Security Updates | End of Life |
|---------|-----------|------------------|-------------|
| 1.0.x   | ✅ Yes    | Active           | TBD         |
| 0.x.x   | ❌ No     | None             | Nov 2025    |

**Upgrade Policy**: Users should upgrade to the latest 1.0.x release within 30 days of release to receive security fixes.

---

## Reporting a Vulnerability

### Disclosure Process

We follow coordinated vulnerability disclosure practices:

1. **Report Privately**: Submit vulnerabilities via [GitHub Security Advisories](https://github.com/neuron7x/mlsdm/security/advisories/new)
2. **Initial Response**: Within 48 hours
3. **Triage and Validation**: Within 7 days
4. **Fix Development**: Based on severity (see timeline below)
5. **Public Disclosure**: After patch release + 7 days

### Severity Classification

| Severity | Response Time | Fix Timeline | Examples |
|----------|---------------|--------------|----------|
| **Critical** | 24 hours | 7 days | RCE, authentication bypass, data exfiltration |
| **High** | 48 hours | 14 days | SQL injection, XSS, privilege escalation |
| **Medium** | 7 days | 30 days | DoS, information disclosure |
| **Low** | 14 days | 90 days | Minor information leaks, low-impact issues |

### What to Include

When reporting a security issue, please provide:
- Detailed description of the vulnerability
- Steps to reproduce
- Proof of concept (if available)
- Potential impact assessment
- Suggested remediation (if known)
- Your contact information for follow-up

### Hall of Fame

We recognize security researchers who responsibly disclose vulnerabilities. With permission, we will acknowledge contributors in our security advisories.

---

## Security Architecture

### Threat Model

The system is designed to resist the following threat categories:

1. **External Attacks**
   - Network-based attacks (DDoS, man-in-the-middle)
   - Application-level attacks (injection, XSS, CSRF)
   - Authentication bypass attempts
   - Rate limit circumvention

2. **Internal Threats**
   - Malicious input injection via LLM prompts
   - Memory exhaustion attacks
   - Resource starvation
   - Timing attacks

3. **Supply Chain**
   - Compromised dependencies
   - Malicious code injection
   - Vulnerable transitive dependencies

### Security Boundaries

```
┌────────────────────────────────────────────────┐
│            External Network (Untrusted)        │
└───────────────────┬────────────────────────────┘
                    │
    ┌───────────────▼────────────────┐
    │   API Gateway / Load Balancer  │
    │   - TLS Termination            │
    │   - Rate Limiting              │
    │   - DDoS Protection            │
    └───────────────┬────────────────┘
                    │
    ┌───────────────▼────────────────┐
    │   Application Layer            │
    │   - Authentication             │
    │   - Input Validation           │
    │   - Authorization              │
    └───────────────┬────────────────┘
                    │
    ┌───────────────▼────────────────┐
    │   Cognitive Controller         │
    │   - Memory Protection          │
    │   - Resource Limits            │
    │   - Moral Filtering            │
    └───────────────┬────────────────┘
                    │
    ┌───────────────▼────────────────┐
    │   Memory Layer                 │
    │   - Bounded Storage            │
    │   - Data Sanitization          │
    └────────────────────────────────┘
```

---

## Security Controls

### SC-1: Input Validation

**Objective**: Prevent injection attacks and invalid data processing

**Implementation**:
```python
# Location: src/mlsdm/utils/input_validator.py

def validate_event_vector(vector: Any) -> np.ndarray:
    """Validate and sanitize event vectors."""
    # Type checking
    if not isinstance(vector, (list, np.ndarray)):
        raise ValueError("Vector must be list or numpy array")
    
    # Dimension validation
    if len(vector) != EXPECTED_DIM:
        raise ValueError(f"Vector dimension must be {EXPECTED_DIM}")
    
    # Range validation (prevent overflow/underflow)
    if not np.all(np.isfinite(vector)):
        raise ValueError("Vector contains non-finite values")
    
    # Normalization (prevent adversarial inputs)
    vector = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if norm > 1e6 or norm < 1e-6:
        raise ValueError("Vector norm out of acceptable range")
    
    return vector

def validate_moral_value(value: Any) -> float:
    """Validate moral value parameter."""
    if not isinstance(value, (int, float)):
        raise ValueError("Moral value must be numeric")
    
    if not 0.0 <= value <= 1.0:
        raise ValueError("Moral value must be in range [0.0, 1.0]")
    
    return float(value)
```

**Coverage**:
- ✅ Type validation for all API inputs
- ✅ Range checking for numeric parameters
- ✅ Dimension validation for vectors
- ✅ Sanitization of special values (NaN, Inf)

### SC-2: Authentication

**Objective**: Verify identity of API clients

**Implementation**:
```python
# Location: src/mlsdm/api/app.py

def verify_bearer_token(authorization: str) -> bool:
    """Verify Bearer token against environment variable."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "Invalid authentication scheme")
    
    token = authorization[7:]  # Remove "Bearer " prefix
    expected_token = os.getenv("API_KEY")
    
    if not expected_token:
        raise ConfigurationError("API_KEY not configured")
    
    # Constant-time comparison to prevent timing attacks
    return secrets.compare_digest(token, expected_token)
```

**Properties**:
- ✅ Bearer token authentication
- ✅ Constant-time comparison (timing attack resistant)
- ✅ Environment-based configuration
- ✅ No hardcoded credentials

**Security Considerations**:
- Tokens should be rotated regularly (recommended: every 90 days)
- Use strong, randomly generated tokens (≥32 bytes entropy)
- Never commit tokens to version control
- Store tokens in secure secret management systems

### SC-3: Rate Limiting

**Objective**: Prevent abuse and DoS attacks

**Implementation**:
```python
# Location: src/mlsdm/utils/rate_limiter.py

class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: int = 5, period: int = 1):
        """
        Initialize rate limiter.
        
        Args:
            rate: Number of requests allowed per period
            period: Time period in seconds
        """
        self.rate = rate
        self.period = period
        self.tokens = {}  # client_id -> (tokens, last_update)
    
    def allow_request(self, client_id: str) -> bool:
        """Check if request is allowed under rate limit."""
        now = time.time()
        
        if client_id not in self.tokens:
            self.tokens[client_id] = (self.rate - 1, now)
            return True
        
        tokens, last_update = self.tokens[client_id]
        elapsed = now - last_update
        
        # Refill tokens based on elapsed time
        tokens = min(self.rate, tokens + elapsed * (self.rate / self.period))
        
        if tokens >= 1:
            self.tokens[client_id] = (tokens - 1, now)
            return True
        else:
            self.tokens[client_id] = (tokens, now)
            return False
```

**Configuration**:
- Default: 5 requests per second per client
- Configurable via environment: `RATE_LIMIT_RPS`
- Client identification: API key or IP address
- Algorithm: Token bucket (allows bursts)

### SC-4: Memory Protection

**Objective**: Prevent memory exhaustion and overflow attacks

**Implementation**:
- Hard capacity limit: 20,000 vectors
- Circular buffer eviction (FIFO)
- Pre-allocated memory (zero dynamic allocation)
- Input size validation

**Guarantees**:
- ✅ Maximum memory: 29.37 MB (verified)
- ✅ No unbounded growth
- ✅ Deterministic memory usage
- ✅ No memory leaks (verified 24h soak test)

---

## Data Protection

### DP-1: Data Classification

| Data Type | Classification | Storage | Retention | Encryption |
|-----------|---------------|---------|-----------|------------|
| Event Vectors | Confidential | Memory | Session | In-transit |
| Moral Values | Confidential | Memory | Session | In-transit |
| API Tokens | Secret | Environment | Permanent | At-rest |
| Logs | Internal | Disk | 30 days | Optional |
| Metrics | Internal | Memory | Ephemeral | None |

### DP-2: Encryption

**In-Transit Encryption**:
- TLS 1.2+ for all API communications
- Certificate validation enforced
- Strong cipher suites only (AES-256-GCM, ChaCha20-Poly1305)

**At-Rest Encryption**:
- API tokens stored in environment variables or secret managers
- No sensitive data persisted to disk
- Logs sanitized to exclude PII

### DP-3: Data Sanitization

**PII Exclusion**:
```python
# Location: src/mlsdm/utils/security_logger.py

def sanitize_log_data(data: dict) -> dict:
    """Remove sensitive information from log data."""
    sanitized = data.copy()
    
    # Remove vector data (may contain sensitive embeddings)
    if "event_vector" in sanitized:
        sanitized["event_vector"] = f"<vector dim={len(data['event_vector'])}>"
    
    # Remove API tokens
    if "authorization" in sanitized:
        sanitized["authorization"] = "<redacted>"
    
    # Truncate long text fields
    for key in ["prompt", "response"]:
        if key in sanitized and len(sanitized[key]) > 100:
            sanitized[key] = sanitized[key][:100] + "..."
    
    return sanitized
```

---

## Authentication and Authorization

### Authentication Methods

1. **API Key (Bearer Token)**
   - Environment variable: `API_KEY`
   - Header: `Authorization: Bearer <token>`
   - Validation: Constant-time comparison
   - Rotation: Recommended every 90 days

2. **Future Methods** (planned for v1.1+)
   - OAuth 2.0 / OpenID Connect
   - mTLS (mutual TLS)
   - JWT tokens with expiration

### Authorization Model

**Current**: Simple authentication (authenticated = authorized)

**Future** (v1.1+): Role-Based Access Control (RBAC)
- `read`: View state and metrics
- `write`: Submit events for processing
- `admin`: Configuration and system management

---

## Input Validation

### Validation Rules

| Parameter | Type | Range | Validation |
|-----------|------|-------|------------|
| `event_vector` | np.ndarray | dim=384, finite values | Dimension, type, finiteness |
| `moral_value` | float | [0.0, 1.0] | Range, type |
| `prompt` | str | ≤10,000 chars | Length, encoding |
| `max_tokens` | int | [1, 4096] | Range, type |
| `context_top_k` | int | [1, 100] | Range, type |

### Validation Strategy

1. **Schema Validation**: Pydantic models for type checking
2. **Range Validation**: Explicit bounds checking
3. **Format Validation**: Encoding and structure checks
4. **Sanitization**: Remove/escape dangerous characters
5. **Error Handling**: Reject invalid input with clear errors

---

## Rate Limiting and DDoS Protection

### Rate Limiting Configuration

```yaml
# Default rate limits
global:
  rps: 1000  # Global requests per second
  
per_client:
  rps: 5     # Per-client requests per second
  burst: 10  # Maximum burst size

per_endpoint:
  /v1/process_event: 5 rps
  /v1/state: 20 rps
  /health: 100 rps
```

### DDoS Mitigation

**Layer 7 (Application)**:
- Rate limiting per client (5 RPS)
- Request size limits (10 KB body)
- Timeout enforcement (30s max)
- Connection limits per IP

**Layer 4 (Transport)**:
- SYN flood protection (at load balancer)
- Connection rate limiting
- IP-based blacklisting

**Layer 3 (Network)**:
- Cloudflare / AWS Shield integration
- Geographic filtering
- Traffic anomaly detection

---

## Logging and Monitoring

### Security Logging

**Log Structure**:
```json
{
  "timestamp": "2025-11-21T11:34:38Z",
  "level": "INFO",
  "event": "event_processed",
  "correlation_id": "req-abc123",
  "client_id": "client-xyz",
  "accepted": true,
  "moral_value": 0.8,
  "phase": "wake",
  "latency_ms": 5.2
}
```

**Logged Events**:
- ✅ Authentication attempts (success/failure)
- ✅ Rate limit violations
- ✅ Input validation failures
- ✅ Moral filter rejections
- ✅ System state changes
- ✅ Error conditions

**Log Retention**:
- Standard logs: 30 days
- Security logs: 90 days
- Audit logs: 1 year

**Log Protection**:
- No PII (personally identifiable information)
- Sanitized prompts/responses (truncated)
- Structured JSON format
- Centralized log aggregation

### Security Monitoring

**Metrics to Monitor**:
```yaml
# Authentication
auth_attempts_total{status="success|failure"}
auth_failures_by_client{client_id}

# Rate Limiting
rate_limit_violations_total{client_id}
requests_rejected_total{reason="rate_limit"}

# System Health
memory_usage_bytes
cpu_usage_percent
request_latency_seconds{quantile}

# Security Events
moral_filter_rejections_total
input_validation_errors_total{error_type}
```

**Alerting Thresholds**:
- Auth failures: >10 per minute from single IP
- Rate limit violations: >100 per hour
- Memory usage: >90% capacity
- Error rate: >5% of requests

---

## Dependency Management

### Dependency Security

**Tools**:
- `pip-audit`: Scan for known vulnerabilities
- `safety`: Check against vulnerability database
- Dependabot: Automated dependency updates

**Process**:
1. Weekly automated scans in CI/CD
2. Critical vulnerabilities: Patch within 7 days
3. High vulnerabilities: Patch within 14 days
4. Medium/Low: Patch in next release

**Current Dependencies** (security-relevant):
```
numpy>=2.0.0           # Numerical operations
fastapi>=0.110.0       # Web framework
uvicorn>=0.29.0        # ASGI server
pydantic>=2.0.0        # Data validation
prometheus-client>=0.20.0  # Metrics
```

### Supply Chain Security

**Practices**:
- ✅ Pin exact versions in requirements.txt
- ✅ Verify package checksums
- ✅ Use trusted package sources (PyPI)
- ✅ Review dependency tree for suspicious packages
- ⚠️ SBOM generation (planned for v1.1)
- ⚠️ Signature verification (planned for v1.1)

---

## Secure Deployment Guidelines

### Environment Configuration

**Required Environment Variables**:
```bash
# Authentication
export API_KEY="<strong-random-token-32-bytes-min>"

# Rate Limiting
export RATE_LIMIT_RPS="5"
export RATE_LIMIT_BURST="10"

# TLS Configuration
export TLS_CERT_PATH="/path/to/cert.pem"
export TLS_KEY_PATH="/path/to/key.pem"

# Logging
export LOG_LEVEL="INFO"
export LOG_FORMAT="json"
```

**Security Hardening**:
```bash
# Run as non-root user
adduser --system --no-create-home mlsdm
su - mlsdm

# Restrict file permissions
chmod 600 .env
chmod 700 /app

# Enable firewall
ufw allow 8000/tcp
ufw enable

# Disable unnecessary services
systemctl disable <unused-services>
```

### Network Security

**Firewall Rules**:
```bash
# Allow inbound HTTPS only
iptables -A INPUT -p tcp --dport 443 -j ACCEPT
iptables -A INPUT -p tcp --dport 8000 -j ACCEPT  # Internal API
iptables -A INPUT -j DROP
```

**TLS Configuration**:
```python
# uvicorn with TLS
uvicorn app:app \
    --host 0.0.0.0 \
    --port 443 \
    --ssl-keyfile /path/to/key.pem \
    --ssl-certfile /path/to/cert.pem \
    --ssl-version TLSv1_2 \
    --ssl-ciphers "ECDHE+AESGCM:ECDHE+CHACHA20"
```

### Container Security

**Docker Best Practices**:
```dockerfile
# Use minimal base image
FROM python:3.12-slim

# Run as non-root
RUN useradd -m -u 1000 mlsdm
USER mlsdm

# Copy only necessary files
COPY --chown=mlsdm:mlsdm requirements.txt .
COPY --chown=mlsdm:mlsdm src/ ./src/

# Read-only filesystem
RUN chmod -R 555 /app

# Health check
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8000/health || exit 1
```

---

## Security Testing

### Testing Strategy

1. **Static Analysis**
   - `bandit`: Python security linter
   - `ruff`: Code quality checks
   - `mypy`: Type safety verification

2. **Dependency Scanning**
   - `pip-audit`: Known vulnerability scanning
   - `safety`: Security advisory checking

3. **Dynamic Testing**
   - Integration tests with malicious inputs
   - Fuzzing with hypothesis
   - Load testing with locust

4. **Penetration Testing**
   - ⚠️ Planned for v1.1
   - OWASP Top 10 coverage
   - API security testing

### Test Coverage

**Security Test Categories**:
- ✅ Input validation bypass attempts
- ✅ Authentication bypass attempts
- ✅ Rate limit circumvention
- ✅ Memory exhaustion attacks
- ✅ Timing attacks on authentication
- ⚠️ SQL injection (N/A - no database)
- ⚠️ XSS (N/A - no HTML rendering)

---

## Compliance and Standards

### Standards Alignment

| Standard | Applicable | Status | Notes |
|----------|-----------|---------|-------|
| **OWASP Top 10** | Yes | Partial | No SQL/XSS risks (no DB/HTML) |
| **NIST CSF** | Yes | Aligned | Identify, Protect, Detect principles |
| **CIS Controls** | Yes | Partial | Controls 1-8 applicable |
| **SOC 2** | Optional | N/A | For enterprise deployments |
| **ISO 27001** | Optional | N/A | For enterprise deployments |

### OWASP Top 10 Coverage

| Risk | Applicable | Mitigation | Status |
|------|-----------|------------|---------|
| A01: Broken Access Control | Yes | Authentication + rate limiting | ✅ Implemented |
| A02: Cryptographic Failures | Yes | TLS + token storage | ✅ Implemented |
| A03: Injection | Partial | Input validation | ✅ Implemented |
| A04: Insecure Design | Yes | Threat modeling | ✅ Implemented |
| A05: Security Misconfiguration | Yes | Hardening guides | ✅ Documented |
| A06: Vulnerable Components | Yes | Dependency scanning | ✅ Implemented |
| A07: Auth Failures | Yes | Secure authentication | ✅ Implemented |
| A08: Data Integrity Failures | Partial | Input validation | ✅ Implemented |
| A09: Logging Failures | Yes | Structured logging | ✅ Implemented |
| A10: SSRF | No | No outbound requests | N/A |

---

## Security Roadmap

### v1.0 (Current)
- ✅ Input validation
- ✅ Authentication (API key)
- ✅ Rate limiting
- ✅ Memory protection
- ✅ Dependency scanning
- ✅ Security logging

### v1.1 (Planned)
- ⚠️ OAuth 2.0 / OpenID Connect
- ⚠️ mTLS support
- ⚠️ RBAC (role-based access control)
- ⚠️ SBOM generation
- ⚠️ Penetration testing
- ⚠️ Security audit

### v1.2 (Future)
- ⚠️ WAF integration
- ⚠️ Intrusion detection
- ⚠️ Anomaly detection
- ⚠️ Security orchestration (SOAR)

---

## Incident Response

### Response Process

1. **Detection**
   - Automated monitoring alerts
   - User reports
   - Security research disclosures

2. **Containment**
   - Isolate affected systems
   - Block malicious traffic
   - Revoke compromised credentials

3. **Eradication**
   - Patch vulnerabilities
   - Remove malicious code
   - Update dependencies

4. **Recovery**
   - Restore normal operations
   - Verify security controls
   - Monitor for recurrence

5. **Post-Incident**
   - Document lessons learned
   - Update security controls
   - Public disclosure (if applicable)

### Contact Information

**Security Team**: See [GitHub Security Advisories](https://github.com/neuron7x/mlsdm/security)

---

**Document Status:** Production  
**Review Cycle:** Quarterly  
**Last Reviewed:** November 2025  
**Next Review:** February 2026
