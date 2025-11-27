# API Overview

Quick reference guide for MLSDM API endpoints and SDK usage.

## Quick Start

### Starting the API Server

```bash
# Development mode
uvicorn mlsdm.api.app:app --reload --host 0.0.0.0 --port 8000

# Production mode
CONFIG_PATH=config/production.yaml uvicorn mlsdm.api.app:app --host 0.0.0.0 --port 8000

# With workers
uvicorn mlsdm.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CONFIG_PATH` | Path to configuration file | `config/default_config.yaml` |
| `LLM_BACKEND` | LLM backend (`local_stub`, `openai`) | `local_stub` |
| `DISABLE_RATE_LIMIT` | Disable rate limiting (testing) | `0` |
| `OTEL_SDK_DISABLED` | Disable OpenTelemetry tracing | `false` |
| `OTEL_EXPORTER_TYPE` | Tracing exporter type | `none` |

---

## Endpoints Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Simple health check |
| `/health/liveness` | GET | Kubernetes liveness probe |
| `/health/readiness` | GET | Kubernetes readiness probe |
| `/health/detailed` | GET | Detailed health status |
| `/health/metrics` | GET | Prometheus metrics |
| `/status` | GET | Extended service status |
| `/generate` | POST | Generate response (simple API) |
| `/infer` | POST | Generate response (extended governance API) |

---

## Health Endpoints

### GET /health

Simple health check for load balancers.

**Response:**
```json
{
  "status": "healthy"
}
```

**Status Codes:**
- `200 OK`: Service is running

---

### GET /health/liveness

Kubernetes-compatible liveness probe.

**Response:**
```json
{
  "status": "alive",
  "timestamp": 1732693200.0
}
```

**Status Codes:**
- `200 OK`: Process is alive

---

### GET /health/readiness

Kubernetes-compatible readiness probe.

**Response (ready):**
```json
{
  "ready": true,
  "status": "ready",
  "timestamp": 1732693200.0,
  "checks": {
    "memory_manager": true,
    "memory_available": true,
    "cpu_available": true
  }
}
```

**Response (not ready):**
```json
{
  "ready": false,
  "status": "not_ready",
  "timestamp": 1732693200.0,
  "checks": {
    "memory_manager": false,
    "memory_available": true,
    "cpu_available": true
  }
}
```

**Status Codes:**
- `200 OK`: Ready to accept traffic
- `503 Service Unavailable`: Not ready

---

### GET /health/detailed

Detailed health information including system resources.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1732693200.0,
  "uptime_seconds": 3600.5,
  "system": {
    "memory_percent": 45.2,
    "memory_available_mb": 8192.0,
    "cpu_percent": 15.3,
    "cpu_count": 8,
    "disk_percent": 30.5
  },
  "memory_state": {
    "L1_norm": 3.14,
    "L2_norm": 2.71,
    "L3_norm": 1.41
  },
  "phase": "wake",
  "statistics": {
    "total_events_processed": 1000,
    "accepted_events_count": 950,
    "latent_events_count": 50,
    "moral_filter_threshold": 0.5,
    "avg_latency_ms": 15.2
  }
}
```

**Status Codes:**
- `200 OK`: System is healthy
- `503 Service Unavailable`: System is unhealthy

---

### GET /health/metrics

Prometheus-format metrics for monitoring.

**Response:** `text/plain`
```
# HELP mlsdm_events_processed_total Total events processed
# TYPE mlsdm_events_processed_total counter
mlsdm_events_processed_total 1000

# HELP mlsdm_memory_usage_bytes Memory usage in bytes
# TYPE mlsdm_memory_usage_bytes gauge
mlsdm_memory_usage_bytes 52428800
```

---

## Status Endpoint

### GET /status

Extended service status with system information.

**Response:**
```json
{
  "status": "ok",
  "version": "1.2.0",
  "backend": "local_stub",
  "system": {
    "memory_mb": 150.5,
    "cpu_percent": 12.3
  },
  "config": {
    "dimension": 384,
    "rate_limiting_enabled": true
  }
}
```

---

## Generation Endpoints

### POST /generate

Generate a response using the NeuroCognitiveEngine (simple API).

**Request:**
```json
{
  "prompt": "Explain machine learning",
  "max_tokens": 256,
  "moral_value": 0.7
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt` | string | Yes | Input text (min 1 char) |
| `max_tokens` | integer | No | Max tokens (1-4096) |
| `moral_value` | float | No | Moral threshold (0.0-1.0) |

**Response:**
```json
{
  "response": "Machine learning is a subset of AI...",
  "phase": "wake",
  "accepted": true,
  "metrics": {
    "timing": {
      "total": 15.2,
      "generation": 12.1
    }
  },
  "safety_flags": {
    "validation_steps": [
      {"step": "moral_precheck", "passed": true}
    ],
    "rejected_at": null
  },
  "memory_stats": {
    "step": 1,
    "moral_threshold": 0.7,
    "context_items": 3
  }
}
```

**Status Codes:**
- `200 OK`: Successful generation
- `400 Bad Request`: Invalid input (e.g., whitespace-only prompt)
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

---

### POST /infer

Generate a response with extended governance options.

**Request:**
```json
{
  "prompt": "Explain quantum computing",
  "moral_value": 0.6,
  "max_tokens": 256,
  "secure_mode": true,
  "aphasia_mode": false,
  "rag_enabled": true,
  "context_top_k": 5,
  "user_intent": "analytical"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | Input text (min 1 char) |
| `moral_value` | float | No | 0.5 | Moral threshold (0.0-1.0) |
| `max_tokens` | integer | No | 512 | Max tokens (1-4096) |
| `secure_mode` | boolean | No | false | Enhanced security filtering |
| `aphasia_mode` | boolean | No | false | Enable aphasia detection |
| `rag_enabled` | boolean | No | true | Enable RAG retrieval |
| `context_top_k` | integer | No | 5 | Context items for RAG (1-100) |
| `user_intent` | string | No | null | Intent category |

**Response:**
```json
{
  "response": "Quantum computing leverages quantum mechanics...",
  "accepted": true,
  "phase": "wake",
  "moral_metadata": {
    "threshold": 0.5,
    "secure_mode": true,
    "applied_moral_value": 0.8
  },
  "aphasia_metadata": {
    "enabled": false,
    "detected": false,
    "severity": 0.0
  },
  "rag_metadata": {
    "enabled": true,
    "context_items_retrieved": 3,
    "top_k": 5
  },
  "timing": {
    "total": 18.5,
    "generation": 15.2,
    "rag_retrieval": 2.1
  },
  "governance": { ... }
}
```

#### Secure Mode

When `secure_mode=true`:
- Moral threshold is increased by 0.2 (capped at 1.0)
- Enhanced filtering for security-critical contexts

**Example:**
```json
{
  "prompt": "Process sensitive data",
  "secure_mode": true,
  "moral_value": 0.5
}
```
→ `applied_moral_value = 0.7` (0.5 + 0.2)

---

## SDK Client

### Python SDK

```python
from mlsdm.sdk import NeuroCognitiveClient

# Using local stub backend
client = NeuroCognitiveClient(backend="local_stub")

# Generate a response
result = client.generate(
    prompt="What is consciousness?",
    max_tokens=256,
    moral_value=0.7
)

print(f"Response: {result['response']}")
print(f"Phase: {result['mlsdm']['phase']}")
print(f"Timing: {result['timing']}")
```

### With OpenAI Backend

```python
from mlsdm.sdk import NeuroCognitiveClient

client = NeuroCognitiveClient(
    backend="openai",
    api_key="sk-...",
    model="gpt-4"
)

result = client.generate(
    prompt="Explain neural networks",
    max_tokens=512
)
```

### Error Handling

```python
from mlsdm.adapters import LLMProviderError, LLMTimeoutError

try:
    result = client.generate("Test prompt")
except LLMTimeoutError as e:
    print(f"Request timed out after {e.timeout_seconds}s")
except LLMProviderError as e:
    print(f"Provider error: {e.provider_id}: {e}")
```

---

## Error Response Format

All 4xx and 5xx errors (except 422) follow this structure:

```json
{
  "error": {
    "error_type": "validation_error",
    "message": "Human-readable error message",
    "details": { "field": "prompt" }
  }
}
```

422 errors use FastAPI's default validation format:

```json
{
  "detail": [
    {
      "loc": ["body", "prompt"],
      "msg": "String should have at least 1 character",
      "type": "string_too_short"
    }
  ]
}
```

---

## Response Headers

All responses include:

| Header | Description |
|--------|-------------|
| `X-Request-ID` | Unique request identifier |
| `X-Response-Time` | Response time in ms |
| `X-Content-Type-Options` | `nosniff` |
| `X-Frame-Options` | `DENY` |
| `X-XSS-Protection` | `1; mode=block` |
| `Content-Security-Policy` | `default-src 'self'` |

---

## Rate Limiting

Default: 5 requests per second per client.

When exceeded:
```json
{
  "error": {
    "error_type": "rate_limit_exceeded",
    "message": "Rate limit exceeded. Maximum 5 requests per second.",
    "details": null
  }
}
```

Status Code: `429 Too Many Requests`

Disable for testing: `DISABLE_RATE_LIMIT=1`

---

## See Also

- [API Reference](API_REFERENCE.md) - Detailed API documentation
- [Architecture Specification](ARCHITECTURE_SPEC.md) - System architecture
- [Configuration Guide](CONFIGURATION_GUIDE.md) - Configuration options
- [Security Policy](SECURITY_POLICY.md) - Security guidelines
