# MLSDM API Contract

**Document Version:** 1.1.0  
**Last Updated:** November 2025  
**Status:** Production

This document defines the HTTP API contract for the MLSDM (Governed Cognitive Memory) service. All endpoints are documented with their request/response schemas, HTTP status codes, and error formats.

---

## Table of Contents

- [Overview](#overview)
- [Contract Stability Guarantees](#contract-stability-guarantees)
- [Base URL](#base-url)
- [Authentication](#authentication)
- [Common Response Format](#common-response-format)
- [Endpoints](#endpoints)
  - [Health Endpoints](#health-endpoints)
  - [Generation Endpoints](#generation-endpoints)
  - [State Endpoints](#state-endpoints)
- [Error Responses](#error-responses)
- [Pydantic Schema Reference](#pydantic-schema-reference)
- [SDK Reference](#sdk-reference)

---

## Overview

The MLSDM API provides endpoints for:
- **Health checks**: Liveness, readiness, and detailed health status
- **Text generation**: Cognitive response generation with moral filtering
- **State management**: System state queries and event processing

All endpoints return JSON responses with standardized error formats.

---

## Contract Stability Guarantees

The following fields are considered **stable contract** and will **not change without a major version bump** (breaking change). Clients can safely rely on these fields being present and having the documented types.

### GenerateResponse Stable Fields
| Field | Type | Stability |
|-------|------|-----------|
| `response` | string | ✅ Stable |
| `phase` | string | ✅ Stable |
| `accepted` | boolean | ✅ Stable |

### HealthStatus Stable Fields
| Field | Type | Stability |
|-------|------|-----------|
| `status` | string | ✅ Stable |
| `timestamp` | float | ✅ Stable |

### ReadinessStatus Stable Fields
| Field | Type | Stability |
|-------|------|-----------|
| `ready` | boolean | ✅ Stable |
| `status` | string | ✅ Stable |
| `timestamp` | float | ✅ Stable |
| `checks` | object | ✅ Stable |

### ErrorResponse Stable Fields
| Field | Type | Stability |
|-------|------|-----------|
| `error.error_type` | string | ✅ Stable |
| `error.message` | string | ✅ Stable |
| `error.details` | object \| null | ✅ Stable |

**Note:** Optional/diagnostic fields (e.g., `metrics`, `safety_flags`, `memory_stats`) may be added or modified in minor versions without breaking changes.

---

## Base URL

```
http://<host>:8000
```

Default port is `8000` when running with uvicorn.

---

## Authentication

Some endpoints require OAuth2 Bearer token authentication:

```
Authorization: Bearer <API_KEY>
```

The API key is configured via the `API_KEY` environment variable. Rate limiting is enforced at 5 requests per second per client.

---

## Common Response Format

### Success Response
All successful responses return HTTP 200 with JSON body matching the endpoint's response schema.

### Error Response
All errors use the `ErrorResponse` schema:

```json
{
  "error": {
    "error_type": "string",
    "message": "string",
    "details": { } | null
  }
}
```

---

## Endpoints

### Health Endpoints

All health endpoints are under the `/health` prefix, with `/ready` as a convenience alias.

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|---------------|
| `/health` | GET | Simple health check | No |
| `/health/liveness` | GET | Kubernetes liveness probe | No |
| `/health/readiness` | GET | Kubernetes readiness probe | No |
| `/ready` | GET | Alias for /health/readiness | No |
| `/health/detailed` | GET | Detailed system health | No |
| `/health/metrics` | GET | Prometheus metrics | No |

#### GET /health

Simple health check endpoint.

**Response Schema:** `SimpleHealthStatus`

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Health status ("healthy") |

**Example Response (200 OK):**
```json
{
  "status": "healthy"
}
```

---

#### GET /health/liveness

Kubernetes liveness probe. Always returns 200 if the process is responsive.

**Response Schema:** `HealthStatus`

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Liveness status ("alive") |
| `timestamp` | float | Unix timestamp |

**Example Response (200 OK):**
```json
{
  "status": "alive",
  "timestamp": 1732649400.123
}
```

---

#### GET /health/readiness

Kubernetes readiness probe. Returns 200 if ready to accept traffic, 503 if not.

**Response Schema:** `ReadinessStatus`

| Field | Type | Description |
|-------|------|-------------|
| `ready` | boolean | Whether service is ready |
| `status` | string | "ready" or "not_ready" |
| `timestamp` | float | Unix timestamp |
| `checks` | object | Individual check results |
| `emergency_shutdown` | boolean \| null | Whether emergency shutdown is active (optional) |
| `cognitive_state` | object \| null | Aggregated cognitive state (optional, safe subset only) |

**Example Response (200 OK):**
```json
{
  "ready": true,
  "status": "ready",
  "timestamp": 1732649400.123,
  "checks": {
    "memory_manager": true,
    "memory_available": true,
    "cpu_available": true,
    "emergency_shutdown_inactive": true
  },
  "emergency_shutdown": false,
  "cognitive_state": {
    "phase": "wake",
    "stateless_mode": false,
    "rhythm_state": "wake",
    "emergency_shutdown": false
  }
}
```

**Example Response (503 Service Unavailable):**
```json
{
  "ready": false,
  "status": "not_ready",
  "timestamp": 1732649400.123,
  "checks": {
    "memory_manager": false,
    "memory_available": true,
    "cpu_available": true
  },
  "emergency_shutdown": null,
  "cognitive_state": null
}
```

---

#### GET /ready

Convenience alias for `/health/readiness`. Returns the same `ReadinessStatus` schema.

**Response Schema:** `ReadinessStatus` (same as `/health/readiness`)

---

#### GET /health/detailed

Detailed health status with system metrics.

**Response Schema:** `DetailedHealthStatus`

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | "healthy" or "unhealthy" |
| `timestamp` | float | Unix timestamp |
| `uptime_seconds` | float | Service uptime in seconds |
| `system` | object | System resource info |
| `memory_state` | object \| null | Memory layer norms |
| `phase` | string \| null | Current cognitive phase |
| `statistics` | object \| null | Processing statistics |

**Example Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": 1732649400.123,
  "uptime_seconds": 3600.5,
  "system": {
    "memory_percent": 45.2,
    "memory_available_mb": 8192.0,
    "memory_total_mb": 16384.0,
    "cpu_percent": 12.5,
    "cpu_count": 8,
    "disk_percent": 35.0
  },
  "memory_state": {
    "L1_norm": 1.5,
    "L2_norm": 2.3,
    "L3_norm": 0.8
  },
  "phase": "wake",
  "statistics": {
    "total_events_processed": 1000,
    "accepted_events_count": 850,
    "latent_events_count": 150,
    "moral_filter_threshold": 0.5,
    "avg_latency_ms": 15.2
  }
}
```

---

#### GET /health/metrics

Prometheus metrics endpoint.

**Response:** Plain text in Prometheus exposition format.

**Content-Type:** `text/plain`

---

### Generation Endpoints

#### POST /generate

Generate a response using the NeuroCognitiveEngine.

**Request Schema:** `GenerateRequest`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt` | string | Yes | Input text prompt (min 1 char) |
| `max_tokens` | integer | No | Max tokens to generate (1-4096) |
| `moral_value` | float | No | Moral threshold (0.0-1.0) |

**Response Schema:** `GenerateResponse`

| Field | Type | Description |
|-------|------|-------------|
| `response` | string | Generated response text |
| `phase` | string | Current cognitive phase |
| `accepted` | boolean | Whether request was accepted |
| `metrics` | object \| null | Performance timing metrics |
| `safety_flags` | object \| null | Safety validation results |
| `memory_stats` | object \| null | Memory state statistics |

**Example Request:**
```json
{
  "prompt": "What is consciousness?",
  "max_tokens": 256,
  "moral_value": 0.7
}
```

**Example Response (200 OK):**
```json
{
  "response": "NEURO-RESPONSE: What is consciousness?...",
  "phase": "wake",
  "accepted": true,
  "metrics": {
    "timing": {
      "pre_flight_ms": 0.5,
      "llm_call_ms": 10.2,
      "total": 12.5
    }
  },
  "safety_flags": {
    "validation_steps": ["moral_filter", "memory_check"],
    "rejected_at": null
  },
  "memory_stats": {
    "step": 42,
    "moral_threshold": 0.5,
    "context_items": 3
  }
}
```

**Error Responses:**

| Status | Error Type | Description |
|--------|------------|-------------|
| 400 | `validation_error` | Invalid input (empty prompt) |
| 422 | Validation Error | Pydantic validation failed |
| 429 | `rate_limit_exceeded` | Too many requests |
| 500 | `internal_error` | Server error |

---

#### POST /infer

Generate a response with extended governance options.

**Request Schema:** `InferRequest`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompt` | string | Yes | - | Input text prompt (min 1 char) |
| `moral_value` | float | No | 0.5 | Moral threshold (0.0-1.0) |
| `max_tokens` | integer | No | - | Max tokens (1-4096) |
| `secure_mode` | boolean | No | false | Enable enhanced security filtering |
| `aphasia_mode` | boolean | No | false | Enable aphasia detection/repair |
| `rag_enabled` | boolean | No | true | Enable RAG context retrieval |
| `context_top_k` | integer | No | 5 | Number of context items (1-100) |
| `user_intent` | string | No | - | User intent category |

**Response Schema:** `InferResponse`

| Field | Type | Description |
|-------|------|-------------|
| `response` | string | Generated response text |
| `accepted` | boolean | Whether request was accepted |
| `phase` | string | Current cognitive phase |
| `moral_metadata` | object \| null | Moral filtering metadata |
| `aphasia_metadata` | object \| null | Aphasia detection results |
| `rag_metadata` | object \| null | RAG retrieval metadata |
| `timing` | object \| null | Performance timing |
| `governance` | object \| null | Full governance state |

**Example Request:**
```json
{
  "prompt": "Explain quantum computing",
  "secure_mode": true,
  "moral_value": 0.6,
  "rag_enabled": true,
  "context_top_k": 3
}
```

**Example Response (200 OK):**
```json
{
  "response": "NEURO-RESPONSE: Explain quantum computing...",
  "accepted": true,
  "phase": "wake",
  "moral_metadata": {
    "threshold": 0.5,
    "secure_mode": true,
    "applied_moral_value": 0.8
  },
  "aphasia_metadata": null,
  "rag_metadata": {
    "enabled": true,
    "context_items_retrieved": 3,
    "top_k": 3
  },
  "timing": {
    "pre_flight_ms": 0.5,
    "llm_call_ms": 15.3,
    "total": 18.2
  },
  "governance": null
}
```

---

### State Endpoints

#### GET /status

Get extended service status with system info.

**Response Schema:** (inline)

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Service status ("ok") |
| `version` | string | API version |
| `backend` | string | LLM backend name |
| `system` | object | System metrics |
| `config` | object | Configuration info |

**Example Response (200 OK):**
```json
{
  "status": "ok",
  "version": "1.2.0",
  "backend": "local_stub",
  "system": {
    "memory_mb": 256.5,
    "cpu_percent": 15.2
  },
  "config": {
    "dimension": 384,
    "rate_limiting_enabled": true
  }
}
```

---

#### GET /v1/state/

Get system state (requires authentication).

**Response Schema:** `StateResponse`

| Field | Type | Description |
|-------|------|-------------|
| `L1_norm` | float | L1 memory layer norm |
| `L2_norm` | float | L2 memory layer norm |
| `L3_norm` | float | L3 memory layer norm |
| `current_phase` | string | Current cognitive phase |
| `latent_events_count` | integer | Latent events count |
| `accepted_events_count` | integer | Accepted events count |
| `total_events_processed` | integer | Total events processed |
| `moral_filter_threshold` | float | Current moral threshold |

---

#### POST /v1/process_event/

Process an event (requires authentication).

**Request Schema:** `EventInput`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `event_vector` | array[float] | Yes | Event embedding vector |
| `moral_value` | float | Yes | Moral value for filtering |

**Response Schema:** `StateResponse` (same as GET /v1/state/)

---

## Error Responses

### Standard Error Format

All API errors follow the `ErrorResponse` schema:

```json
{
  "error": {
    "error_type": "string",
    "message": "string",
    "details": { } | null
  }
}
```

### HTTP Status Codes

| Code | Error Type | Description |
|------|------------|-------------|
| 400 | `validation_error` | Invalid input data |
| 401 | `unauthorized` | Missing or invalid authentication |
| 422 | (Pydantic) | Request body validation failed |
| 429 | `rate_limit_exceeded` | Rate limit exceeded (5 RPS) |
| 500 | `internal_error` | Internal server error |
| 503 | Service Unavailable | Service not ready or overloaded |
| 504 | Gateway Timeout | Request processing timeout |

### Validation Error (422)

Pydantic validation errors follow FastAPI's standard format:

```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "prompt"],
      "msg": "String should have at least 1 character",
      "input": "",
      "ctx": {"min_length": 1}
    }
  ]
}
```

---

## Pydantic Schema Reference

All schemas are defined in `src/mlsdm/api/app.py` and `src/mlsdm/api/health.py`.

### Request Schemas

#### GenerateRequest
```python
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_tokens: int | None = Field(None, ge=1, le=4096)
    moral_value: float | None = Field(None, ge=0.0, le=1.0)
```

#### InferRequest
```python
class InferRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    moral_value: float | None = Field(None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(None, ge=1, le=4096)
    secure_mode: bool = Field(default=False)
    aphasia_mode: bool = Field(default=False)
    rag_enabled: bool = Field(default=True)
    context_top_k: int | None = Field(None, ge=1, le=100)
    user_intent: str | None = None
```

#### EventInput
```python
class EventInput(BaseModel):
    event_vector: list[float]
    moral_value: float
```

### Response Schemas

#### GenerateResponse
```python
class GenerateResponse(BaseModel):
    response: str
    phase: str
    accepted: bool
    metrics: dict[str, Any] | None = None
    safety_flags: dict[str, Any] | None = None
    memory_stats: dict[str, Any] | None = None
```

#### InferResponse
```python
class InferResponse(BaseModel):
    response: str
    accepted: bool
    phase: str
    moral_metadata: dict[str, Any] | None = None
    aphasia_metadata: dict[str, Any] | None = None
    rag_metadata: dict[str, Any] | None = None
    timing: dict[str, float] | None = None
    governance: dict[str, Any] | None = None
```

#### ErrorResponse
```python
class ErrorResponse(BaseModel):
    error: ErrorDetail

class ErrorDetail(BaseModel):
    error_type: str
    message: str
    details: dict[str, Any] | None = None
    debug_id: str | None = None
```

---

## SDK Reference

The MLSDM Python SDK provides two client interfaces:

### MLSDMHttpClient (HTTP Client)

For remote API server communication with typed responses and exception handling.

```python
from mlsdm.sdk import MLSDMHttpClient, MLSDMClientError, MLSDMServerError, MLSDMTimeoutError

# Create client
client = MLSDMHttpClient(base_url="http://localhost:8000", timeout=30.0)

# Generate with typed response
try:
    response = client.generate(
        prompt="What is consciousness?",
        moral_value=0.7,
        max_tokens=256
    )
    print(f"Response: {response.response}")
    print(f"Phase: {response.phase}")
    print(f"Accepted: {response.accepted}")
except MLSDMClientError as e:
    print(f"Client error [{e.status_code}]: {e.message}")
except MLSDMServerError as e:
    print(f"Server error [{e.status_code}]: {e.message}")
except MLSDMTimeoutError as e:
    print(f"Timeout after {e.timeout_seconds}s: {e.message}")
```

### SDK Exceptions

| Exception | Description | Attributes |
|-----------|-------------|------------|
| `MLSDMError` | Base exception | `message`, `debug_id` |
| `MLSDMClientError` | 4xx HTTP errors | `status_code`, `error_type`, `details` |
| `MLSDMServerError` | 5xx HTTP errors | `status_code`, `error_type` |
| `MLSDMTimeoutError` | Request timeout | `timeout_seconds` |
| `MLSDMConnectionError` | Connection failed | `url` |

### GenerateResponseDTO

The SDK returns a typed `GenerateResponseDTO` that mirrors the API's `GenerateResponse`:

```python
class GenerateResponseDTO(BaseModel):
    response: str          # Generated text (stable)
    phase: str             # Cognitive phase (stable)
    accepted: bool         # Request accepted (stable)
    metrics: dict | None   # Timing metrics
    safety_flags: dict | None
    memory_stats: dict | None
    moral_score: float | None
    aphasia_flags: dict | None
    emergency_shutdown: bool | None
    latency_ms: float | None
    cognitive_state: dict | None
```

---

## Document Maintenance

This API contract should be updated when:
1. New endpoints are added
2. Request/response schemas change
3. Error handling is modified
4. Authentication requirements change

**Owner:** Principal API & Boundary-Security Engineer
