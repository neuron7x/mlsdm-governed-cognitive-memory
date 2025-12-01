# MLSDM Event API Contract

**Document Version:** 1.0.0  
**Last Updated:** December 2025  
**Status:** Production

This document defines the API contract for the event processing endpoints (`/v1/process_event/` and `/v1/state/`). These endpoints handle cognitive event processing and system state queries.

---

## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [POST /v1/process_event/](#post-v1process_event)
  - [GET /v1/state/](#get-v1state)
- [Models](#models)
  - [EventInput](#eventinput)
  - [StateResponse](#stateresponse)
  - [ApiError](#apierror)
- [Error Handling](#error-handling)
- [Examples](#examples)

---

## Overview

The Event API provides endpoints for:
- **Event Processing**: Submit cognitive events for processing through the moral filter and memory system
- **State Queries**: Get current system state including memory layer norms and cognitive phase

All endpoints require OAuth2 Bearer token authentication.

---

## Authentication

All event endpoints require authentication:

```
Authorization: Bearer <API_KEY>
```

The API key is configured via the `API_KEY` environment variable.

---

## Endpoints

### POST /v1/process_event/

Process a cognitive event through the complete pipeline including moral filtering and memory updates.

**Request:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `event_vector` | `list[float]` | Yes | Embedding vector matching configured dimension (default: 10 for testing, 384 for production) |
| `moral_value` | `float` | Yes | Moral score for the event (0.0-1.0) |

**Response:** [StateResponse](#stateresponse)

**Error Responses:**

| Status | Code | Description |
|--------|------|-------------|
| 400 | `dimension_mismatch` | Vector dimension doesn't match configured dimension |
| 400 | `moral_value_invalid` | Moral value validation failed |
| 400 | `vector_validation_failed` | Vector contains invalid values (NaN, Inf) |
| 401 | `unauthorized` | Missing or invalid authentication |
| 422 | (Pydantic) | Request body validation failed |
| 429 | `rate_limit_exceeded` | Rate limit exceeded (5 RPS per client) |
| 500 | `processing_error` | Internal error during processing |

**Request Example:**

```json
{
  "event_vector": [0.1, 0.2, -0.1, 0.3, ...],
  "moral_value": 0.75
}
```

**Response Example (200 OK):**

```json
{
  "L1_norm": 1.5,
  "L2_norm": 2.3,
  "L3_norm": 0.8,
  "current_phase": "wake",
  "latent_events_count": 10,
  "accepted_events_count": 85,
  "total_events_processed": 100,
  "moral_filter_threshold": 0.5
}
```

---

### GET /v1/state/

Get the current system state snapshot.

**Response:** [StateResponse](#stateresponse)

**Error Responses:**

| Status | Code | Description |
|--------|------|-------------|
| 401 | `unauthorized` | Missing or invalid authentication |
| 429 | `rate_limit_exceeded` | Rate limit exceeded |
| 500 | `state_retrieval_error` | Error retrieving system state |

**Response Example (200 OK):**

```json
{
  "L1_norm": 1.5,
  "L2_norm": 2.3,
  "L3_norm": 0.8,
  "current_phase": "wake",
  "latent_events_count": 10,
  "accepted_events_count": 85,
  "total_events_processed": 100,
  "moral_filter_threshold": 0.5
}
```

---

## Models

### EventInput

Request model for `/v1/process_event/` endpoint.

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `event_vector` | `list[float]` | Yes | Non-empty, finite values | Embedding vector for the cognitive event |
| `moral_value` | `float` | Yes | 0.0 ≤ value ≤ 1.0 | Moral score for the event |

**Validation Rules:**
- `event_vector` must have length equal to configured dimension (default: 384, test: 10)
- All values in `event_vector` must be finite (no NaN, Inf, -Inf)
- `moral_value` must be in range [0.0, 1.0]

```python
from pydantic import BaseModel, Field, field_validator

class EventInput(BaseModel):
    event_vector: list[float] = Field(..., min_length=1)
    moral_value: float = Field(..., ge=0.0, le=1.0)
    
    @field_validator("event_vector")
    @classmethod
    def validate_vector_values(cls, v):
        # Validates all values are finite (not NaN or Inf)
        ...
```

---

### StateResponse

Response model for state endpoints.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `L1_norm` | `float` | ≥ 0.0 | Euclidean norm of L1 (working) memory |
| `L2_norm` | `float` | ≥ 0.0 | Euclidean norm of L2 (short-term) memory |
| `L3_norm` | `float` | ≥ 0.0 | Euclidean norm of L3 (long-term) memory |
| `current_phase` | `Literal["wake", "sleep"]` | - | Current cognitive rhythm phase |
| `latent_events_count` | `int` | ≥ 0 | Events in latent/pending state |
| `accepted_events_count` | `int` | ≥ 0 | Events accepted by moral filter |
| `total_events_processed` | `int` | ≥ 0 | Total events (accepted + rejected) |
| `moral_filter_threshold` | `float` | 0.0-1.0 | Current adaptive moral threshold |

```python
from pydantic import BaseModel, Field
from typing import Literal

class StateResponse(BaseModel):
    L1_norm: float = Field(..., ge=0.0)
    L2_norm: float = Field(..., ge=0.0)
    L3_norm: float = Field(..., ge=0.0)
    current_phase: Literal["wake", "sleep"]
    latent_events_count: int = Field(..., ge=0)
    accepted_events_count: int = Field(..., ge=0)
    total_events_processed: int = Field(..., ge=0)
    moral_filter_threshold: float = Field(..., ge=0.0, le=1.0)
```

---

### ApiError

Standardized error response model.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `code` | `str` | Yes | Machine-readable error code |
| `message` | `str` | Yes | Human-readable error message |
| `details` | `dict \| null` | No | Additional error context |

**Common Error Codes:**
- `validation_error` - Input validation failed
- `dimension_mismatch` - Vector dimension doesn't match
- `moral_value_invalid` - Moral value out of range
- `vector_validation_failed` - Vector contains invalid values
- `rate_limit_exceeded` - Too many requests
- `processing_error` - Event processing failed
- `state_retrieval_error` - State query failed

```python
from pydantic import BaseModel, Field

class ApiError(BaseModel):
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: dict | None = Field(None, description="Additional context")

class ApiErrorResponse(BaseModel):
    error: ApiError
```

---

## Error Handling

All error responses follow the `ApiErrorResponse` schema:

```json
{
  "error": {
    "code": "dimension_mismatch",
    "message": "Vector dimension mismatch: expected 10, got 100",
    "details": {
      "field": "event_vector",
      "expected_dimension": 10,
      "actual_dimension": 100
    }
  }
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Invalid input (dimension mismatch, invalid values) |
| 401 | Authentication required or failed |
| 422 | Pydantic validation failed (missing fields, type errors) |
| 429 | Rate limit exceeded |
| 500 | Internal server error |

---

## Examples

### Process Event (cURL)

```bash
curl -X POST "http://localhost:8000/v1/process_event/" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "event_vector": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "moral_value": 0.75
  }'
```

### Get State (cURL)

```bash
curl -X GET "http://localhost:8000/v1/state/" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### Python SDK Example

```python
import requests

API_BASE = "http://localhost:8000"
API_KEY = "your-api-key"

# Process event
response = requests.post(
    f"{API_BASE}/v1/process_event/",
    json={
        "event_vector": [0.1] * 10,  # Match configured dimension
        "moral_value": 0.75
    },
    headers={"Authorization": f"Bearer {API_KEY}"}
)

if response.status_code == 200:
    state = response.json()
    print(f"Phase: {state['current_phase']}")
    print(f"Moral threshold: {state['moral_filter_threshold']}")
else:
    error = response.json()["error"]
    print(f"Error: {error['code']} - {error['message']}")
```

---

## Contract Stability

These fields are part of the stable API contract:

- **EventInput**: `event_vector`, `moral_value`
- **StateResponse**: All 8 fields listed above
- **ApiError**: `code`, `message`, `details`

**Breaking changes require a major version bump.**

See `src/mlsdm/contracts/event_models.py` for the canonical Pydantic models.

---

## Related Documentation

- [API_CONTRACT.md](API_CONTRACT.md) - Main API contract reference
- [SECURITY_POLICY.md](../SECURITY_POLICY.md) - Security policies including rate limiting
- [FORMAL_INVARIANTS.md](FORMAL_INVARIANTS.md) - System invariants and constraints
