"""
Comprehensive API Contract Tests for MLSDM.

This module provides thorough contract tests to ensure:
1. All endpoints return expected response schemas
2. Error responses follow the ErrorResponse format
3. Response structure is stable (regression protection)
4. Health/readiness endpoints reflect actual system state

These tests serve as a contract guarantee - any breaking change
to the API response structure should be detected by these tests.
"""

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def setup_environment():
    """Set up test environment."""
    # Disable rate limiting for tests
    os.environ["DISABLE_RATE_LIMIT"] = "1"
    # Use local stub backend
    os.environ["LLM_BACKEND"] = "local_stub"
    yield
    # Cleanup
    if "DISABLE_RATE_LIMIT" in os.environ:
        del os.environ["DISABLE_RATE_LIMIT"]


@pytest.fixture
def client():
    """Create a test client with rate limiting disabled."""
    from mlsdm.api.app import app

    return TestClient(app)


# ============================================================================
# GenerateResponse Contract Tests
# ============================================================================


class TestGenerateResponseContract:
    """Test that /generate endpoint response follows the GenerateResponse schema.

    Contract fields that MUST be present:
    - response (str): Generated response text
    - phase (str): Current cognitive phase
    - accepted (bool): Whether the request was accepted

    Optional fields (may be null):
    - metrics: Performance timing metrics
    - safety_flags: Safety validation results
    - memory_stats: Memory state statistics
    - moral_score: Computed moral score (new in v1.2)
    - aphasia_flags: Aphasia detection flags (new in v1.2)
    - emergency_shutdown: Emergency shutdown status (new in v1.2)
    - latency_ms: Request latency in milliseconds
    - cognitive_state: Aggregated cognitive state
    """

    def test_generate_response_has_required_fields(self, client):
        """GenerateResponse must have all required contract fields."""
        response = client.post("/generate", json={"prompt": "Test prompt"})
        assert response.status_code == 200

        data = response.json()

        # Required fields - MUST be present
        assert "response" in data, "response field is required"
        assert "phase" in data, "phase field is required"
        assert "accepted" in data, "accepted field is required"

        # Type validation
        assert isinstance(data["response"], str), "response must be string"
        assert isinstance(data["phase"], str), "phase must be string"
        assert isinstance(data["accepted"], bool), "accepted must be boolean"

    def test_generate_response_phase_values(self, client):
        """Phase field must be a recognized value."""
        response = client.post("/generate", json={"prompt": "Test"})
        assert response.status_code == 200

        data = response.json()
        # Phase should be one of the known values
        assert data["phase"] in ["wake", "sleep", "unknown"], (
            f"Unexpected phase value: {data['phase']}"
        )

    def test_generate_response_optional_fields_structure(self, client):
        """Optional fields must follow expected structure when present."""
        response = client.post("/generate", json={"prompt": "Test"})
        assert response.status_code == 200

        data = response.json()

        # Optional fields should be present (can be None)
        assert "metrics" in data
        assert "safety_flags" in data
        assert "memory_stats" in data

        # If present, validate structure
        if data["metrics"] is not None:
            assert isinstance(data["metrics"], dict)

        if data["safety_flags"] is not None:
            assert isinstance(data["safety_flags"], dict)

        if data["memory_stats"] is not None:
            assert isinstance(data["memory_stats"], dict)

    def test_generate_with_moral_value_parameter(self, client):
        """Generate with moral_value parameter works correctly."""
        response = client.post(
            "/generate",
            json={"prompt": "Test", "moral_value": 0.8}
        )
        assert response.status_code == 200

        data = response.json()
        assert "response" in data
        assert "accepted" in data

    def test_generate_with_max_tokens_parameter(self, client):
        """Generate with max_tokens parameter works correctly."""
        response = client.post(
            "/generate",
            json={"prompt": "Test", "max_tokens": 128}
        )
        assert response.status_code == 200

        data = response.json()
        assert "response" in data


# ============================================================================
# Health/Readiness Contract Tests
# ============================================================================


class TestHealthContract:
    """Test health endpoint contracts.

    Contract fields:
    - /health: SimpleHealthStatus {status}
    - /health/liveness: HealthStatus {status, timestamp}
    - /health/readiness: ReadinessStatus {ready, status, timestamp, checks, cognitive_state?}
    - /health/detailed: DetailedHealthStatus {status, timestamp, uptime_seconds, system, ...}
    """

    def test_health_simple_contract(self, client):
        """GET /health returns SimpleHealthStatus."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_liveness_contract(self, client):
        """GET /health/liveness returns HealthStatus with timestamp."""
        response = client.get("/health/liveness")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert data["status"] == "alive"
        assert isinstance(data["timestamp"], (int, float))
        assert data["timestamp"] > 0

    def test_health_readiness_contract(self, client):
        """GET /health/readiness returns ReadinessStatus with checks."""
        response = client.get("/health/readiness")
        # Can be 200 or 503 depending on system state
        assert response.status_code in [200, 503]

        data = response.json()

        # Required fields
        assert "ready" in data
        assert "status" in data
        assert "timestamp" in data
        assert "checks" in data

        # Type validation
        assert isinstance(data["ready"], bool)
        assert data["status"] in ["ready", "not_ready"]
        assert isinstance(data["timestamp"], (int, float))
        assert isinstance(data["checks"], dict)

        # Required checks
        assert "memory_manager" in data["checks"]
        assert "memory_available" in data["checks"]
        assert "cpu_available" in data["checks"]

    def test_health_readiness_cognitive_state_optional(self, client):
        """GET /health/readiness may include cognitive_state."""
        response = client.get("/health/readiness")
        assert response.status_code in [200, 503]

        data = response.json()

        # cognitive_state is optional (may be None or missing in older versions)
        if "cognitive_state" in data and data["cognitive_state"] is not None:
            cognitive_state = data["cognitive_state"]
            # Validate safe fields are present
            assert "phase" in cognitive_state
            assert "emergency_shutdown" in cognitive_state

    def test_health_detailed_contract(self, client):
        """GET /health/detailed returns DetailedHealthStatus."""
        response = client.get("/health/detailed")
        assert response.status_code in [200, 503]

        data = response.json()

        # Required fields
        assert "status" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "system" in data

        # Type validation
        assert data["status"] in ["healthy", "unhealthy"]
        assert isinstance(data["timestamp"], (int, float))
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0
        assert isinstance(data["system"], dict)

        # Optional fields
        assert "memory_state" in data
        assert "phase" in data
        assert "statistics" in data

    def test_health_detailed_system_info(self, client):
        """GET /health/detailed system info contains expected fields."""
        response = client.get("/health/detailed")
        assert response.status_code in [200, 503]

        data = response.json()
        system = data["system"]

        # Should have system resource info
        # (fields may vary based on system, but common ones expected)
        assert isinstance(system, dict)


# ============================================================================
# ErrorResponse Contract Tests
# ============================================================================


class TestErrorResponseContract:
    """Test that all errors follow the ErrorResponse schema.

    ErrorResponse schema:
    {
        "error": {
            "error_type": str,
            "message": str,
            "details": dict | null
        }
    }

    Or for 422 validation errors, FastAPI format:
    {
        "detail": [
            {"loc": [...], "msg": str, "type": str, ...}
        ]
    }
    """

    def test_400_validation_error_format(self, client):
        """400 errors follow ErrorResponse schema."""
        response = client.post("/generate", json={"prompt": "   "})
        assert response.status_code == 400

        data = response.json()
        assert "error" in data

        error = data["error"]
        assert "error_type" in error
        assert "message" in error
        assert "details" in error  # Can be null

        assert error["error_type"] == "validation_error"

    def test_422_pydantic_validation_format(self, client):
        """422 validation errors follow FastAPI/Pydantic format."""
        response = client.post("/generate", json={"prompt": ""})
        assert response.status_code == 422

        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], list)

        # Each error should have loc, msg, type
        for error in data["detail"]:
            assert "loc" in error
            assert "msg" in error
            assert "type" in error

    def test_422_invalid_moral_value_error(self, client):
        """Invalid moral_value returns 422 with validation details."""
        response = client.post(
            "/generate",
            json={"prompt": "Test", "moral_value": 2.0}  # Out of range
        )
        assert response.status_code == 422

        data = response.json()
        assert "detail" in data

    def test_422_invalid_max_tokens_error(self, client):
        """Invalid max_tokens returns 422 with validation details."""
        response = client.post(
            "/generate",
            json={"prompt": "Test", "max_tokens": -1}  # Invalid
        )
        assert response.status_code == 422

        data = response.json()
        assert "detail" in data


# ============================================================================
# Infer Endpoint Contract Tests
# ============================================================================


class TestInferResponseContract:
    """Test that /infer endpoint response follows the InferResponse schema.

    Contract fields:
    - response (str): Generated response text
    - accepted (bool): Whether the request was accepted
    - phase (str): Current cognitive phase

    Optional fields:
    - moral_metadata: Moral filtering metadata
    - aphasia_metadata: Aphasia detection results
    - rag_metadata: RAG retrieval metadata
    - timing: Performance timing
    - governance: Full governance state
    """

    def test_infer_response_has_required_fields(self, client):
        """InferResponse must have all required contract fields."""
        response = client.post("/infer", json={"prompt": "Test prompt"})
        assert response.status_code == 200

        data = response.json()

        # Required fields
        assert "response" in data
        assert "accepted" in data
        assert "phase" in data

        # Type validation
        assert isinstance(data["response"], str)
        assert isinstance(data["accepted"], bool)
        assert isinstance(data["phase"], str)

    def test_infer_response_optional_fields(self, client):
        """InferResponse optional fields follow expected structure."""
        response = client.post("/infer", json={"prompt": "Test"})
        assert response.status_code == 200

        data = response.json()

        # Optional fields should be present (can be None)
        assert "moral_metadata" in data
        assert "aphasia_metadata" in data
        assert "rag_metadata" in data
        assert "timing" in data
        assert "governance" in data

    def test_infer_moral_metadata_structure(self, client):
        """Moral metadata has expected structure."""
        response = client.post(
            "/infer",
            json={"prompt": "Test", "secure_mode": True, "moral_value": 0.5}
        )
        assert response.status_code == 200

        data = response.json()
        moral = data.get("moral_metadata")
        assert moral is not None

        # Expected fields in moral_metadata
        assert "threshold" in moral or "applied_moral_value" in moral
        assert "secure_mode" in moral

    def test_infer_rag_metadata_structure(self, client):
        """RAG metadata has expected structure when enabled."""
        response = client.post(
            "/infer",
            json={"prompt": "Test", "rag_enabled": True, "context_top_k": 5}
        )
        assert response.status_code == 200

        data = response.json()
        rag = data.get("rag_metadata")
        assert rag is not None

        # Expected fields
        assert "enabled" in rag
        assert "context_items_retrieved" in rag
        assert "top_k" in rag
        assert rag["enabled"] is True


# ============================================================================
# Status Endpoint Contract Tests
# ============================================================================


class TestStatusResponseContract:
    """Test that /status endpoint response follows expected schema."""

    def test_status_response_structure(self, client):
        """Status response has all expected fields."""
        response = client.get("/status")
        assert response.status_code == 200

        data = response.json()

        # Required fields
        assert "status" in data
        assert "version" in data
        assert "backend" in data
        assert "system" in data
        assert "config" in data

        # Value validation
        assert data["status"] == "ok"
        assert isinstance(data["version"], str)
        assert isinstance(data["backend"], str)

    def test_status_system_info(self, client):
        """Status system info has expected fields."""
        response = client.get("/status")
        assert response.status_code == 200

        data = response.json()
        system = data["system"]

        # Expected system fields
        assert "memory_mb" in system
        assert "cpu_percent" in system

    def test_status_config_info(self, client):
        """Status config info has expected fields."""
        response = client.get("/status")
        assert response.status_code == 200

        data = response.json()
        config = data["config"]

        # Expected config fields
        assert "dimension" in config
        assert "rate_limiting_enabled" in config


# ============================================================================
# Response Headers Contract Tests
# ============================================================================


class TestResponseHeadersContract:
    """Test that responses include expected headers."""

    def test_security_headers_present(self, client):
        """Security headers are present in responses."""
        response = client.get("/health")
        assert response.status_code == 200

        # Security headers from SecurityHeadersMiddleware
        assert "x-content-type-options" in response.headers
        assert "x-frame-options" in response.headers

    def test_request_id_header_present(self, client):
        """X-Request-ID header is present in responses."""
        response = client.get("/health")
        assert response.status_code == 200

        # Request ID from RequestIDMiddleware
        assert "x-request-id" in response.headers
        assert len(response.headers["x-request-id"]) > 0

    def test_content_type_is_json(self, client):
        """Content-Type is application/json for JSON endpoints."""
        response = client.get("/health")
        assert response.status_code == 200
        assert "application/json" in response.headers.get("content-type", "")


# ============================================================================
# Schema Backward Compatibility Tests
# ============================================================================


class TestSchemaBackwardCompatibility:
    """Test that response schemas maintain backward compatibility.

    These tests verify that adding new optional fields doesn't break
    existing clients that only expect the original fields.
    """

    def test_generate_response_minimal_parsing(self, client):
        """Client only parsing required fields should work."""
        response = client.post("/generate", json={"prompt": "Test"})
        assert response.status_code == 200

        data = response.json()

        # Minimal parsing - only required fields
        result = {
            "response": data["response"],
            "phase": data["phase"],
            "accepted": data["accepted"],
        }

        # Should not raise any errors
        assert result["response"] is not None
        assert result["phase"] is not None
        assert result["accepted"] is not None

    def test_health_readiness_minimal_parsing(self, client):
        """Client only parsing required fields should work."""
        response = client.get("/health/readiness")
        assert response.status_code in [200, 503]

        data = response.json()

        # Minimal parsing - only required fields
        result = {
            "ready": data["ready"],
            "status": data["status"],
            "timestamp": data["timestamp"],
            "checks": data["checks"],
        }

        # Should not raise any errors
        assert result["ready"] is not None
        assert result["status"] is not None

    def test_infer_response_minimal_parsing(self, client):
        """Client only parsing required fields should work."""
        response = client.post("/infer", json={"prompt": "Test"})
        assert response.status_code == 200

        data = response.json()

        # Minimal parsing - only required fields
        result = {
            "response": data["response"],
            "accepted": data["accepted"],
            "phase": data["phase"],
        }

        # Should not raise any errors
        assert result["response"] is not None
        assert result["accepted"] is not None
        assert result["phase"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
