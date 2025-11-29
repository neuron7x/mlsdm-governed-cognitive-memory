"""
API Contract Tests for MLSDM.

This module validates that all HTTP endpoints conform to their documented contracts
as specified in docs/API_CONTRACT.md. These tests serve as a guarantee that the
API schema and behavior remain stable across versions.

Test Categories:
- Health endpoint contracts
- Generate endpoint contract
- Infer endpoint contract
- Error response format
- Response schema validation
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


class TestGenerateEndpointContract:
    """Test /generate endpoint contract stability.

    These tests validate that GenerateResponse schema is stable:
    - response (str): Required, always present
    - phase (str): Required, always present
    - accepted (bool): Required, always present
    - metrics (dict | None): Optional
    - safety_flags (dict | None): Optional
    - memory_stats (dict | None): Optional
    """

    def test_generate_returns_200_with_valid_prompt(self, client):
        """POST /generate with valid prompt returns 200."""
        response = client.post("/generate", json={"prompt": "Hello, world!"})
        assert response.status_code == 200

    def test_generate_response_has_stable_fields(self, client):
        """POST /generate response includes all stable contract fields."""
        response = client.post("/generate", json={"prompt": "Test prompt"})
        assert response.status_code == 200

        data = response.json()

        # STABLE CONTRACT FIELDS - these must always be present
        assert "response" in data, "response field is part of stable contract"
        assert "phase" in data, "phase field is part of stable contract"
        assert "accepted" in data, "accepted field is part of stable contract"

        # Type validation for stable fields
        assert isinstance(data["response"], str)
        assert isinstance(data["phase"], str)
        assert isinstance(data["accepted"], bool)

    def test_generate_optional_fields_present(self, client):
        """POST /generate response includes optional fields (may be None)."""
        response = client.post("/generate", json={"prompt": "Test"})
        assert response.status_code == 200

        data = response.json()

        # Optional fields should be present (can be None)
        assert "metrics" in data
        assert "safety_flags" in data
        assert "memory_stats" in data

    def test_generate_phase_is_valid_value(self, client):
        """POST /generate phase field is a valid cognitive phase."""
        response = client.post("/generate", json={"prompt": "Test"})
        assert response.status_code == 200

        data = response.json()
        # Phase should be one of the valid values
        assert data["phase"] in ["wake", "sleep", "unknown"]

    def test_generate_with_all_parameters(self, client):
        """POST /generate accepts all documented parameters."""
        response = client.post(
            "/generate",
            json={
                "prompt": "Test with all params",
                "max_tokens": 256,
                "moral_value": 0.7,
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert "response" in data
        assert "phase" in data
        assert "accepted" in data

    def test_generate_empty_prompt_returns_422(self, client):
        """POST /generate with empty prompt returns 422 validation error."""
        response = client.post("/generate", json={"prompt": ""})
        assert response.status_code == 422

        data = response.json()
        # FastAPI/Pydantic validation error format
        assert "detail" in data
        assert isinstance(data["detail"], list)

    def test_generate_whitespace_prompt_returns_400(self, client):
        """POST /generate with whitespace-only prompt returns 400."""
        response = client.post("/generate", json={"prompt": "   "})
        assert response.status_code == 400

        data = response.json()
        # Custom ErrorResponse format
        assert "error" in data
        assert "error_type" in data["error"]
        assert data["error"]["error_type"] == "validation_error"

    def test_generate_invalid_moral_value_returns_422(self, client):
        """POST /generate with invalid moral_value returns 422."""
        response = client.post("/generate", json={"prompt": "Test", "moral_value": 1.5})
        assert response.status_code == 422

    def test_generate_invalid_max_tokens_returns_422(self, client):
        """POST /generate with invalid max_tokens returns 422."""
        response = client.post("/generate", json={"prompt": "Test", "max_tokens": 5000})
        assert response.status_code == 422


class TestInferEndpointContract:
    """Test /infer endpoint contract stability.

    These tests validate that InferResponse schema is stable:
    - response (str): Required
    - accepted (bool): Required
    - phase (str): Required
    - moral_metadata (dict | None): Optional
    - aphasia_metadata (dict | None): Optional
    - rag_metadata (dict | None): Optional
    - timing (dict | None): Optional
    - governance (dict | None): Optional
    """

    def test_infer_returns_200_with_valid_prompt(self, client):
        """POST /infer with valid prompt returns 200."""
        response = client.post("/infer", json={"prompt": "Hello, world!"})
        assert response.status_code == 200

    def test_infer_response_has_stable_fields(self, client):
        """POST /infer response includes all stable contract fields."""
        response = client.post("/infer", json={"prompt": "Test prompt"})
        assert response.status_code == 200

        data = response.json()

        # STABLE CONTRACT FIELDS
        assert "response" in data
        assert "accepted" in data
        assert "phase" in data

        # Type validation
        assert isinstance(data["response"], str)
        assert isinstance(data["accepted"], bool)
        assert isinstance(data["phase"], str)

    def test_infer_optional_fields_present(self, client):
        """POST /infer response includes optional fields."""
        response = client.post("/infer", json={"prompt": "Test"})
        assert response.status_code == 200

        data = response.json()

        # Optional fields should be present
        assert "moral_metadata" in data
        assert "aphasia_metadata" in data
        assert "rag_metadata" in data
        assert "timing" in data
        assert "governance" in data

    def test_infer_secure_mode_affects_moral_value(self, client):
        """POST /infer with secure_mode boosts moral threshold."""
        response = client.post(
            "/infer",
            json={
                "prompt": "Test secure mode",
                "secure_mode": True,
                "moral_value": 0.5,
            },
        )
        assert response.status_code == 200

        data = response.json()
        moral_meta = data.get("moral_metadata", {})
        assert moral_meta.get("secure_mode") is True
        # Moral value should be boosted by 0.2
        assert moral_meta.get("applied_moral_value") == 0.7

    def test_infer_rag_enabled_provides_metadata(self, client):
        """POST /infer with rag_enabled includes RAG metadata."""
        response = client.post(
            "/infer",
            json={"prompt": "Test RAG", "rag_enabled": True, "context_top_k": 5},
        )
        assert response.status_code == 200

        data = response.json()
        rag_meta = data.get("rag_metadata", {})
        assert rag_meta.get("enabled") is True
        assert "context_items_retrieved" in rag_meta

    def test_infer_aphasia_mode_provides_metadata(self, client):
        """POST /infer with aphasia_mode includes aphasia metadata."""
        response = client.post(
            "/infer",
            json={"prompt": "Test aphasia", "aphasia_mode": True},
        )
        assert response.status_code == 200

        data = response.json()
        aphasia_meta = data.get("aphasia_metadata")
        assert aphasia_meta is not None
        assert aphasia_meta.get("enabled") is True


class TestHealthEndpointsContract:
    """Test health endpoint contracts.

    These tests validate health endpoint stability:
    - /health: SimpleHealthStatus
    - /health/liveness: HealthStatus
    - /health/readiness: ReadinessStatus
    - /health/detailed: DetailedHealthStatus
    """

    def test_health_simple_returns_healthy(self, client):
        """GET /health returns simple healthy status."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_liveness_returns_alive(self, client):
        """GET /health/liveness returns alive status with timestamp."""
        response = client.get("/health/liveness")
        assert response.status_code == 200

        data = response.json()
        # STABLE CONTRACT FIELDS
        assert "status" in data
        assert "timestamp" in data

        assert data["status"] == "alive"
        assert isinstance(data["timestamp"], (int, float))

    def test_health_readiness_returns_proper_schema(self, client):
        """GET /health/readiness returns ReadinessStatus schema."""
        response = client.get("/health/readiness")
        # Can be 200 or 503
        assert response.status_code in [200, 503]

        data = response.json()
        # STABLE CONTRACT FIELDS
        assert "ready" in data
        assert "status" in data
        assert "timestamp" in data
        assert "checks" in data

        assert isinstance(data["ready"], bool)
        assert data["status"] in ["ready", "not_ready"]
        assert isinstance(data["checks"], dict)

    def test_health_readiness_includes_required_checks(self, client):
        """GET /health/readiness includes required check keys."""
        response = client.get("/health/readiness")
        assert response.status_code in [200, 503]

        data = response.json()
        checks = data.get("checks", {})

        # Required checks per contract
        assert "memory_manager" in checks
        assert "memory_available" in checks
        assert "cpu_available" in checks

    def test_health_detailed_returns_proper_schema(self, client):
        """GET /health/detailed returns DetailedHealthStatus schema."""
        response = client.get("/health/detailed")
        assert response.status_code in [200, 503]

        data = response.json()
        # STABLE CONTRACT FIELDS
        assert "status" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "system" in data

        assert data["status"] in ["healthy", "unhealthy"]
        assert isinstance(data["timestamp"], (int, float))
        assert isinstance(data["uptime_seconds"], (int, float))
        assert isinstance(data["system"], dict)

    def test_health_metrics_returns_prometheus_format(self, client):
        """GET /health/metrics returns Prometheus text format."""
        response = client.get("/health/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")

        content = response.text
        assert "# HELP" in content
        assert "# TYPE" in content


class TestErrorResponseContract:
    """Test error response format contract.

    All API errors should follow the ErrorResponse schema:
    - error.error_type: str
    - error.message: str
    - error.details: dict | None
    - error.debug_id: str | None (optional request correlation ID)
    """

    def test_400_error_follows_contract(self, client):
        """400 errors follow ErrorResponse schema."""
        response = client.post("/generate", json={"prompt": "   "})
        assert response.status_code == 400

        data = response.json()
        # ErrorResponse structure
        assert "error" in data
        error = data["error"]
        assert "error_type" in error
        assert "message" in error
        assert "details" in error  # Can be None

    def test_error_response_has_debug_id(self, client):
        """Error responses include debug_id for correlation."""
        response = client.post("/generate", json={"prompt": "   "})
        assert response.status_code == 400

        data = response.json()
        error = data["error"]
        # debug_id should be present (can be None if no request ID)
        assert "debug_id" in error

    def test_422_error_follows_fastapi_format(self, client):
        """422 errors follow FastAPI validation format."""
        response = client.post("/generate", json={"prompt": ""})
        assert response.status_code == 422

        data = response.json()
        # FastAPI validation error format
        assert "detail" in data
        assert isinstance(data["detail"], list)

        if len(data["detail"]) > 0:
            error_item = data["detail"][0]
            assert "loc" in error_item
            assert "msg" in error_item
            assert "type" in error_item


class TestResponseHeaders:
    """Test response headers contract."""

    def test_security_headers_present(self, client):
        """Responses include security headers."""
        response = client.get("/health")
        assert response.status_code == 200

        # Security headers from SecurityHeadersMiddleware
        assert "x-content-type-options" in response.headers
        assert "x-frame-options" in response.headers

    def test_request_id_header_present(self, client):
        """Responses include request ID header."""
        response = client.get("/health")
        assert response.status_code == 200

        # Request ID from RequestIDMiddleware
        assert "x-request-id" in response.headers

    def test_request_id_is_uuid_format(self, client):
        """Request ID is in UUID format."""
        response = client.get("/health")
        request_id = response.headers.get("x-request-id", "")

        # UUID format check (8-4-4-4-12)
        parts = request_id.split("-")
        assert len(parts) == 5
        assert len(parts[0]) == 8
        assert len(parts[1]) == 4
        assert len(parts[2]) == 4
        assert len(parts[3]) == 4
        assert len(parts[4]) == 12


class TestStatusEndpointContract:
    """Test /status endpoint contract."""

    def test_status_returns_200(self, client):
        """GET /status returns 200 with expected schema."""
        response = client.get("/status")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "backend" in data
        assert "system" in data
        assert "config" in data

    def test_status_system_info_present(self, client):
        """GET /status includes system information."""
        response = client.get("/status")
        assert response.status_code == 200

        data = response.json()
        system = data.get("system", {})
        assert "memory_mb" in system
        assert "cpu_percent" in system

    def test_status_config_info_present(self, client):
        """GET /status includes configuration information."""
        response = client.get("/status")
        assert response.status_code == 200

        data = response.json()
        config = data.get("config", {})
        assert "dimension" in config
        assert "rate_limiting_enabled" in config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
