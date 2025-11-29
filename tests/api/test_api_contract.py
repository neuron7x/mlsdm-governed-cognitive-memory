"""
API Contract Tests for MLSDM.

Tests validate that API endpoints conform to their documented contract
as specified in docs/API_CONTRACT.md. These tests ensure the API is
stable and can be safely used in production environments.

## Contract Stability Testing

These tests specifically verify:
1. Stable contract fields are present and have correct types
2. Error responses follow the ErrorResponse schema
3. Status codes match the documented behavior
4. Response schemas are consistent across calls
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


class TestGenerateResponseContract:
    """Test GenerateResponse schema contract stability."""

    def test_stable_contract_fields_present(self, client):
        """Verify all stable contract fields are present in response."""
        response = client.post("/generate", json={"prompt": "Test prompt"})
        assert response.status_code == 200

        data = response.json()
        # These are STABLE CONTRACT fields - must be present
        assert "response" in data, "Missing stable contract field: response"
        assert "phase" in data, "Missing stable contract field: phase"
        assert "accepted" in data, "Missing stable contract field: accepted"

    def test_stable_contract_field_types(self, client):
        """Verify stable contract fields have correct types."""
        response = client.post("/generate", json={"prompt": "Test prompt"})
        assert response.status_code == 200

        data = response.json()
        # Verify types of stable contract fields
        assert isinstance(data["response"], str), "response must be string"
        assert isinstance(data["phase"], str), "phase must be string"
        assert isinstance(data["accepted"], bool), "accepted must be boolean"

    def test_phase_valid_values(self, client):
        """Verify phase field has expected values."""
        response = client.post("/generate", json={"prompt": "Test prompt"})
        assert response.status_code == 200

        data = response.json()
        # Phase should be one of the expected values
        assert data["phase"] in ["wake", "sleep", "unknown"], \
            f"Unexpected phase value: {data['phase']}"

    def test_optional_fields_have_correct_types_when_present(self, client):
        """Verify optional fields have correct types when present."""
        response = client.post("/generate", json={"prompt": "Test prompt"})
        assert response.status_code == 200

        data = response.json()

        # Check optional fields if present
        if data.get("metrics") is not None:
            assert isinstance(data["metrics"], dict), "metrics must be dict or None"

        if data.get("safety_flags") is not None:
            assert isinstance(data["safety_flags"], dict), "safety_flags must be dict or None"

        if data.get("memory_stats") is not None:
            assert isinstance(data["memory_stats"], dict), "memory_stats must be dict or None"


class TestErrorResponseContract:
    """Test ErrorResponse schema contract stability."""

    def test_error_response_schema_on_400(self, client):
        """Verify ErrorResponse schema on 400 error."""
        response = client.post("/generate", json={"prompt": "   "})  # Whitespace only
        assert response.status_code == 400

        data = response.json()
        # ErrorResponse stable contract
        assert "error" in data, "Missing error field in ErrorResponse"
        error = data["error"]
        assert "error_type" in error, "Missing error_type in ErrorDetail"
        assert "message" in error, "Missing message in ErrorDetail"
        assert "details" in error, "Missing details in ErrorDetail"  # Can be None

    def test_error_response_field_types(self, client):
        """Verify ErrorResponse fields have correct types."""
        response = client.post("/generate", json={"prompt": "   "})
        assert response.status_code == 400

        data = response.json()
        error = data["error"]
        assert isinstance(error["error_type"], str), "error_type must be string"
        assert isinstance(error["message"], str), "message must be string"
        assert error["details"] is None or isinstance(error["details"], dict), \
            "details must be dict or None"

    def test_validation_error_422_format(self, client):
        """Verify 422 validation error follows FastAPI format."""
        response = client.post("/generate", json={"prompt": ""})  # Empty string
        assert response.status_code == 422

        data = response.json()
        # FastAPI/Pydantic validation error format
        assert "detail" in data, "Missing detail field in validation error"
        assert isinstance(data["detail"], list), "detail must be a list"

        if len(data["detail"]) > 0:
            error = data["detail"][0]
            assert "loc" in error, "Missing loc in validation error detail"
            assert "msg" in error, "Missing msg in validation error detail"
            assert "type" in error, "Missing type in validation error detail"


class TestHealthEndpointsContract:
    """Test health endpoint contract stability."""

    def test_health_simple_contract(self, client):
        """Verify /health returns SimpleHealthStatus schema."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        # SimpleHealthStatus stable contract
        assert "status" in data, "Missing status field"
        assert isinstance(data["status"], str), "status must be string"
        assert data["status"] == "healthy"

    def test_health_liveness_contract(self, client):
        """Verify /health/liveness returns HealthStatus schema."""
        response = client.get("/health/liveness")
        assert response.status_code == 200

        data = response.json()
        # HealthStatus stable contract
        assert "status" in data, "Missing status field"
        assert "timestamp" in data, "Missing timestamp field"
        assert isinstance(data["status"], str), "status must be string"
        assert isinstance(data["timestamp"], (int, float)), "timestamp must be number"
        assert data["status"] == "alive"
        assert data["timestamp"] > 0

    def test_health_readiness_contract(self, client):
        """Verify /health/readiness returns ReadinessStatus schema."""
        response = client.get("/health/readiness")
        # Can be 200 or 503 depending on system state
        assert response.status_code in [200, 503]

        data = response.json()
        # ReadinessStatus stable contract
        assert "ready" in data, "Missing ready field"
        assert "status" in data, "Missing status field"
        assert "timestamp" in data, "Missing timestamp field"
        assert "checks" in data, "Missing checks field"

        # Type checks
        assert isinstance(data["ready"], bool), "ready must be boolean"
        assert isinstance(data["status"], str), "status must be string"
        assert isinstance(data["timestamp"], (int, float)), "timestamp must be number"
        assert isinstance(data["checks"], dict), "checks must be dict"

        # Status values
        assert data["status"] in ["ready", "not_ready"]

    def test_ready_alias_endpoint(self, client):
        """Verify /ready alias returns same schema as /health/readiness."""
        response = client.get("/ready")
        # Can be 200 or 503 depending on system state
        assert response.status_code in [200, 503]

        data = response.json()
        # Same schema as /health/readiness
        assert "ready" in data
        assert "status" in data
        assert "timestamp" in data
        assert "checks" in data


class TestInferEndpointContract:
    """Test /infer endpoint contract stability."""

    def test_infer_response_contract(self, client):
        """Verify /infer returns InferResponse schema."""
        response = client.post("/infer", json={"prompt": "Test prompt"})
        assert response.status_code == 200

        data = response.json()
        # InferResponse required fields
        assert "response" in data, "Missing response field"
        assert "accepted" in data, "Missing accepted field"
        assert "phase" in data, "Missing phase field"

        # Type checks
        assert isinstance(data["response"], str)
        assert isinstance(data["accepted"], bool)
        assert isinstance(data["phase"], str)

    def test_infer_metadata_fields_present(self, client):
        """Verify /infer returns expected metadata fields."""
        response = client.post(
            "/infer",
            json={"prompt": "Test prompt", "secure_mode": True, "rag_enabled": True}
        )
        assert response.status_code == 200

        data = response.json()
        # Metadata fields (can be None)
        assert "moral_metadata" in data
        assert "aphasia_metadata" in data
        assert "rag_metadata" in data
        assert "timing" in data
        assert "governance" in data


class TestStatusEndpointContract:
    """Test /status endpoint contract stability."""

    def test_status_response_contract(self, client):
        """Verify /status returns expected schema."""
        response = client.get("/status")
        assert response.status_code == 200

        data = response.json()
        # Required fields
        assert "status" in data
        assert "version" in data
        assert "backend" in data
        assert "system" in data
        assert "config" in data

        # Type checks
        assert isinstance(data["status"], str)
        assert data["status"] == "ok"
        assert isinstance(data["version"], str)
        assert isinstance(data["backend"], str)
        assert isinstance(data["system"], dict)
        assert isinstance(data["config"], dict)


class TestResponseHeaders:
    """Test response headers for security and tracking."""

    def test_security_headers_present(self, client):
        """Verify security headers are included in responses."""
        response = client.get("/health")

        # Security headers from middleware
        assert "x-content-type-options" in response.headers
        assert "x-frame-options" in response.headers

    def test_request_id_header_present(self, client):
        """Verify request ID header is included in responses."""
        response = client.get("/health")

        # Request ID from middleware
        assert "x-request-id" in response.headers


class TestContractConsistency:
    """Test that responses are consistent across multiple calls."""

    def test_generate_response_consistency(self, client):
        """Verify generate responses have consistent structure."""
        # Make multiple calls
        responses = [
            client.post("/generate", json={"prompt": f"Test {i}"})
            for i in range(3)
        ]

        # All should succeed
        for response in responses:
            assert response.status_code == 200

        # All should have the same structure
        keys_set = set()
        for response in responses:
            data = response.json()
            keys_set.add(frozenset(data.keys()))

        # All responses should have the same keys
        assert len(keys_set) == 1, "Response structure is inconsistent"

    def test_health_endpoints_consistency(self, client):
        """Verify health endpoints return consistent structure."""
        # Make multiple calls
        for _ in range(3):
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert data["status"] == "healthy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
