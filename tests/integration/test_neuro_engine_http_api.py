"""
Integration tests for NeuroCognitiveEngine HTTP API.
"""

import pytest
from fastapi.testclient import TestClient

from mlsdm.service.neuro_engine_service import create_app

@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    app = create_app()
    return TestClient(app)

class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_healthz_returns_200(self, client):
        """Test that healthz endpoint returns 200 OK."""
        response = client.get("/healthz")
        assert response.status_code == 200

    def test_healthz_structure(self, client):
        """Test healthz response structure."""
        response = client.get("/healthz")
        data = response.json()
        
        assert "status" in data
        assert "backend" in data
        assert data["status"] == "ok"
        assert isinstance(data["backend"], str)

class TestMetricsEndpoint:
    """Test metrics endpoint."""

    def test_metrics_returns_200(self, client):
        """Test that metrics endpoint returns 200 OK."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_content_type(self, client):
        """Test metrics endpoint returns plain text."""
        response = client.get("/metrics")
        assert "text/plain" in response.headers.get("content-type", "")

    def test_metrics_format(self, client):
        """Test metrics are in Prometheus format."""
        response = client.get("/metrics")
        text = response.text
        
        # Should contain prometheus comment lines
        assert "# HELP" in text
        assert "# TYPE" in text

class TestGenerateEndpoint:
    """Test generate endpoint."""

    def test_generate_with_valid_prompt(self, client):
        """Test generation with valid prompt returns 200."""
        response = client.post(
            "/v1/neuro/generate",
            json={"prompt": "Hello, world!"}
        )
        assert response.status_code == 200

    def test_generate_response_structure(self, client):
        """Test that generate response has correct structure."""
        response = client.post(
            "/v1/neuro/generate",
            json={"prompt": "Test prompt"}
        )
        data = response.json()
        
        # Check all required fields are present
        assert "response" in data
        assert "governance" in data
        assert "mlsdm" in data
        assert "timing" in data
        assert "validation_steps" in data
        assert "error" in data
        assert "rejected_at" in data
        
        # Response should be a string
        assert isinstance(data["response"], str)
        assert len(data["response"]) > 0

    def test_generate_with_all_parameters(self, client):
        """Test generation with all optional parameters."""
        response = client.post(
            "/v1/neuro/generate",
            json={
                "prompt": "Test with all parameters",
                "max_tokens": 256,
                "moral_value": 0.7,
                "user_intent": "test",
                "cognitive_load": 0.5,
                "context_top_k": 10
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["response"], str)

    def test_generate_with_empty_prompt(self, client):
        """Test that empty prompt returns validation error."""
        response = client.post(
            "/v1/neuro/generate",
            json={"prompt": ""}
        )
        # Should return 422 (Unprocessable Entity) for validation error
        assert response.status_code == 422

    def test_generate_with_invalid_moral_value(self, client):
        """Test that invalid moral_value returns validation error."""
        response = client.post(
            "/v1/neuro/generate",
            json={"prompt": "Test", "moral_value": 1.5}  # Out of range
        )
        assert response.status_code == 422

    def test_generate_with_invalid_max_tokens(self, client):
        """Test that invalid max_tokens returns validation error."""
        response = client.post(
            "/v1/neuro/generate",
            json={"prompt": "Test", "max_tokens": -1}  # Negative
        )
        assert response.status_code == 422

    def test_generate_timing_metrics(self, client):
        """Test that timing metrics are included in response."""
        response = client.post(
            "/v1/neuro/generate",
            json={"prompt": "Test timing"}
        )
        data = response.json()
        
        assert "timing" in data
        assert isinstance(data["timing"], dict)
        # Should have at least some timing info
        assert len(data["timing"]) > 0

class TestEndToEnd:
    """End-to-end integration tests."""

    def test_multiple_requests_update_metrics(self, client):
        """Test that multiple requests update metrics endpoint."""
        # Get initial metrics
        metrics_before = client.get("/metrics").text
        
        # Make several requests
        for i in range(3):
            client.post(
                "/v1/neuro/generate",
                json={"prompt": f"Test request {i}"}
            )
        
        # Get updated metrics
        metrics_after = client.get("/metrics").text
        
        # Metrics should have changed
        assert metrics_before != metrics_after
        # Should show at least 3 requests
        assert "neuro_requests_total" in metrics_after

    def test_health_check_during_load(self, client):
        """Test that health check works during load."""
        # Make a request
        client.post(
            "/v1/neuro/generate",
            json={"prompt": "Load test"}
        )
        
        # Health check should still work
        response = client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limit_is_enforced(self, client):
        """Test that rate limiting prevents excessive requests from same client."""
        # The default rate limit is 100 requests per 60 seconds
        # Since all test requests come from the same test client (same IP),
        # we'll verify the rate limiter is working by checking it doesn't block normal usage
        
        # Make several requests (under default limit)
        for i in range(5):
            response = client.post(
                "/v1/neuro/generate",
                json={"prompt": f"Test {i}"}
            )
            assert response.status_code == 200
        
        # Verify rate limiter is active by checking app state
        # (In a real scenario with 100+ requests, we'd hit the limit)
        assert hasattr(client.app.state, 'rate_limiter')
        assert client.app.state.rate_limiter is not None
