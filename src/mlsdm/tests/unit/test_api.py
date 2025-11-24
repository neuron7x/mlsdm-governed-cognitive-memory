"""Tests for the FastAPI application."""
import os
import tempfile
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def test_config_file():
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("dimension: 10\n")
        f.write("moral_filter:\n")
        f.write("  threshold: 0.5\n")
        f.write("cognitive_rhythm:\n")
        f.write("  wake_duration: 5\n")
        f.write("  sleep_duration: 2\n")
        f.write("ontology_matcher:\n")
        f.write("  ontology_vectors:\n")
        f.write("    - [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n")
        f.write("    - [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]\n")
        config_path = f.name

    yield config_path

    if os.path.exists(config_path):
        os.unlink(config_path)


@pytest.fixture
def client(test_config_file):
    """Create a test client with mocked config."""
    with patch.dict(os.environ, {"CONFIG_PATH": test_config_file, "DISABLE_RATE_LIMIT": "1"}):
        # Import app after setting env variable
        from mlsdm.api.app import app
        with TestClient(app) as client:
            yield client


class TestAPIEndpoints:
    """Test suite for FastAPI endpoints."""

    def test_health_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_get_state_without_auth(self, client):
        """Test that state endpoint requires authentication."""
        response = client.get("/v1/state/")
        assert response.status_code == 401

    def test_get_state_with_invalid_auth(self, client):
        """Test state endpoint with invalid token."""
        with patch.dict(os.environ, {"API_KEY": "test_key"}):
            response = client.get(
                "/v1/state/",
                headers={"Authorization": "Bearer wrong_key"}
            )
            assert response.status_code == 401

    def test_get_state_with_valid_auth(self, client):
        """Test state endpoint with valid authentication."""
        with patch.dict(os.environ, {"API_KEY": "test_key"}):
            response = client.get(
                "/v1/state/",
                headers={"Authorization": "Bearer test_key"}
            )
            assert response.status_code == 200
            data = response.json()
            assert "L1_norm" in data
            assert "L2_norm" in data
            assert "L3_norm" in data
            assert "current_phase" in data
            assert "moral_filter_threshold" in data

    def test_process_event_without_auth(self, client):
        """Test that process_event requires authentication."""
        event_data = {
            "event_vector": [1.0] * 10,
            "moral_value": 0.8
        }
        response = client.post("/v1/process_event/", json=event_data)
        assert response.status_code == 401

    def test_process_event_with_valid_auth(self, client):
        """Test processing an event with valid authentication."""
        with patch.dict(os.environ, {"API_KEY": "test_key"}):
            event_data = {
                "event_vector": [1.0] * 10,
                "moral_value": 0.8
            }
            response = client.post(
                "/v1/process_event/",
                json=event_data,
                headers={"Authorization": "Bearer test_key"}
            )
            assert response.status_code == 200
            data = response.json()
            assert "L1_norm" in data
            assert "moral_filter_threshold" in data

    def test_process_event_dimension_mismatch(self, client):
        """Test that dimension mismatch returns 400."""
        with patch.dict(os.environ, {"API_KEY": "test_key"}):
            event_data = {
                "event_vector": [1.0] * 5,  # Wrong dimension
                "moral_value": 0.8
            }
            response = client.post(
                "/v1/process_event/",
                json=event_data,
                headers={"Authorization": "Bearer test_key"}
            )
            assert response.status_code == 400
            # Updated to match new validator error message
            assert "dimension" in response.json()["detail"].lower()

    def test_process_event_invalid_moral_value(self, client):
        """Test processing event with various moral values."""
        with patch.dict(os.environ, {"API_KEY": "test_key"}):
            # Test with low moral value
            event_data = {
                "event_vector": [1.0] * 10,
                "moral_value": 0.1
            }
            response = client.post(
                "/v1/process_event/",
                json=event_data,
                headers={"Authorization": "Bearer test_key"}
            )
            assert response.status_code == 200

    def test_process_multiple_events(self, client):
        """Test processing multiple events in sequence."""
        with patch.dict(os.environ, {"API_KEY": "test_key"}):
            for i in range(5):
                event_data = {
                    "event_vector": [float(i)] * 10,
                    "moral_value": 0.6 + i * 0.05
                }
                response = client.post(
                    "/v1/process_event/",
                    json=event_data,
                    headers={"Authorization": "Bearer test_key"}
                )
                assert response.status_code == 200

    def test_state_tracking_after_events(self, client):
        """Test that state is updated after processing events."""
        with patch.dict(os.environ, {"API_KEY": "test_key"}):
            # Get initial state
            response1 = client.get(
                "/v1/state/",
                headers={"Authorization": "Bearer test_key"}
            )
            initial_count = response1.json()["total_events_processed"]

            # Process an event
            event_data = {
                "event_vector": [1.0] * 10,
                "moral_value": 0.8
            }
            client.post(
                "/v1/process_event/",
                json=event_data,
                headers={"Authorization": "Bearer test_key"}
            )

            # Check state updated
            response2 = client.get(
                "/v1/state/",
                headers={"Authorization": "Bearer test_key"}
            )
            final_count = response2.json()["total_events_processed"]
            assert final_count > initial_count

    def test_api_without_env_api_key(self, client):
        """Test API behavior when API_KEY is not set."""
        # Without API_KEY in environment, any token should work
        response = client.get(
            "/v1/state/",
            headers={"Authorization": "Bearer any_token"}
        )
        assert response.status_code == 200

    def test_event_input_validation(self, client):
        """Test input validation for event data."""
        with patch.dict(os.environ, {"API_KEY": "test_key"}):
            # Missing field
            invalid_data = {
                "event_vector": [1.0] * 10
                # Missing moral_value
            }
            response = client.post(
                "/v1/process_event/",
                json=invalid_data,
                headers={"Authorization": "Bearer test_key"}
            )
            assert response.status_code == 422

    def test_state_response_fields(self, client):
        """Test that state response has all required fields."""
        with patch.dict(os.environ, {"API_KEY": "test_key"}):
            response = client.get(
                "/v1/state/",
                headers={"Authorization": "Bearer test_key"}
            )
            data = response.json()

            required_fields = [
                "L1_norm", "L2_norm", "L3_norm",
                "current_phase", "latent_events_count",
                "accepted_events_count", "total_events_processed",
                "moral_filter_threshold"
            ]

            for field in required_fields:
                assert field in data

    def test_concurrent_requests(self, client):
        """Test handling concurrent requests."""
        with patch.dict(os.environ, {"API_KEY": "test_key"}):
            import concurrent.futures

            def make_request():
                event_data = {
                    "event_vector": [1.0] * 10,
                    "moral_value": 0.7
                }
                return client.post(
                    "/v1/process_event/",
                    json=event_data,
                    headers={"Authorization": "Bearer test_key"}
                )

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_request) for _ in range(10)]
                results = [f.result() for f in futures]

            # All requests should succeed
            assert all(r.status_code == 200 for r in results)

    def test_numeric_values_in_response(self, client):
        """Test that numeric values in response are valid."""
        with patch.dict(os.environ, {"API_KEY": "test_key"}):
            response = client.get(
                "/v1/state/",
                headers={"Authorization": "Bearer test_key"}
            )
            data = response.json()

            # Check numeric fields are valid
            assert isinstance(data["L1_norm"], (int, float))
            assert isinstance(data["L2_norm"], (int, float))
            assert isinstance(data["L3_norm"], (int, float))
            assert isinstance(data["moral_filter_threshold"], (int, float))

            assert data["L1_norm"] >= 0
            assert data["L2_norm"] >= 0
            assert data["L3_norm"] >= 0
