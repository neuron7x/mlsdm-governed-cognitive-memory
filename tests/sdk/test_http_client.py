"""
SDK HTTP Client Tests for MLSDMHttpClient.

Tests cover:
- HTTP request/response handling
- Error handling (4xx, 5xx, timeouts)
- Response DTO parsing
- Connection error handling

These tests mock the HTTP layer to verify SDK behavior without
requiring a running server.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from mlsdm.api.schemas import GenerateResponseDTO
from mlsdm.sdk import (
    MLSDMClientError,
    MLSDMConnectionError,
    MLSDMHttpClient,
    MLSDMServerError,
    MLSDMTimeoutError,
)


class TestMLSDMHttpClientGenerate:
    """Test HTTP client generate method."""

    def test_generate_success_returns_dto(self):
        """Test that successful generate returns GenerateResponseDTO."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Test response",
            "phase": "wake",
            "accepted": True,
            "metrics": None,
            "safety_flags": None,
            "memory_stats": None,
        }

        with patch.object(requests.Session, "post", return_value=mock_response):
            client = MLSDMHttpClient(base_url="http://localhost:8000")
            result = client.generate("Test prompt")

            assert isinstance(result, GenerateResponseDTO)
            assert result.response == "Test response"
            assert result.phase == "wake"
            assert result.accepted is True

    def test_generate_passes_parameters(self):
        """Test that generate passes all parameters correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Test",
            "phase": "wake",
            "accepted": True,
        }

        with patch.object(requests.Session, "post", return_value=mock_response) as mock_post:
            client = MLSDMHttpClient(base_url="http://localhost:8000")
            client.generate(
                prompt="Test prompt",
                moral_value=0.8,
                max_tokens=256,
            )

            # Verify the request body
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args[1]
            body = call_kwargs["json"]
            assert body["prompt"] == "Test prompt"
            assert body["moral_value"] == 0.8
            assert body["max_tokens"] == 256


class TestMLSDMHttpClientErrors:
    """Test HTTP client error handling."""

    def test_400_error_raises_client_error(self):
        """Test that 400 errors raise MLSDMClientError."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {
                "error_type": "validation_error",
                "message": "Prompt cannot be empty",
                "details": {"field": "prompt"},
            }
        }

        with patch.object(requests.Session, "post", return_value=mock_response):
            client = MLSDMHttpClient(base_url="http://localhost:8000")

            with pytest.raises(MLSDMClientError) as exc_info:
                client.generate("   ")

            assert exc_info.value.status_code == 400
            assert exc_info.value.error_type == "validation_error"
            assert "Prompt cannot be empty" in exc_info.value.message

    def test_422_validation_error_raises_client_error(self):
        """Test that 422 validation errors raise MLSDMClientError."""
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.json.return_value = {
            "detail": [
                {
                    "loc": ["body", "prompt"],
                    "msg": "String should have at least 1 character",
                    "type": "string_too_short",
                }
            ]
        }

        with patch.object(requests.Session, "post", return_value=mock_response):
            client = MLSDMHttpClient(base_url="http://localhost:8000")

            with pytest.raises(MLSDMClientError) as exc_info:
                client.generate("")

            assert exc_info.value.status_code == 422
            assert exc_info.value.error_type == "validation_error"

    def test_429_rate_limit_raises_client_error(self):
        """Test that 429 errors raise MLSDMClientError."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {
            "error": {
                "error_type": "rate_limit_exceeded",
                "message": "Rate limit exceeded",
                "details": None,
            }
        }

        with patch.object(requests.Session, "post", return_value=mock_response):
            client = MLSDMHttpClient(base_url="http://localhost:8000")

            with pytest.raises(MLSDMClientError) as exc_info:
                client.generate("Test")

            assert exc_info.value.status_code == 429
            assert exc_info.value.error_type == "rate_limit_exceeded"

    def test_500_error_raises_server_error(self):
        """Test that 500 errors raise MLSDMServerError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {
            "error": {
                "error_type": "internal_error",
                "message": "Internal server error",
                "details": None,
            }
        }

        with patch.object(requests.Session, "post", return_value=mock_response):
            client = MLSDMHttpClient(base_url="http://localhost:8000")

            with pytest.raises(MLSDMServerError) as exc_info:
                client.generate("Test")

            assert exc_info.value.status_code == 500
            assert exc_info.value.error_type == "internal_error"

    def test_503_service_unavailable_raises_server_error(self):
        """Test that 503 errors raise MLSDMServerError."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.json.return_value = {
            "error": {
                "error_type": "service_unavailable",
                "message": "Service unavailable",
                "details": None,
            }
        }

        with patch.object(requests.Session, "post", return_value=mock_response):
            client = MLSDMHttpClient(base_url="http://localhost:8000")

            with pytest.raises(MLSDMServerError) as exc_info:
                client.generate("Test")

            assert exc_info.value.status_code == 503


class TestMLSDMHttpClientTimeout:
    """Test HTTP client timeout handling."""

    def test_timeout_raises_timeout_error(self):
        """Test that timeouts raise MLSDMTimeoutError."""
        with patch.object(
            requests.Session, "post",
            side_effect=requests.Timeout("Connection timed out")
        ):
            client = MLSDMHttpClient(base_url="http://localhost:8000", timeout=5.0)

            with pytest.raises(MLSDMTimeoutError) as exc_info:
                client.generate("Test")

            assert exc_info.value.timeout_seconds == 5.0
            assert "timed out" in exc_info.value.message


class TestMLSDMHttpClientConnection:
    """Test HTTP client connection handling."""

    def test_connection_error_raises_connection_error(self):
        """Test that connection errors raise MLSDMConnectionError."""
        with patch.object(
            requests.Session, "post",
            side_effect=requests.ConnectionError("Connection refused")
        ):
            client = MLSDMHttpClient(base_url="http://localhost:8000")

            with pytest.raises(MLSDMConnectionError) as exc_info:
                client.generate("Test")

            assert "localhost:8000" in exc_info.value.url


class TestMLSDMHttpClientHealth:
    """Test HTTP client health endpoints."""

    def test_health_endpoint(self):
        """Test health endpoint returns correct data."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}

        with patch.object(requests.Session, "get", return_value=mock_response):
            client = MLSDMHttpClient(base_url="http://localhost:8000")
            result = client.health()

            assert result["status"] == "healthy"

    def test_ready_endpoint(self):
        """Test ready endpoint returns correct data."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ready": True,
            "status": "ready",
            "timestamp": 1234567890.0,
            "checks": {"memory_manager": True},
        }

        with patch.object(requests.Session, "get", return_value=mock_response):
            client = MLSDMHttpClient(base_url="http://localhost:8000")
            result = client.ready()

            assert result["ready"] is True
            assert result["status"] == "ready"

    def test_readiness_endpoint(self):
        """Test readiness endpoint returns correct data."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ready": True,
            "status": "ready",
            "timestamp": 1234567890.0,
            "checks": {"memory_manager": True},
        }

        with patch.object(requests.Session, "get", return_value=mock_response):
            client = MLSDMHttpClient(base_url="http://localhost:8000")
            result = client.readiness()

            assert result["ready"] is True


class TestMLSDMHttpClientContextManager:
    """Test HTTP client context manager."""

    def test_context_manager_closes_session(self):
        """Test that context manager closes session."""
        with patch.object(requests.Session, "close") as mock_close:
            with MLSDMHttpClient(base_url="http://localhost:8000") as client:
                assert client is not None

            mock_close.assert_called_once()


class TestGenerateResponseDTO:
    """Test GenerateResponseDTO parsing."""

    def test_from_api_response_full(self):
        """Test DTO parsing with all fields."""
        data = {
            "response": "Test response",
            "phase": "wake",
            "accepted": True,
            "metrics": {"timing": {"total": 100}},
            "safety_flags": {"validation_steps": []},
            "memory_stats": {"step": 1},
            "moral_score": 0.8,
            "aphasia_flags": None,
            "emergency_shutdown": False,
            "latency_ms": 100.5,
            "cognitive_state": {"phase": "wake"},
        }

        dto = GenerateResponseDTO.from_api_response(data)

        assert dto.response == "Test response"
        assert dto.phase == "wake"
        assert dto.accepted is True
        assert dto.metrics == {"timing": {"total": 100}}
        assert dto.safety_flags == {"validation_steps": []}
        assert dto.memory_stats == {"step": 1}
        assert dto.moral_score == 0.8
        assert dto.aphasia_flags is None
        assert dto.emergency_shutdown is False
        assert dto.latency_ms == 100.5
        assert dto.cognitive_state == {"phase": "wake"}

    def test_from_api_response_minimal(self):
        """Test DTO parsing with minimal fields."""
        data = {
            "response": "Test",
            "phase": "sleep",
            "accepted": False,
        }

        dto = GenerateResponseDTO.from_api_response(data)

        assert dto.response == "Test"
        assert dto.phase == "sleep"
        assert dto.accepted is False
        assert dto.metrics is None
        assert dto.safety_flags is None

    def test_from_api_response_missing_fields_uses_defaults(self):
        """Test DTO parsing handles missing fields gracefully."""
        data = {}  # Empty response

        dto = GenerateResponseDTO.from_api_response(data)

        assert dto.response == ""
        assert dto.phase == "unknown"
        assert dto.accepted is False


class TestMLSDMExceptionStringRepresentation:
    """Test exception string representations."""

    def test_client_error_str(self):
        """Test MLSDMClientError string representation."""
        error = MLSDMClientError(
            message="Validation failed",
            status_code=400,
            error_type="validation_error",
            debug_id="abc123",
        )

        str_repr = str(error)
        assert "400" in str_repr
        assert "validation_error" in str_repr
        assert "Validation failed" in str_repr
        assert "abc123" in str_repr

    def test_server_error_str(self):
        """Test MLSDMServerError string representation."""
        error = MLSDMServerError(
            message="Internal error",
            status_code=500,
            error_type="internal_error",
        )

        str_repr = str(error)
        assert "500" in str_repr
        assert "Internal error" in str_repr

    def test_timeout_error_str(self):
        """Test MLSDMTimeoutError string representation."""
        error = MLSDMTimeoutError(
            message="Request timed out",
            timeout_seconds=30.0,
        )

        str_repr = str(error)
        assert "timed out" in str_repr
        assert "30.0" in str_repr

    def test_connection_error_str(self):
        """Test MLSDMConnectionError string representation."""
        error = MLSDMConnectionError(
            message="Failed to connect",
            url="http://localhost:8000",
        )

        str_repr = str(error)
        assert "Failed to connect" in str_repr
        assert "localhost:8000" in str_repr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
