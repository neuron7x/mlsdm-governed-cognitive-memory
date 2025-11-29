"""
HTTP SDK Client Tests for NeuroEngineHTTPClient.

Tests cover:
- HTTP request behavior (mocked)
- Error handling for 4xx/5xx responses
- Proper exception types for different errors
- Timeout handling
- Response parsing and DTO creation

Note: These tests mock the HTTP layer to test SDK behavior
without requiring a running API server.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from mlsdm.api.schemas import GenerateResponseDTO
from mlsdm.sdk import (
    MLSDMAuthenticationError,
    MLSDMClientError,
    MLSDMConnectionError,
    MLSDMRateLimitError,
    MLSDMServerError,
    MLSDMTimeoutError,
    MLSDMValidationError,
    NeuroEngineHTTPClient,
)


class TestNeuroEngineHTTPClientInit:
    """Test SDK client initialization."""

    def test_default_initialization(self):
        """Client initializes with default values."""
        client = NeuroEngineHTTPClient()

        assert client.base_url == "http://localhost:8000"
        assert client.timeout == 30.0

    def test_custom_base_url(self):
        """Client accepts custom base URL."""
        client = NeuroEngineHTTPClient(base_url="http://custom:9000")

        assert client.base_url == "http://custom:9000"

    def test_base_url_trailing_slash_stripped(self):
        """Base URL trailing slash is stripped."""
        client = NeuroEngineHTTPClient(base_url="http://custom:9000/")

        assert client.base_url == "http://custom:9000"

    def test_custom_timeout(self):
        """Client accepts custom timeout."""
        client = NeuroEngineHTTPClient(timeout=60.0)

        assert client.timeout == 60.0

    def test_api_key_sets_auth_header(self):
        """API key sets Authorization header."""
        client = NeuroEngineHTTPClient(api_key="test-key")

        assert client._session.headers["Authorization"] == "Bearer test-key"


class TestNeuroEngineHTTPClientGenerate:
    """Test SDK client generate method."""

    def test_generate_returns_dto(self):
        """Generate returns GenerateResponseDTO."""
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

        with patch.object(requests.Session, "request", return_value=mock_response):
            client = NeuroEngineHTTPClient()
            result = client.generate("Test prompt")

            assert isinstance(result, GenerateResponseDTO)
            assert result.response == "Test response"
            assert result.phase == "wake"
            assert result.accepted is True

    def test_generate_passes_parameters(self):
        """Generate passes all parameters to request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Test",
            "phase": "wake",
            "accepted": True,
        }

        with patch.object(requests.Session, "request", return_value=mock_response) as mock_request:
            client = NeuroEngineHTTPClient()
            client.generate(
                prompt="Test",
                max_tokens=256,
                moral_value=0.8
            )

            # Verify request was made with correct parameters
            mock_request.assert_called_once()
            call_args = mock_request.call_args
            assert call_args[1]["json"]["prompt"] == "Test"
            assert call_args[1]["json"]["max_tokens"] == 256
            assert call_args[1]["json"]["moral_value"] == 0.8

    def test_generate_minimal_parameters(self):
        """Generate works with only prompt."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Test",
            "phase": "wake",
            "accepted": True,
        }

        with patch.object(requests.Session, "request", return_value=mock_response) as mock_request:
            client = NeuroEngineHTTPClient()
            client.generate("Test")

            call_args = mock_request.call_args
            assert call_args[1]["json"] == {"prompt": "Test"}


class TestNeuroEngineHTTPClient4xxErrors:
    """Test SDK client 4xx error handling."""

    def test_400_raises_client_error(self):
        """400 error raises MLSDMClientError."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.headers = {"x-request-id": "test-123"}
        mock_response.json.return_value = {
            "error": {
                "error_type": "validation_error",
                "message": "Prompt cannot be empty",
                "details": {"field": "prompt"},
            }
        }

        with patch.object(requests.Session, "request", return_value=mock_response):
            client = NeuroEngineHTTPClient()

            with pytest.raises(MLSDMClientError) as exc_info:
                client.generate("   ")

            assert exc_info.value.status_code == 400
            assert exc_info.value.error_type == "validation_error"
            assert "Prompt cannot be empty" in exc_info.value.message
            assert exc_info.value.debug_id == "test-123"

    def test_401_raises_authentication_error(self):
        """401 error raises MLSDMAuthenticationError."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}
        mock_response.json.return_value = {
            "detail": "Invalid authentication"
        }

        with patch.object(requests.Session, "request", return_value=mock_response):
            client = NeuroEngineHTTPClient()

            with pytest.raises(MLSDMAuthenticationError) as exc_info:
                client.generate("Test")

            assert exc_info.value.status_code == 401

    def test_422_raises_validation_error(self):
        """422 error raises MLSDMValidationError."""
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.headers = {"x-request-id": "req-456"}
        mock_response.json.return_value = {
            "detail": [
                {
                    "loc": ["body", "prompt"],
                    "msg": "String should have at least 1 character",
                    "type": "string_too_short",
                }
            ]
        }

        with patch.object(requests.Session, "request", return_value=mock_response):
            client = NeuroEngineHTTPClient()

            with pytest.raises(MLSDMValidationError) as exc_info:
                client.generate("")

            assert exc_info.value.status_code == 422
            assert len(exc_info.value.validation_errors) == 1
            assert exc_info.value.debug_id == "req-456"

    def test_429_raises_rate_limit_error(self):
        """429 error raises MLSDMRateLimitError."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "1.0", "x-request-id": "req-789"}
        mock_response.json.return_value = {
            "error": {
                "error_type": "rate_limit_exceeded",
                "message": "Rate limit exceeded. Maximum 5 requests per second.",
                "details": None,
            }
        }

        with patch.object(requests.Session, "request", return_value=mock_response):
            client = NeuroEngineHTTPClient()

            with pytest.raises(MLSDMRateLimitError) as exc_info:
                client.generate("Test")

            assert exc_info.value.status_code == 429
            assert exc_info.value.retry_after == 1.0
            assert "rate limit" in exc_info.value.message.lower()


class TestNeuroEngineHTTPClient5xxErrors:
    """Test SDK client 5xx error handling."""

    def test_500_raises_server_error(self):
        """500 error raises MLSDMServerError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {"x-request-id": "err-500"}
        mock_response.json.return_value = {
            "error": {
                "error_type": "internal_error",
                "message": "An internal error occurred. Please try again later.",
                "details": None,
            }
        }

        with patch.object(requests.Session, "request", return_value=mock_response):
            client = NeuroEngineHTTPClient()

            with pytest.raises(MLSDMServerError) as exc_info:
                client.generate("Test")

            assert exc_info.value.status_code == 500
            assert exc_info.value.error_type == "internal_error"
            assert exc_info.value.debug_id == "err-500"

    def test_503_raises_server_error(self):
        """503 error raises MLSDMServerError."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.headers = {}
        mock_response.json.return_value = {}

        with patch.object(requests.Session, "request", return_value=mock_response):
            client = NeuroEngineHTTPClient()

            with pytest.raises(MLSDMServerError) as exc_info:
                client.generate("Test")

            assert exc_info.value.status_code == 503


class TestNeuroEngineHTTPClientTimeoutErrors:
    """Test SDK client timeout handling."""

    def test_timeout_raises_timeout_error(self):
        """Timeout raises MLSDMTimeoutError."""
        with patch.object(
            requests.Session,
            "request",
            side_effect=requests.exceptions.Timeout("Request timed out")
        ):
            client = NeuroEngineHTTPClient(timeout=5.0)

            with pytest.raises(MLSDMTimeoutError) as exc_info:
                client.generate("Test")

            assert exc_info.value.timeout_seconds == 5.0

    def test_timeout_error_message(self):
        """MLSDMTimeoutError has descriptive message."""
        error = MLSDMTimeoutError(timeout_seconds=30.0)
        assert "30.0" in str(error)


class TestNeuroEngineHTTPClientConnectionErrors:
    """Test SDK client connection error handling."""

    def test_connection_error_raises_connection_error(self):
        """Connection error raises MLSDMConnectionError."""
        with patch.object(
            requests.Session,
            "request",
            side_effect=requests.exceptions.ConnectionError("Failed to connect")
        ):
            client = NeuroEngineHTTPClient(base_url="http://nonexistent:9999")

            with pytest.raises(MLSDMConnectionError) as exc_info:
                client.generate("Test")

            assert exc_info.value.url is not None
            assert "nonexistent" in exc_info.value.url


class TestNeuroEngineHTTPClientHealthEndpoints:
    """Test SDK client health endpoint methods."""

    def test_health_returns_dict(self):
        """Health method returns dictionary."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}

        with patch.object(requests.Session, "request", return_value=mock_response):
            client = NeuroEngineHTTPClient()
            result = client.health()

            assert result == {"status": "healthy"}

    def test_health_liveness_returns_dict(self):
        """Health liveness method returns dictionary."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "alive", "timestamp": 1234567890.0}

        with patch.object(requests.Session, "request", return_value=mock_response):
            client = NeuroEngineHTTPClient()
            result = client.health_liveness()

            assert result["status"] == "alive"
            assert "timestamp" in result

    def test_health_readiness_returns_dict(self):
        """Health readiness method returns dictionary."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ready": True,
            "status": "ready",
            "timestamp": 1234567890.0,
            "checks": {"memory_manager": True},
        }

        with patch.object(requests.Session, "request", return_value=mock_response):
            client = NeuroEngineHTTPClient()
            result = client.health_readiness()

            assert result["ready"] is True
            assert "checks" in result

    def test_health_detailed_returns_dict(self):
        """Health detailed method returns dictionary."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "healthy",
            "timestamp": 1234567890.0,
            "uptime_seconds": 3600.0,
            "system": {"memory_percent": 50.0},
        }

        with patch.object(requests.Session, "request", return_value=mock_response):
            client = NeuroEngineHTTPClient()
            result = client.health_detailed()

            assert result["status"] == "healthy"
            assert "system" in result


class TestNeuroEngineHTTPClientInfer:
    """Test SDK client infer method."""

    def test_infer_returns_dict(self):
        """Infer method returns dictionary."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Test response",
            "accepted": True,
            "phase": "wake",
            "moral_metadata": {"secure_mode": True},
        }

        with patch.object(requests.Session, "request", return_value=mock_response):
            client = NeuroEngineHTTPClient()
            result = client.infer("Test", secure_mode=True)

            assert result["response"] == "Test response"
            assert result["moral_metadata"]["secure_mode"] is True

    def test_infer_passes_all_parameters(self):
        """Infer passes all parameters to request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Test",
            "accepted": True,
            "phase": "wake",
        }

        with patch.object(requests.Session, "request", return_value=mock_response) as mock_request:
            client = NeuroEngineHTTPClient()
            client.infer(
                prompt="Test",
                moral_value=0.7,
                max_tokens=256,
                secure_mode=True,
                aphasia_mode=True,
                rag_enabled=False,
                context_top_k=10,
                user_intent="analytical",
            )

            call_args = mock_request.call_args
            body = call_args[1]["json"]

            assert body["prompt"] == "Test"
            assert body["moral_value"] == 0.7
            assert body["max_tokens"] == 256
            assert body["secure_mode"] is True
            assert body["aphasia_mode"] is True
            assert body["rag_enabled"] is False
            assert body["context_top_k"] == 10
            assert body["user_intent"] == "analytical"


class TestNeuroEngineHTTPClientContextManager:
    """Test SDK client context manager behavior."""

    def test_context_manager_closes_session(self):
        """Context manager closes session on exit."""
        with patch.object(requests.Session, "close") as mock_close:
            with NeuroEngineHTTPClient():
                pass

            mock_close.assert_called_once()

    def test_context_manager_returns_client(self):
        """Context manager returns client instance."""
        with NeuroEngineHTTPClient() as client:
            assert isinstance(client, NeuroEngineHTTPClient)


class TestGenerateResponseDTO:
    """Test GenerateResponseDTO class."""

    def test_from_dict_required_fields(self):
        """DTO parses required fields correctly."""
        data = {
            "response": "Test response",
            "phase": "wake",
            "accepted": True,
        }

        dto = GenerateResponseDTO.from_dict(data)

        assert dto.response == "Test response"
        assert dto.phase == "wake"
        assert dto.accepted is True

    def test_from_dict_optional_fields(self):
        """DTO parses optional fields correctly."""
        data = {
            "response": "Test",
            "phase": "sleep",
            "accepted": False,
            "moral_score": 0.85,
            "latency_ms": 150.5,
            "emergency_shutdown": False,
            "metrics": {"timing": {"total": 150}},
        }

        dto = GenerateResponseDTO.from_dict(data)

        assert dto.moral_score == 0.85
        assert dto.latency_ms == 150.5
        assert dto.emergency_shutdown is False
        assert dto.metrics == {"timing": {"total": 150}}

    def test_from_dict_missing_required_raises(self):
        """DTO raises error for missing required fields."""
        data = {"response": "Test"}  # Missing phase and accepted

        with pytest.raises(KeyError):
            GenerateResponseDTO.from_dict(data)

    def test_to_dict(self):
        """DTO converts back to dictionary correctly."""
        dto = GenerateResponseDTO(
            response="Test",
            phase="wake",
            accepted=True,
            moral_score=0.9,
        )

        result = dto.to_dict()

        assert result["response"] == "Test"
        assert result["phase"] == "wake"
        assert result["accepted"] is True
        assert result["moral_score"] == 0.9

    def test_repr(self):
        """DTO has readable repr."""
        dto = GenerateResponseDTO(
            response="Short response",
            phase="wake",
            accepted=True,
        )

        repr_str = repr(dto)
        assert "GenerateResponseDTO" in repr_str
        assert "wake" in repr_str


class TestSDKExceptionMessages:
    """Test SDK exception message formatting."""

    def test_client_error_str_includes_status(self):
        """MLSDMClientError string includes status code."""
        error = MLSDMClientError(
            message="Test error",
            status_code=400,
            error_type="validation_error",
        )

        assert "400" in str(error)
        assert "validation_error" in str(error)

    def test_server_error_str_includes_debug_id(self):
        """MLSDMServerError string includes debug_id."""
        error = MLSDMServerError(
            message="Server error",
            status_code=500,
            debug_id="debug-123",
        )

        assert "500" in str(error)
        assert "debug-123" in str(error)

    def test_timeout_error_str_includes_timeout(self):
        """MLSDMTimeoutError string includes timeout value."""
        error = MLSDMTimeoutError(
            message="Timed out",
            timeout_seconds=30.0,
        )

        assert "30.0" in str(error)

    def test_connection_error_str_includes_url(self):
        """MLSDMConnectionError string includes URL."""
        error = MLSDMConnectionError(
            message="Connection failed",
            url="http://test:8000",
        )

        assert "http://test:8000" in str(error)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
