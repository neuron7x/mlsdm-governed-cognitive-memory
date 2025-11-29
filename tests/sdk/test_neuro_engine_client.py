"""
SDK Client Tests for NeuroCognitiveClient.

Tests cover:
- HTTP-like request behavior (simulated via engine)
- Error handling (timeouts, retries, network errors)
- Parameter passing and response handling
- Backend configuration
- Typed DTO response structure
- SDK-specific exceptions

Note: These tests use the local_stub backend to simulate HTTP behavior
since the SDK wraps the engine directly rather than making HTTP calls.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from mlsdm.adapters import LLMProviderError, LLMTimeoutError
from mlsdm.engine import NeuroEngineConfig
from mlsdm.sdk import (
    GenerateResponseDTO,
    MLSDMServerError,
    MLSDMTimeoutError,
    NeuroCognitiveClient,
)


class TestNeuroCognitiveClientHttpBehavior:
    """Test SDK client HTTP-like behavior."""

    def test_generate_request_to_engine(self):
        """Test that generate makes correct request to engine."""
        client = NeuroCognitiveClient(backend="local_stub")
        result = client.generate("Test prompt")

        # Verify response is a typed DTO
        assert isinstance(result, GenerateResponseDTO)

        # Verify DTO has required fields
        assert hasattr(result, "response")
        assert hasattr(result, "phase")
        assert hasattr(result, "accepted")
        assert hasattr(result, "metrics")
        assert hasattr(result, "safety_flags")
        assert hasattr(result, "memory_stats")

    def test_generate_passes_all_parameters(self):
        """Test that all parameters are passed to engine."""
        client = NeuroCognitiveClient()

        with patch.object(client._engine, "generate") as mock_generate:
            mock_generate.return_value = {
                "response": "test",
                "timing": {},
                "validation_steps": [],
                "mlsdm": {},
                "governance": {},
                "error": None,
                "rejected_at": None,
            }

            client.generate(
                prompt="Test",
                max_tokens=256,
                moral_value=0.8,
                user_intent="analytical",
                cognitive_load=0.3,
                context_top_k=10,
            )

            mock_generate.assert_called_once()
            call_kwargs = mock_generate.call_args[1]
            assert call_kwargs["prompt"] == "Test"
            assert call_kwargs["max_tokens"] == 256
            assert call_kwargs["moral_value"] == 0.8
            assert call_kwargs["user_intent"] == "analytical"
            assert call_kwargs["cognitive_load"] == 0.3
            assert call_kwargs["context_top_k"] == 10

    def test_generate_with_minimal_parameters(self):
        """Test generate with only required prompt parameter."""
        client = NeuroCognitiveClient()
        result = client.generate("Minimal test")

        # Result is a DTO
        assert isinstance(result, GenerateResponseDTO)
        assert len(result.response) > 0

    def test_generate_returns_neuro_response_prefix(self):
        """Test that local_stub returns recognizable response."""
        client = NeuroCognitiveClient(backend="local_stub")
        result = client.generate("Hello world")

        # Access response via DTO attribute
        assert "NEURO-RESPONSE" in result.response

    def test_generate_raw_returns_dict(self):
        """Test that generate_raw returns raw dictionary."""
        client = NeuroCognitiveClient(backend="local_stub")
        result = client.generate_raw("Test prompt")

        # Verify raw dict structure
        assert isinstance(result, dict)
        assert "response" in result
        assert "timing" in result
        assert "mlsdm" in result


class TestNeuroCognitiveClientErrorHandling:
    """Test SDK client error handling."""

    def test_handles_engine_exception(self):
        """Test that engine exceptions are propagated."""
        client = NeuroCognitiveClient()

        with patch.object(client._engine, "generate") as mock_generate:
            mock_generate.side_effect = RuntimeError("Engine error")

            with pytest.raises(RuntimeError, match="Engine error"):
                client.generate("Test")

    def test_handles_provider_error_as_sdk_error(self):
        """Test that LLMProviderError is wrapped in MLSDMServerError."""
        client = NeuroCognitiveClient()

        with patch.object(client._engine, "generate") as mock_generate:
            mock_generate.side_effect = LLMProviderError(
                "Provider failed",
                provider_id="test",
            )

            with pytest.raises(MLSDMServerError) as exc_info:
                client.generate("Test")

            assert "Provider failed" in str(exc_info.value)

    def test_handles_timeout_error_as_sdk_error(self):
        """Test that timeout errors are wrapped in MLSDMTimeoutError."""
        client = NeuroCognitiveClient()

        with patch.object(client._engine, "generate") as mock_generate:
            mock_generate.side_effect = LLMTimeoutError(
                "Request timed out",
                provider_id="test",
                timeout_seconds=30.0,
            )

            with pytest.raises(MLSDMTimeoutError) as exc_info:
                client.generate("Test")

            assert exc_info.value.timeout_seconds == 30.0

    def test_sdk_error_contains_details(self):
        """Test that SDK error includes relevant details."""
        client = NeuroCognitiveClient()

        with patch.object(client._engine, "generate") as mock_generate:
            mock_generate.side_effect = LLMProviderError(
                "Provider error",
                provider_id="local_stub",
            )

            with pytest.raises(MLSDMServerError) as exc_info:
                client.generate("Test")

            assert exc_info.value.details.get("provider_id") == "local_stub"


class TestNeuroCognitiveClientTimeout:
    """Test SDK client timeout behavior."""

    def test_default_timeout_behavior(self):
        """Test client works with default timeout settings."""
        client = NeuroCognitiveClient()
        # Should complete without timeout on local stub
        result = client.generate("Quick test")
        assert isinstance(result, GenerateResponseDTO)
        assert result.response  # Non-empty response

    def test_engine_with_different_max_tokens(self):
        """Test that max_tokens parameter is passed to engine."""
        client = NeuroCognitiveClient()

        # Both should work and return valid responses
        result_small = client.generate("Test", max_tokens=10)
        result_large = client.generate("Test", max_tokens=500)

        # Both should return valid DTOs with responses
        assert isinstance(result_small, GenerateResponseDTO)
        assert isinstance(result_large, GenerateResponseDTO)

    def test_client_has_timeout_property(self):
        """Test that client exposes timeout configuration."""
        client = NeuroCognitiveClient(timeout=60.0)
        assert client.timeout == 60.0


class TestNeuroCognitiveClient4xxErrors:
    """Test SDK client 4xx error simulation."""

    def test_invalid_backend_raises_value_error(self):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Invalid backend"):
            NeuroCognitiveClient(backend="invalid")

    def test_openai_without_api_key_raises_error(self):
        """Test that OpenAI backend without API key raises error."""
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        with pytest.raises(ValueError, match="api_key"):
            NeuroCognitiveClient(backend="openai")

    def test_empty_prompt_is_passed_to_engine(self):
        """Test that empty prompt handling is done by engine."""
        client = NeuroCognitiveClient()

        # The SDK passes through to engine, validation happens there
        # Local stub will process even empty prompts
        result = client.generate("")
        assert isinstance(result, GenerateResponseDTO)


class TestNeuroCognitiveClient5xxErrors:
    """Test SDK client 5xx error simulation."""

    def test_engine_internal_error_propagates(self):
        """Test that internal engine errors propagate correctly."""
        client = NeuroCognitiveClient()

        with patch.object(client._engine, "generate") as mock_generate:
            mock_generate.side_effect = Exception("Internal server error")

            with pytest.raises(Exception, match="Internal server error"):
                client.generate("Test")

    def test_generator_failure_wraps_in_sdk_error(self):
        """Test that generator failure is wrapped in SDK error."""
        client = NeuroCognitiveClient()

        with patch.object(client._engine, "generate") as mock_generate:
            mock_generate.side_effect = LLMProviderError(
                "Generator failed",
                provider_id="local_stub",
                original_error=RuntimeError("LLM crash"),
            )

            with pytest.raises(MLSDMServerError) as exc_info:
                client.generate("Test")

            assert "Generator failed" in str(exc_info.value)


class TestNeuroCognitiveClientBackendConfiguration:
    """Test SDK client backend configuration."""

    def test_local_stub_backend(self):
        """Test client with local_stub backend."""
        client = NeuroCognitiveClient(backend="local_stub")
        assert client.backend == "local_stub"

        result = client.generate("Test")
        assert "NEURO-RESPONSE" in result.response

    def test_openai_backend_with_api_key(self):
        """Test client initialization with OpenAI backend (mocked)."""
        with patch("mlsdm.sdk.neuro_engine_client.build_neuro_engine_from_env") as mock_factory:
            mock_engine = MagicMock()
            mock_factory.return_value = mock_engine

            client = NeuroCognitiveClient(
                backend="openai",
                api_key="sk-test-12345",
                model="gpt-4"
            )

            assert client.backend == "openai"
            assert os.environ.get("OPENAI_API_KEY") == "sk-test-12345"
            assert os.environ.get("OPENAI_MODEL") == "gpt-4"

    def test_config_passthrough(self):
        """Test that config is passed to engine."""
        config = NeuroEngineConfig(dim=256, enable_fslgs=False)
        client = NeuroCognitiveClient(config=config)

        assert client.config == config
        assert client.config.dim == 256


class TestNeuroCognitiveClientResponseStructure:
    """Test SDK client response structure matches API contract."""

    def test_response_dto_has_all_required_fields(self):
        """Test that response DTO includes all expected fields."""
        client = NeuroCognitiveClient()
        result = client.generate("Test prompt")

        # Core fields (STABLE CONTRACT)
        assert hasattr(result, "response")
        assert isinstance(result.response, str)
        assert hasattr(result, "phase")
        assert isinstance(result.phase, str)
        assert hasattr(result, "accepted")
        assert isinstance(result.accepted, bool)

        # Optional fields
        assert hasattr(result, "metrics")
        assert hasattr(result, "safety_flags")
        assert hasattr(result, "memory_stats")
        assert hasattr(result, "error")
        assert hasattr(result, "rejected_at")

    def test_dto_phase_is_valid(self):
        """Test that phase is a valid value."""
        client = NeuroCognitiveClient()
        result = client.generate("Test")

        assert result.phase in ["wake", "sleep", "unknown"]

    def test_dto_has_helper_properties(self):
        """Test that DTO has helper properties."""
        client = NeuroCognitiveClient()
        result = client.generate("Test")

        # Helper properties
        assert hasattr(result, "is_success")
        assert hasattr(result, "is_rejected")
        assert hasattr(result, "has_error")
        assert hasattr(result, "raw")

        # A successful response should be marked as success
        assert result.is_success is True
        assert result.is_rejected is False
        assert result.has_error is False

    def test_dto_to_dict_conversion(self):
        """Test that DTO can be converted to dictionary."""
        client = NeuroCognitiveClient()
        result = client.generate("Test")

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "response" in result_dict
        assert "phase" in result_dict
        assert "accepted" in result_dict


class TestNeuroCognitiveClientRetryBehavior:
    """Test SDK client retry behavior simulation."""

    def test_single_failure_propagates(self):
        """Test that a single failure propagates without automatic retry."""
        client = NeuroCognitiveClient()
        failure_count = 0

        def fail_once(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1
            raise LLMProviderError("Temporary failure")

        with (
            patch.object(client._engine, "generate", side_effect=fail_once),
            pytest.raises(MLSDMServerError),
        ):
            client.generate("Test")

        # SDK doesn't retry by default
        assert failure_count == 1

    def test_consistent_results_on_success(self):
        """Test that successful calls return consistent structure."""
        client = NeuroCognitiveClient()

        result1 = client.generate("Test 1")
        result2 = client.generate("Test 2")

        # Both should be DTOs with same structure
        for result in [result1, result2]:
            assert isinstance(result, GenerateResponseDTO)
            assert hasattr(result, "response")
            assert hasattr(result, "phase")
            assert hasattr(result, "accepted")


class TestNeuroCognitiveClientSDKExceptions:
    """Test SDK-specific exception handling."""

    def test_sdk_exceptions_are_exported(self):
        """Test that SDK exceptions are exported from module."""
        from mlsdm.sdk import (
            MLSDMClientError,
            MLSDMConfigError,
            MLSDMError,
            MLSDMRateLimitError,
            MLSDMServerError,
            MLSDMTimeoutError,
            MLSDMValidationError,
        )

        # Verify exception hierarchy
        assert issubclass(MLSDMClientError, MLSDMError)
        assert issubclass(MLSDMServerError, MLSDMError)
        assert issubclass(MLSDMTimeoutError, MLSDMError)
        assert issubclass(MLSDMValidationError, MLSDMClientError)
        assert issubclass(MLSDMConfigError, MLSDMClientError)
        assert issubclass(MLSDMRateLimitError, MLSDMClientError)

    def test_server_error_has_error_type(self):
        """Test that MLSDMServerError has error_type attribute."""
        from mlsdm.sdk import MLSDMServerError

        error = MLSDMServerError("Test error", error_type="test_error")
        assert error.error_type == "test_error"
        assert error.message == "Test error"

    def test_timeout_error_has_timeout_seconds(self):
        """Test that MLSDMTimeoutError has timeout_seconds attribute."""
        from mlsdm.sdk import MLSDMTimeoutError

        error = MLSDMTimeoutError("Timeout", timeout_seconds=30.0, operation="generate")
        assert error.timeout_seconds == 30.0
        assert error.operation == "generate"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
