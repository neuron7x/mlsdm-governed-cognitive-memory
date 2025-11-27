"""
SDK Client Integration Tests.

Tests the NeuroCognitiveClient SDK against the FastAPI test client to verify
HTTP-like behavior, timeouts, and error handling.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from requests.exceptions import ConnectionError, ReadTimeout

from mlsdm.adapters import LLMProviderError, LLMTimeoutError
from mlsdm.engine import NeuroEngineConfig
from mlsdm.sdk import NeuroCognitiveClient


class TestSDKAgainstTestServer:
    """Test SDK client with actual responses from engine."""

    @pytest.fixture(autouse=True)
    def setup_environment(self):
        """Set up test environment."""
        os.environ["LLM_BACKEND"] = "local_stub"
        yield
        if "LLM_BACKEND" in os.environ:
            del os.environ["LLM_BACKEND"]

    def test_sdk_generate_returns_valid_response(self):
        """Test that SDK generate returns valid response structure."""
        client = NeuroCognitiveClient(backend="local_stub")
        result = client.generate("Test prompt")

        # Core response fields
        assert "response" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    def test_sdk_generate_with_all_parameters(self):
        """Test SDK generate with all optional parameters."""
        client = NeuroCognitiveClient(backend="local_stub")
        result = client.generate(
            prompt="Test with all parameters",
            max_tokens=256,
            moral_value=0.7,
            user_intent="analytical",
            cognitive_load=0.5,
            context_top_k=5
        )

        assert "response" in result
        assert isinstance(result["response"], str)

    def test_sdk_response_contains_governance(self):
        """Test that SDK response contains governance information."""
        client = NeuroCognitiveClient(backend="local_stub")
        result = client.generate("Test governance")

        assert "governance" in result
        assert "mlsdm" in result

    def test_sdk_response_contains_timing(self):
        """Test that SDK response contains timing information."""
        client = NeuroCognitiveClient(backend="local_stub")
        result = client.generate("Test timing")

        assert "timing" in result
        # Timing can be dict or None
        if result["timing"] is not None:
            assert isinstance(result["timing"], dict)

    def test_sdk_response_contains_validation_steps(self):
        """Test that SDK response contains validation steps."""
        client = NeuroCognitiveClient(backend="local_stub")
        result = client.generate("Test validation")

        assert "validation_steps" in result
        assert isinstance(result["validation_steps"], list)

    def test_sdk_response_contains_mlsdm_state(self):
        """Test that SDK response contains MLSDM state."""
        client = NeuroCognitiveClient(backend="local_stub")
        result = client.generate("Test MLSDM state")

        assert "mlsdm" in result
        mlsdm_state = result["mlsdm"]
        assert isinstance(mlsdm_state, dict)
        assert "phase" in mlsdm_state


class TestSDKNetworkErrors:
    """Test SDK client network error handling."""

    def test_sdk_handles_provider_error(self):
        """Test that SDK propagates LLMProviderError."""
        client = NeuroCognitiveClient(backend="local_stub")

        with patch.object(client._engine, "generate") as mock_generate:
            mock_generate.side_effect = LLMProviderError(
                "Backend unavailable",
                provider_id="local_stub",
            )

            with pytest.raises(LLMProviderError) as exc_info:
                client.generate("Test")

            assert exc_info.value.provider_id == "local_stub"
            assert "Backend unavailable" in str(exc_info.value)

    def test_sdk_handles_timeout_error(self):
        """Test that SDK propagates LLMTimeoutError."""
        client = NeuroCognitiveClient(backend="local_stub")

        with patch.object(client._engine, "generate") as mock_generate:
            mock_generate.side_effect = LLMTimeoutError(
                "Request timed out",
                provider_id="local_stub",
                timeout_seconds=30.0,
            )

            with pytest.raises(LLMTimeoutError) as exc_info:
                client.generate("Test")

            assert exc_info.value.timeout_seconds == 30.0

    def test_sdk_handles_runtime_error(self):
        """Test that SDK propagates RuntimeError."""
        client = NeuroCognitiveClient(backend="local_stub")

        with patch.object(client._engine, "generate") as mock_generate:
            mock_generate.side_effect = RuntimeError("Unexpected error")

            with pytest.raises(RuntimeError, match="Unexpected error"):
                client.generate("Test")

    def test_sdk_handles_value_error(self):
        """Test that SDK propagates ValueError."""
        client = NeuroCognitiveClient(backend="local_stub")

        with patch.object(client._engine, "generate") as mock_generate:
            mock_generate.side_effect = ValueError("Invalid parameter")

            with pytest.raises(ValueError, match="Invalid parameter"):
                client.generate("Test")


class TestSDKConfiguration:
    """Test SDK client configuration options."""

    def test_sdk_with_local_stub_backend(self):
        """Test SDK with local_stub backend."""
        client = NeuroCognitiveClient(backend="local_stub")
        assert client.backend == "local_stub"
        result = client.generate("Test")
        assert "NEURO-RESPONSE" in result["response"]

    def test_sdk_invalid_backend_raises_error(self):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Invalid backend"):
            NeuroCognitiveClient(backend="invalid_backend")

    def test_sdk_openai_without_api_key_raises_error(self):
        """Test that OpenAI backend without API key raises error."""
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        with pytest.raises(ValueError, match="api_key"):
            NeuroCognitiveClient(backend="openai")

    def test_sdk_with_custom_config(self):
        """Test SDK with custom NeuroEngineConfig."""
        config = NeuroEngineConfig(
            dim=256,
            enable_fslgs=False,
            enable_metrics=True
        )
        client = NeuroCognitiveClient(backend="local_stub", config=config)

        assert client.config == config
        assert client.config.dim == 256
        assert client.config.enable_fslgs is False

    def test_sdk_openai_with_mocked_factory(self):
        """Test SDK OpenAI configuration with mocked factory."""
        import mlsdm.sdk.neuro_engine_client as sdk_module

        with patch.object(sdk_module, "build_neuro_engine_from_env") as mock_factory:
            mock_engine = MagicMock()
            mock_engine.generate.return_value = {
                "response": "test",
                "timing": {},
                "validation_steps": [],
                "mlsdm": {"phase": "wake"},
                "governance": {},
                "error": None,
                "rejected_at": None,
            }
            mock_factory.return_value = mock_engine

            client = NeuroCognitiveClient(
                backend="openai",
                api_key="sk-test-12345",
                model="gpt-4"
            )

            assert client.backend == "openai"
            assert os.environ.get("OPENAI_API_KEY") == "sk-test-12345"
            assert os.environ.get("OPENAI_MODEL") == "gpt-4"


class TestSDKResponseConsistency:
    """Test SDK response consistency across multiple calls."""

    def test_sdk_multiple_calls_consistent_structure(self):
        """Test that multiple SDK calls return consistent structure."""
        client = NeuroCognitiveClient(backend="local_stub")

        results = [client.generate(f"Test {i}") for i in range(3)]

        for result in results:
            assert "response" in result
            assert "timing" in result
            assert "mlsdm" in result
            assert "governance" in result
            assert "error" in result
            assert "rejected_at" in result

    def test_sdk_response_phase_valid(self):
        """Test that MLSDM phase is valid."""
        client = NeuroCognitiveClient(backend="local_stub")
        result = client.generate("Test phase")

        mlsdm_state = result.get("mlsdm", {})
        phase = mlsdm_state.get("phase")
        assert phase in ["wake", "sleep", "unknown"]


class TestSDKEdgeCases:
    """Test SDK edge cases."""

    def test_sdk_empty_prompt(self):
        """Test SDK with empty prompt passes to engine."""
        client = NeuroCognitiveClient(backend="local_stub")
        # The SDK passes through to engine, validation happens there
        result = client.generate("")
        assert "response" in result

    def test_sdk_very_long_prompt(self):
        """Test SDK with very long prompt."""
        client = NeuroCognitiveClient(backend="local_stub")
        long_prompt = "Test " * 1000
        result = client.generate(long_prompt)
        assert "response" in result

    def test_sdk_unicode_prompt(self):
        """Test SDK with unicode prompt."""
        client = NeuroCognitiveClient(backend="local_stub")
        result = client.generate("こんにちは 🌍 مرحبا")
        assert "response" in result

    def test_sdk_moral_value_boundary_low(self):
        """Test SDK with moral_value at lower boundary."""
        client = NeuroCognitiveClient(backend="local_stub")
        result = client.generate("Test", moral_value=0.0)
        assert "response" in result

    def test_sdk_moral_value_boundary_high(self):
        """Test SDK with moral_value at upper boundary."""
        client = NeuroCognitiveClient(backend="local_stub")
        result = client.generate("Test", moral_value=1.0)
        assert "response" in result


class TestSDKRetryBehavior:
    """Test SDK retry behavior (or lack thereof)."""

    def test_sdk_does_not_retry_by_default(self):
        """Test that SDK does not automatically retry on failure."""
        client = NeuroCognitiveClient(backend="local_stub")
        failure_count = 0

        def fail_once(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1
            raise LLMProviderError("Temporary failure", provider_id="local_stub")

        with patch.object(client._engine, "generate", side_effect=fail_once):
            with pytest.raises(LLMProviderError):
                client.generate("Test")

        # Should only be called once (no retry)
        assert failure_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
