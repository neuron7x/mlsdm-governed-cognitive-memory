"""
Integration tests for API inference endpoints.

Tests cover:
- Minimal inference scenario (happy path)
- 4xx error responses for invalid payloads
- Response contract validation
- Edge cases and error handling
"""

import os

import pytest
from fastapi import status
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


class TestInferHappyPath:
    """Test /infer endpoint happy path scenarios."""

    def test_infer_minimal_request(self, client):
        """Test minimal inference request with just prompt."""
        response = client.post(
            "/infer",
            json={"prompt": "Hello, world!"}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # Verify response structure is valid
        assert "response" in data
        assert "accepted" in data
        assert "phase" in data
        assert isinstance(data["response"], str)
        assert isinstance(data["accepted"], bool)
        assert isinstance(data["phase"], str)

    def test_infer_response_contract(self, client):
        """Test that infer response matches InferResponse contract."""
        response = client.post(
            "/infer",
            json={"prompt": "Test prompt for contract validation"}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Required fields
        assert "response" in data
        assert "accepted" in data
        assert "phase" in data

        # Type checks
        assert isinstance(data["response"], str)
        assert isinstance(data["accepted"], bool)
        assert isinstance(data["phase"], str)

        # Optional fields (must be present, can be None)
        assert "moral_metadata" in data
        assert "aphasia_metadata" in data
        assert "rag_metadata" in data
        assert "timing" in data
        assert "governance" in data

    def test_infer_with_moral_value(self, client):
        """Test inference with custom moral value."""
        response = client.post(
            "/infer",
            json={"prompt": "Test with moral value", "moral_value": 0.7}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        moral_meta = data.get("moral_metadata", {})
        assert moral_meta.get("applied_moral_value") == 0.7

    def test_infer_with_max_tokens(self, client):
        """Test inference with max_tokens parameter."""
        response = client.post(
            "/infer",
            json={"prompt": "Test with max tokens", "max_tokens": 100}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "response" in data

    def test_infer_returns_timing_metrics(self, client):
        """Test that inference returns timing metrics."""
        response = client.post(
            "/infer",
            json={"prompt": "Test timing metrics"}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # Timing should be present (may be dict or None)
        assert "timing" in data


class TestInfer4xxErrors:
    """Test /infer endpoint 4xx error responses."""

    def test_infer_empty_prompt_returns_422(self, client):
        """Test that empty prompt returns 422 validation error."""
        response = client.post(
            "/infer",
            json={"prompt": ""}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], list)

    def test_infer_whitespace_prompt_returns_400(self, client):
        """Test that whitespace-only prompt returns 400."""
        response = client.post(
            "/infer",
            json={"prompt": "   "}
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "error" in data
        assert data["error"]["error_type"] == "validation_error"

    def test_infer_missing_prompt_returns_422(self, client):
        """Test that missing prompt returns 422."""
        response = client.post(
            "/infer",
            json={}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "detail" in data

    def test_infer_invalid_moral_value_range_returns_422(self, client):
        """Test that moral_value > 1.0 returns 422."""
        response = client.post(
            "/infer",
            json={"prompt": "Test", "moral_value": 1.5}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_infer_negative_moral_value_returns_422(self, client):
        """Test that moral_value < 0.0 returns 422."""
        response = client.post(
            "/infer",
            json={"prompt": "Test", "moral_value": -0.5}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_infer_invalid_max_tokens_returns_422(self, client):
        """Test that max_tokens > 4096 returns 422."""
        response = client.post(
            "/infer",
            json={"prompt": "Test", "max_tokens": 5000}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_infer_negative_max_tokens_returns_422(self, client):
        """Test that max_tokens < 1 returns 422."""
        response = client.post(
            "/infer",
            json={"prompt": "Test", "max_tokens": 0}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_infer_invalid_content_type_returns_422(self, client):
        """Test that non-JSON content type returns error."""
        response = client.post(
            "/infer",
            data="prompt=test",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestGenerateHappyPath:
    """Test /generate endpoint happy path scenarios."""

    def test_generate_minimal_request(self, client):
        """Test minimal generate request."""
        response = client.post(
            "/generate",
            json={"prompt": "Hello, world!"}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "response" in data
        assert "phase" in data
        assert "accepted" in data

    def test_generate_response_contract(self, client):
        """Test that generate response matches GenerateResponse contract."""
        response = client.post(
            "/generate",
            json={"prompt": "Test generate contract"}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        # Required fields
        assert "response" in data
        assert "phase" in data
        assert "accepted" in data

        # Type checks
        assert isinstance(data["response"], str)
        assert isinstance(data["phase"], str)
        assert isinstance(data["accepted"], bool)

        # Optional fields (must be present, can be None)
        assert "metrics" in data
        assert "safety_flags" in data
        assert "memory_stats" in data

    def test_generate_with_all_parameters(self, client):
        """Test generate with all optional parameters."""
        response = client.post(
            "/generate",
            json={
                "prompt": "Test with all params",
                "max_tokens": 256,
                "moral_value": 0.8
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data["response"], str)


class TestGenerate4xxErrors:
    """Test /generate endpoint 4xx error responses."""

    def test_generate_empty_prompt_returns_422(self, client):
        """Test that empty prompt returns 422."""
        response = client.post(
            "/generate",
            json={"prompt": ""}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_generate_whitespace_prompt_returns_400(self, client):
        """Test that whitespace-only prompt returns 400."""
        response = client.post(
            "/generate",
            json={"prompt": "   "}
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "error" in data
        assert data["error"]["error_type"] == "validation_error"

    def test_generate_invalid_moral_value_returns_422(self, client):
        """Test that invalid moral_value returns 422."""
        response = client.post(
            "/generate",
            json={"prompt": "Test", "moral_value": 2.0}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_generate_invalid_max_tokens_returns_422(self, client):
        """Test that invalid max_tokens returns 422."""
        response = client.post(
            "/generate",
            json={"prompt": "Test", "max_tokens": 10000}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestInferSecureMode:
    """Test /infer endpoint secure mode behavior."""

    def test_secure_mode_boosts_moral_threshold(self, client):
        """Test that secure_mode increases moral threshold by 0.2."""
        response = client.post(
            "/infer",
            json={
                "prompt": "Test secure mode",
                "secure_mode": True,
                "moral_value": 0.5
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        moral_meta = data.get("moral_metadata", {})
        assert moral_meta.get("secure_mode") is True
        assert moral_meta.get("applied_moral_value") == 0.7  # 0.5 + 0.2

    def test_secure_mode_caps_at_one(self, client):
        """Test that secure_mode caps moral value at 1.0."""
        response = client.post(
            "/infer",
            json={
                "prompt": "Test secure mode cap",
                "secure_mode": True,
                "moral_value": 0.9  # 0.9 + 0.2 = 1.1, should cap at 1.0
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        moral_meta = data.get("moral_metadata", {})
        assert moral_meta.get("applied_moral_value") == 1.0

    def test_secure_mode_without_rag(self, client):
        """Test secure_mode works with RAG disabled."""
        response = client.post(
            "/infer",
            json={
                "prompt": "Test secure without RAG",
                "secure_mode": True,
                "rag_enabled": False
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data.get("rag_metadata", {}).get("enabled") is False


class TestInferRagMode:
    """Test /infer endpoint RAG mode behavior."""

    def test_rag_enabled_by_default(self, client):
        """Test that RAG is enabled by default."""
        response = client.post(
            "/infer",
            json={"prompt": "Test RAG default"}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        rag_meta = data.get("rag_metadata", {})
        assert rag_meta.get("enabled") is True

    def test_rag_disabled_explicitly(self, client):
        """Test that RAG can be explicitly disabled."""
        response = client.post(
            "/infer",
            json={"prompt": "Test RAG disabled", "rag_enabled": False}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        rag_meta = data.get("rag_metadata", {})
        assert rag_meta.get("enabled") is False
        assert rag_meta.get("context_items_retrieved") == 0

    def test_rag_with_custom_top_k(self, client):
        """Test RAG with custom context_top_k."""
        response = client.post(
            "/infer",
            json={
                "prompt": "Test RAG top_k",
                "rag_enabled": True,
                "context_top_k": 10
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        rag_meta = data.get("rag_metadata", {})
        assert rag_meta.get("top_k") == 10


class TestInferAphasiaMode:
    """Test /infer endpoint aphasia mode behavior."""

    def test_aphasia_mode_enabled(self, client):
        """Test that aphasia_mode returns aphasia metadata."""
        response = client.post(
            "/infer",
            json={"prompt": "Test aphasia mode", "aphasia_mode": True}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        aphasia_meta = data.get("aphasia_metadata")
        assert aphasia_meta is not None
        assert aphasia_meta.get("enabled") is True

    def test_aphasia_mode_disabled_by_default(self, client):
        """Test that aphasia_mode is disabled by default."""
        response = client.post(
            "/infer",
            json={"prompt": "Test aphasia default"}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # aphasia_metadata should be None or not enabled
        aphasia_meta = data.get("aphasia_metadata")
        if aphasia_meta is not None:
            assert aphasia_meta.get("enabled") is not True


class TestInferResponseHeaders:
    """Test /infer endpoint response headers."""

    def test_infer_returns_request_id_header(self, client):
        """Test that infer returns X-Request-ID header."""
        response = client.post(
            "/infer",
            json={"prompt": "Test headers"}
        )
        assert "x-request-id" in response.headers

    def test_infer_returns_security_headers(self, client):
        """Test that infer returns security headers."""
        response = client.post(
            "/infer",
            json={"prompt": "Test security headers"}
        )
        assert "x-content-type-options" in response.headers
        assert "x-frame-options" in response.headers

    def test_infer_propagates_custom_request_id(self, client):
        """Test that custom X-Request-ID is propagated."""
        custom_id = "custom-inference-12345"
        response = client.post(
            "/infer",
            json={"prompt": "Test custom ID"},
            headers={"X-Request-ID": custom_id}
        )
        assert response.headers.get("x-request-id") == custom_id


class TestInferEdgeCases:
    """Test /infer endpoint edge cases."""

    def test_infer_very_long_prompt(self, client):
        """Test inference with a very long prompt."""
        long_prompt = "Test " * 1000  # 5000 characters
        response = client.post(
            "/infer",
            json={"prompt": long_prompt}
        )
        # Should still work (may be slow)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "response" in data

    def test_infer_unicode_prompt(self, client):
        """Test inference with unicode characters."""
        response = client.post(
            "/infer",
            json={"prompt": "こんにちは世界 🌍 مرحبا العالم"}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "response" in data

    def test_infer_special_characters_prompt(self, client):
        """Test inference with special characters."""
        response = client.post(
            "/infer",
            json={"prompt": "Test <script>alert('xss')</script> & special chars"}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "response" in data

    def test_infer_newlines_in_prompt(self, client):
        """Test inference with newlines in prompt."""
        response = client.post(
            "/infer",
            json={"prompt": "Line 1\nLine 2\nLine 3"}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "response" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
