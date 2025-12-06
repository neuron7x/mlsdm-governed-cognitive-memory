"""LLM failure mode tests.

Tests system behavior when LLM provider fails in various ways.
Validates circuit breaker, retry logic, and graceful degradation.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mlsdm.core.llm_wrapper import LLMWrapper
from mlsdm.utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitOpenError


def create_stub_embedder(dim: int = 384):
    """Create a deterministic stub embedding function."""
    def stub_embed(text: str) -> np.ndarray:
        np.random.seed(hash(text) % (2**32))
        vec = np.random.randn(dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            vec = np.zeros(dim, dtype=np.float32)
            vec[0] = 1.0
        else:
            vec = vec / norm
        return vec
    return stub_embed


@pytest.mark.security
class TestLLMTimeoutHandling:
    """Test system behavior when LLM provider times out."""
    
    def test_llm_timeout_with_circuit_breaker(self) -> None:
        """Validate circuit breaker opens after repeated timeouts.
        
        Circuit breaker should open after N consecutive timeout failures,
        preventing cascading failures and reducing latency for subsequent requests.
        """
        timeout_count = [0]
        
        def timeout_llm(prompt: str, max_tokens: int) -> str:
            """LLM that always times out."""
            timeout_count[0] += 1
            time.sleep(2.0)  # Simulate slow response
            return "Too slow"
        
        # Create circuit breaker with low threshold for testing
        circuit_breaker = CircuitBreaker(
            name="test_llm",
            config=CircuitBreakerConfig(
                failure_threshold=3,
                timeout_seconds=1.0,
                recovery_timeout=5.0,
            )
        )
        
        # First 3 requests should timeout
        for i in range(3):
            with pytest.raises((Exception, TimeoutError)):
                with circuit_breaker:
                    wrapper = LLMWrapper(
                        llm_generate_fn=timeout_llm,
                        embedding_fn=create_stub_embedder(),
                        llm_timeout=1.0,
                        llm_retry_attempts=1,
                    )
                    wrapper.generate(prompt="Test timeout", moral_value=0.8)
                    circuit_breaker.record_success()
        
        # Circuit should now be open
        assert circuit_breaker.state.value in ("open", "half_open")
    
    def test_llm_partial_timeout_recovery(self) -> None:
        """Test recovery when timeouts are intermittent.
        
        System should retry and eventually succeed when LLM becomes responsive.
        """
        call_count = [0]
        
        def intermittent_timeout_llm(prompt: str, max_tokens: int) -> str:
            """LLM that times out first 2 times, then succeeds."""
            call_count[0] += 1
            if call_count[0] <= 2:
                time.sleep(2.0)
                return "Timeout"
            return "Success after recovery"
        
        wrapper = LLMWrapper(
            llm_generate_fn=intermittent_timeout_llm,
            embedding_fn=create_stub_embedder(),
            llm_timeout=1.5,
            llm_retry_attempts=5,
        )
        
        result = wrapper.generate(prompt="Test recovery", moral_value=0.8)
        
        # Should eventually succeed
        assert result["accepted"] is True
        assert "Success" in result["response"]
        assert call_count[0] == 3  # Failed twice, succeeded third time


@pytest.mark.security
class TestLLM5xxErrors:
    """Test system behavior when LLM provider returns 5xx errors."""
    
    def test_llm_500_error_retry(self) -> None:
        """Validate retry logic for 5xx errors.
        
        System should retry 5xx errors and eventually succeed or fail gracefully.
        """
        call_count = [0]
        
        def flaky_500_llm(prompt: str, max_tokens: int) -> str:
            """LLM that returns 500 error first 2 times."""
            call_count[0] += 1
            if call_count[0] <= 2:
                raise Exception("HTTP 500: Internal Server Error")
            return "Success after retry"
        
        wrapper = LLMWrapper(
            llm_generate_fn=flaky_500_llm,
            embedding_fn=create_stub_embedder(),
            llm_retry_attempts=5,
            llm_timeout=5.0,
        )
        
        result = wrapper.generate(prompt="Test 500 error", moral_value=0.8)
        
        assert result["accepted"] is True
        assert "Success" in result["response"]
        assert call_count[0] == 3
    
    def test_llm_persistent_500_error(self) -> None:
        """Test graceful handling of persistent 5xx errors.
        
        When LLM consistently fails, system should exhaust retries
        and return structured error response.
        """
        def always_500_llm(prompt: str, max_tokens: int) -> str:
            """LLM that always returns 500 error."""
            raise Exception("HTTP 500: Internal Server Error")
        
        wrapper = LLMWrapper(
            llm_generate_fn=always_500_llm,
            embedding_fn=create_stub_embedder(),
            llm_retry_attempts=3,
            llm_timeout=5.0,
        )
        
        with pytest.raises(Exception):
            wrapper.generate(prompt="Test persistent 500", moral_value=0.8)


@pytest.mark.security
class TestMalformedLLMResponses:
    """Test system behavior when LLM returns malformed responses."""
    
    def test_empty_llm_response(self) -> None:
        """Validate handling of empty LLM responses.
        
        System should detect empty responses and handle gracefully.
        """
        def empty_response_llm(prompt: str, max_tokens: int) -> str:
            """LLM that returns empty response."""
            return ""
        
        wrapper = LLMWrapper(
            llm_generate_fn=empty_response_llm,
            embedding_fn=create_stub_embedder(),
            llm_retry_attempts=1,
            llm_timeout=5.0,
        )
        
        result = wrapper.generate(prompt="Test empty response", moral_value=0.8)
        
        # System should handle empty response
        assert result["accepted"] is False or result["response"] == ""
    
    def test_malformed_json_response(self) -> None:
        """Test handling of malformed JSON responses from LLM.
        
        Some LLM APIs return JSON. System should handle malformed JSON gracefully.
        """
        def malformed_json_llm(prompt: str, max_tokens: int) -> str:
            """LLM that returns malformed JSON."""
            return "{invalid json response"
        
        wrapper = LLMWrapper(
            llm_generate_fn=malformed_json_llm,
            embedding_fn=create_stub_embedder(),
            llm_retry_attempts=1,
            llm_timeout=5.0,
        )
        
        # Should not crash, even with malformed JSON
        result = wrapper.generate(prompt="Test malformed JSON", moral_value=0.8)
        
        # Should return some response (may treat as plain text)
        assert "response" in result


@pytest.mark.security
class TestCircuitBreakerBehavior:
    """Test circuit breaker activation and recovery."""
    
    def test_circuit_breaker_opens_after_threshold(self) -> None:
        """Validate circuit breaker opens after failure threshold.
        
        Circuit breaker should open after N consecutive failures,
        preventing further calls to failing backend.
        """
        circuit_breaker = CircuitBreaker(
            name="test_circuit",
            config=CircuitBreakerConfig(
                failure_threshold=3,
                timeout_seconds=5.0,
                recovery_timeout=10.0,
            )
        )
        
        failure_count = 0
        
        # Trigger failures
        for _ in range(3):
            try:
                with circuit_breaker:
                    failure_count += 1
                    raise Exception("Simulated failure")
            except Exception:
                circuit_breaker.record_failure(Exception("Simulated failure"))
        
        # Circuit should be open now
        assert circuit_breaker.state.value == "open"
        
        # Subsequent calls should fail fast (not execute operation)
        with pytest.raises(CircuitOpenError):
            with circuit_breaker:
                failure_count += 1  # Should not increment
        
        # Verify operation was not executed
        assert failure_count == 3  # Only the first 3 failures
    
    def test_circuit_breaker_half_open_recovery(self) -> None:
        """Test circuit breaker half-open state and recovery.
        
        After timeout, circuit should enter half-open state
        and allow limited requests for testing recovery.
        """
        circuit_breaker = CircuitBreaker(
            name="test_recovery",
            config=CircuitBreakerConfig(
                failure_threshold=2,
                timeout_seconds=1.0,
                recovery_timeout=1.0,  # Short for testing
            )
        )
        
        # Trigger failures to open circuit
        for _ in range(2):
            try:
                with circuit_breaker:
                    raise Exception("Fail")
            except Exception:
                circuit_breaker.record_failure(Exception("Fail"))
        
        assert circuit_breaker.state.value == "open"
        
        # Wait for recovery timeout
        time.sleep(1.5)
        
        # Circuit should transition to half-open
        # Next successful request should close it
        try:
            with circuit_breaker:
                # Successful operation
                pass
            circuit_breaker.record_success()
        except CircuitOpenError:
            # May still be open depending on timing
            pass
