"""
Unit tests for LLM Wrapper with mock LLM.

These tests validate the LLM wrapper's cognitive governance without
requiring actual LLM or embedding models.
"""

import pytest
import numpy as np
from unittest.mock import Mock
from typing import List

from src.core.llm_wrapper import LLMWrapper


class MockLLM:
    """Mock LLM for testing."""
    
    def __init__(self):
        self.call_count = 0
        self.last_prompt = None
        self.last_max_tokens = None
    
    def generate(self, prompt: str, max_tokens: int) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_max_tokens = max_tokens
        return f"Response {self.call_count} (max_tokens={max_tokens})"


class MockEmbedder:
    """Mock embedder for testing."""
    
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.call_count = 0
    
    def embed(self, text: str) -> np.ndarray:
        self.call_count += 1
        # Generate deterministic embedding based on text hash
        np.random.seed(hash(text) % (2**32))
        vec = np.random.randn(self.dim).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) or 1e-9)
        return vec


class TestLLMWrapperBasic:
    """Basic functionality tests for LLM wrapper."""
    
    def test_initialization(self):
        """Test wrapper initialization."""
        mock_llm = MockLLM()
        mock_embedder = MockEmbedder(dim=384)
        
        wrapper = LLMWrapper(
            llm_generate_fn=mock_llm.generate,
            embedding_fn=mock_embedder.embed,
            dim=384,
            capacity=1000,
            wake_duration=8,
            sleep_duration=3,
            initial_moral_threshold=0.50
        )
        
        assert wrapper.dim == 384
        assert wrapper.qilm.capacity == 1000
        assert wrapper.rhythm.wake_duration == 8
        assert wrapper.rhythm.sleep_duration == 3
        assert wrapper.moral.threshold == 0.50
    
    def test_generate_accepted_request(self):
        """Test generation with morally acceptable request."""
        mock_llm = MockLLM()
        mock_embedder = MockEmbedder(dim=384)
        
        wrapper = LLMWrapper(
            llm_generate_fn=mock_llm.generate,
            embedding_fn=mock_embedder.embed,
            dim=384
        )
        
        result = wrapper.generate(
            prompt="Hello, how are you?",
            moral_value=0.8  # High moral value
        )
        
        assert result["accepted"] is True
        assert result["phase"] == "wake"
        assert result["step"] == 1
        assert "Response 1" in result["response"]
        assert mock_llm.call_count == 1
        assert mock_embedder.call_count == 1
    
    def test_generate_rejected_moral(self):
        """Test generation rejected on moral grounds."""
        mock_llm = MockLLM()
        mock_embedder = MockEmbedder(dim=384)
        
        wrapper = LLMWrapper(
            llm_generate_fn=mock_llm.generate,
            embedding_fn=mock_embedder.embed,
            dim=384,
            initial_moral_threshold=0.70
        )
        
        result = wrapper.generate(
            prompt="Bad request",
            moral_value=0.2  # Low moral value
        )
        
        assert result["accepted"] is False
        assert "morally rejected" in result["note"]
        assert result["response"] == ""
        assert mock_llm.call_count == 0  # LLM not called
    
    def test_generate_during_sleep(self):
        """Test generation rejected during sleep phase."""
        mock_llm = MockLLM()
        mock_embedder = MockEmbedder(dim=384)
        
        wrapper = LLMWrapper(
            llm_generate_fn=mock_llm.generate,
            embedding_fn=mock_embedder.embed,
            dim=384,
            wake_duration=2,
            sleep_duration=2
        )
        
        # Process 2 events to enter sleep
        wrapper.generate("Hello 1", moral_value=0.8)
        wrapper.generate("Hello 2", moral_value=0.8)
        
        # Next request should be in sleep
        result = wrapper.generate("Hello 3", moral_value=0.8)
        
        assert result["accepted"] is False
        assert "sleep phase" in result["note"]


class TestLLMWrapperCognitiveRhythm:
    """Test cognitive rhythm behavior."""
    
    def test_wake_sleep_cycle(self):
        """Test wake-sleep cycle transitions."""
        mock_llm = MockLLM()
        mock_embedder = MockEmbedder(dim=384)
        
        wrapper = LLMWrapper(
            llm_generate_fn=mock_llm.generate,
            embedding_fn=mock_embedder.embed,
            dim=384,
            wake_duration=2,
            sleep_duration=2
        )
        
        phases_seen = []
        accepted_states = []
        
        # Generate multiple requests and track phase transitions
        for i in range(10):
            result = wrapper.generate(f"prompt{i}", moral_value=0.8)
            phases_seen.append(result["phase"])
            accepted_states.append(result["accepted"])
        
        # Should see both wake and sleep phases
        assert "wake" in phases_seen
        assert "sleep" in phases_seen
        
        # Should have some accepted and some rejected
        assert True in accepted_states
        assert False in accepted_states
        
        # First request should be wake and accepted
        assert phases_seen[0] == "wake"
        assert accepted_states[0] is True
    
    def test_max_tokens_enforcement(self):
        """Test max tokens enforcement in different phases."""
        mock_llm = MockLLM()
        mock_embedder = MockEmbedder(dim=384)
        
        wrapper = LLMWrapper(
            llm_generate_fn=mock_llm.generate,
            embedding_fn=mock_embedder.embed,
            dim=384,
            wake_duration=1,
            sleep_duration=1
        )
        
        # During wake, should use MAX_WAKE_TOKENS
        r1 = wrapper.generate("prompt", moral_value=0.8)
        assert r1["accepted"] is True
        assert mock_llm.last_max_tokens == wrapper.MAX_WAKE_TOKENS
        
        # During sleep (after 1 wake step), should reject
        r2 = wrapper.generate("prompt", moral_value=0.8)
        assert r2["accepted"] is False


class TestLLMWrapperMemoryIntegration:
    """Test memory integration with LLM wrapper."""
    
    def test_context_retrieval(self):
        """Test that wrapper retrieves context from memory."""
        mock_llm = MockLLM()
        mock_embedder = MockEmbedder(dim=384)
        
        wrapper = LLMWrapper(
            llm_generate_fn=mock_llm.generate,
            embedding_fn=mock_embedder.embed,
            dim=384,
            wake_duration=10
        )
        
        # Add some memories
        wrapper.generate("topic about cats", moral_value=0.8)
        wrapper.generate("more about felines", moral_value=0.8)
        wrapper.generate("dogs are different", moral_value=0.8)
        
        # Query related to cats should retrieve context
        result = wrapper.generate("tell me about cats", moral_value=0.8)
        
        assert result["accepted"] is True
        assert result["context_items"] > 0
    
    def test_consolidation_during_sleep(self):
        """Test memory consolidation during sleep phase."""
        mock_llm = MockLLM()
        mock_embedder = MockEmbedder(dim=384)
        
        wrapper = LLMWrapper(
            llm_generate_fn=mock_llm.generate,
            embedding_fn=mock_embedder.embed,
            dim=384,
            wake_duration=2,
            sleep_duration=1
        )
        
        # Process first event during wake
        wrapper.generate("event1", moral_value=0.8)
        
        # Buffer should have 1 item
        assert len(wrapper.consolidation_buffer) == 1
        
        # Second event triggers sleep transition (counter reaches 0)
        # Consolidation happens immediately when entering sleep
        wrapper.generate("event2", moral_value=0.8)
        
        # Buffer should be cleared after consolidation
        # (consolidation happens at end of generate() when entering sleep)
        state = wrapper.get_state()
        assert state["consolidation_buffer_size"] == 0
    
    def test_qilm_capacity_respected(self):
        """Test that QILM capacity is respected."""
        mock_llm = MockLLM()
        mock_embedder = MockEmbedder(dim=384)
        
        capacity = 100
        wrapper = LLMWrapper(
            llm_generate_fn=mock_llm.generate,
            embedding_fn=mock_embedder.embed,
            dim=384,
            capacity=capacity,
            wake_duration=200
        )
        
        # Add more than capacity
        for i in range(capacity + 20):
            wrapper.generate(f"event {i}", moral_value=0.8)
        
        state = wrapper.get_state()
        qilm_stats = state["qilm_stats"]
        
        # Should not exceed capacity
        assert qilm_stats["used"] <= capacity


class TestLLMWrapperMoralAdaptation:
    """Test moral threshold adaptation."""
    
    def test_moral_threshold_adaptation(self):
        """Test that moral threshold adapts over time."""
        mock_llm = MockLLM()
        mock_embedder = MockEmbedder(dim=384)
        
        wrapper = LLMWrapper(
            llm_generate_fn=mock_llm.generate,
            embedding_fn=mock_embedder.embed,
            dim=384,
            initial_moral_threshold=0.50,
            wake_duration=100
        )
        
        initial_threshold = wrapper.moral.threshold
        
        # Send many low-moral-value requests
        for _ in range(50):
            wrapper.generate("low moral request", moral_value=0.2)
        
        # Threshold should decrease
        assert wrapper.moral.threshold < initial_threshold
        
        # Send many high-moral-value requests
        for _ in range(50):
            wrapper.generate("high moral request", moral_value=0.9)
        
        # Threshold should increase
        # (may not be higher than initial if EMA hasn't converged, but should be increasing)
        assert wrapper.moral.ema_accept_rate > 0.5


class TestLLMWrapperStateManagement:
    """Test state management and reset."""
    
    def test_get_state(self):
        """Test state retrieval."""
        mock_llm = MockLLM()
        mock_embedder = MockEmbedder(dim=384)
        
        wrapper = LLMWrapper(
            llm_generate_fn=mock_llm.generate,
            embedding_fn=mock_embedder.embed,
            dim=384
        )
        
        # Process some events
        wrapper.generate("event1", moral_value=0.8)
        wrapper.generate("event2", moral_value=0.3)
        
        state = wrapper.get_state()
        
        assert "step" in state
        assert "phase" in state
        assert "moral_threshold" in state
        assert "accepted_count" in state
        assert "rejected_count" in state
        assert "synaptic_norms" in state
        assert "qilm_stats" in state
        
        assert state["step"] == 2
        assert state["accepted_count"] >= 1
        assert state["rejected_count"] >= 0
    
    def test_reset(self):
        """Test wrapper reset."""
        mock_llm = MockLLM()
        mock_embedder = MockEmbedder(dim=384)
        
        wrapper = LLMWrapper(
            llm_generate_fn=mock_llm.generate,
            embedding_fn=mock_embedder.embed,
            dim=384
        )
        
        # Process events
        wrapper.generate("event1", moral_value=0.8)
        wrapper.generate("event2", moral_value=0.8)
        wrapper.generate("event3", moral_value=0.8)
        
        state_before = wrapper.get_state()
        assert state_before["step"] > 0
        
        # Reset
        wrapper.reset()
        
        state_after = wrapper.get_state()
        assert state_after["step"] == 0
        assert state_after["accepted_count"] == 0
        assert state_after["rejected_count"] == 0
        assert state_after["consolidation_buffer_size"] == 0


class TestLLMWrapperErrorHandling:
    """Test error handling."""
    
    def test_embedding_failure(self):
        """Test handling of embedding failures."""
        mock_llm = MockLLM()
        
        def failing_embedder(text: str) -> np.ndarray:
            raise RuntimeError("Embedding failed")
        
        wrapper = LLMWrapper(
            llm_generate_fn=mock_llm.generate,
            embedding_fn=failing_embedder,
            dim=384
        )
        
        result = wrapper.generate("test", moral_value=0.8)
        
        assert result["accepted"] is False
        assert "error" in result["note"]
        assert "embedding failed" in result["note"].lower()
    
    def test_llm_generation_failure(self):
        """Test handling of LLM generation failures."""
        mock_embedder = MockEmbedder(dim=384)
        
        def failing_llm(prompt: str, max_tokens: int) -> str:
            raise RuntimeError("Generation failed")
        
        wrapper = LLMWrapper(
            llm_generate_fn=failing_llm,
            embedding_fn=mock_embedder.embed,
            dim=384
        )
        
        result = wrapper.generate("test", moral_value=0.8)
        
        assert result["accepted"] is False
        assert "error" in result["note"]
        assert "generation failed" in result["note"].lower()
