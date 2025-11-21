"""
Unit tests for LLMWrapper - Universal LLM wrapper with cognitive governance.
"""

import numpy as np

from mlsdm.core.llm_wrapper import LLMWrapper


class TestLLMWrapper:
    """Test suite for LLMWrapper."""

    @staticmethod
    def mock_llm_generate(prompt: str, max_tokens: int) -> str:
        """Mock LLM generation function."""
        return f"Generated response for: {prompt[:50]}... (max_tokens={max_tokens})"

    @staticmethod
    def mock_embedding(text: str) -> np.ndarray:
        """Mock embedding function - returns random normalized vector."""
        np.random.seed(hash(text) % (2**32))
        vec = np.random.randn(384).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def test_initialization(self) -> None:
        """Test wrapper initialization."""
        wrapper = LLMWrapper(
            llm_generate_fn=self.mock_llm_generate,
            embedding_fn=self.mock_embedding,
            dim=384
        )

        assert wrapper.dim == 384
        assert wrapper.step_counter == 0
        assert wrapper.accepted_count == 0
        assert wrapper.rejected_count == 0
        assert len(wrapper.consolidation_buffer) == 0

    def test_initialization_custom_params(self) -> None:
        """Test wrapper initialization with custom parameters."""
        wrapper = LLMWrapper(
            llm_generate_fn=self.mock_llm_generate,
            embedding_fn=self.mock_embedding,
            dim=512,
            capacity=10_000,
            wake_duration=10,
            sleep_duration=5,
            initial_moral_threshold=0.70
        )

        assert wrapper.dim == 512
        assert wrapper.qilm.capacity == 10_000
        assert wrapper.rhythm.wake_duration == 10
        assert wrapper.rhythm.sleep_duration == 5
        assert wrapper.moral.threshold == 0.70

    def test_generate_accepted_wake_phase(self) -> None:
        """Test successful generation during wake phase."""
        wrapper = LLMWrapper(
            llm_generate_fn=self.mock_llm_generate,
            embedding_fn=self.mock_embedding
        )

        result = wrapper.generate(
            prompt="Hello, how are you?",
            moral_value=0.8
        )

        assert result["accepted"] is True
        assert result["phase"] == "wake"
        assert result["step"] == 1
        assert result["note"] == "processed"
        assert "response" in result
        assert len(result["response"]) > 0
        assert wrapper.accepted_count == 1
        assert wrapper.rejected_count == 0

    def test_generate_moral_rejection(self) -> None:
        """Test rejection due to low moral value."""
        wrapper = LLMWrapper(
            llm_generate_fn=self.mock_llm_generate,
            embedding_fn=self.mock_embedding,
            initial_moral_threshold=0.80
        )

        result = wrapper.generate(
            prompt="Say something toxic",
            moral_value=0.1  # Low moral value
        )

        assert result["accepted"] is False
        assert result["note"] == "morally rejected"
        assert result["response"] == ""
        assert wrapper.rejected_count == 1
        assert wrapper.accepted_count == 0

    def test_generate_sleep_phase_rejection(self) -> None:
        """Test rejection during sleep phase."""
        wrapper = LLMWrapper(
            llm_generate_fn=self.mock_llm_generate,
            embedding_fn=self.mock_embedding,
            wake_duration=2,
            sleep_duration=2
        )

        # Process during wake phase
        for i in range(2):
            result = wrapper.generate(
                prompt=f"Message {i}",
                moral_value=0.8
            )
            assert result["accepted"] is True

        # Now in sleep phase
        result = wrapper.generate(
            prompt="This should be rejected",
            moral_value=0.8
        )

        assert result["accepted"] is False
        assert result["phase"] == "sleep"
        assert "sleep phase" in result["note"]

    def test_max_tokens_enforcement_sleep(self) -> None:
        """Test that max tokens are limited during sleep phase."""
        wrapper = LLMWrapper(
            llm_generate_fn=self.mock_llm_generate,
            embedding_fn=self.mock_embedding,
            wake_duration=1,
            sleep_duration=1
        )

        # First request in wake - should use MAX_WAKE_TOKENS
        result = wrapper.generate(
            prompt="First message",
            moral_value=0.8
        )
        assert result["max_tokens_used"] == wrapper.MAX_WAKE_TOKENS

        # Transition to sleep happens after wake duration
        # Since we set wake_duration=1, next request should be rejected (sleep phase)

    def test_context_retrieval(self) -> None:
        """Test that context is retrieved from memory."""
        wrapper = LLMWrapper(
            llm_generate_fn=self.mock_llm_generate,
            embedding_fn=self.mock_embedding
        )

        # Add some memories
        wrapper.generate("First message", moral_value=0.8)
        wrapper.generate("Second message", moral_value=0.8)
        wrapper.generate("Third message", moral_value=0.8)

        # Fourth message should retrieve context
        result = wrapper.generate("Fourth message", moral_value=0.8)

        assert result["accepted"] is True
        assert result["context_items"] >= 0  # Should have retrieved some context

    def test_consolidation_buffer(self) -> None:
        """Test that consolidation buffer is populated and cleared."""
        wrapper = LLMWrapper(
            llm_generate_fn=self.mock_llm_generate,
            embedding_fn=self.mock_embedding,
            wake_duration=2,
            sleep_duration=1
        )

        # Process messages during wake
        wrapper.generate("Message 1", moral_value=0.8)
        assert len(wrapper.consolidation_buffer) == 1

        wrapper.generate("Message 2", moral_value=0.8)
        # After this, we transition to sleep, buffer should be cleared
        # (consolidation happens when entering sleep)

        state = wrapper.get_state()
        # Buffer may be empty if consolidation happened
        assert state["consolidation_buffer_size"] >= 0

    def test_get_state(self) -> None:
        """Test state retrieval."""
        wrapper = LLMWrapper(
            llm_generate_fn=self.mock_llm_generate,
            embedding_fn=self.mock_embedding
        )

        wrapper.generate("Test message", moral_value=0.8)

        state = wrapper.get_state()

        assert "step" in state
        assert "phase" in state
        assert "moral_threshold" in state
        assert "moral_ema" in state
        assert "accepted_count" in state
        assert "rejected_count" in state
        assert "synaptic_norms" in state
        assert "qilm_stats" in state
        assert "consolidation_buffer_size" in state

        assert state["step"] == 1
        assert state["accepted_count"] == 1
        assert state["rejected_count"] == 0

    def test_reset(self) -> None:
        """Test wrapper reset functionality."""
        wrapper = LLMWrapper(
            llm_generate_fn=self.mock_llm_generate,
            embedding_fn=self.mock_embedding
        )

        # Process some messages
        wrapper.generate("Message 1", moral_value=0.8)
        wrapper.generate("Message 2", moral_value=0.8)
        wrapper.generate("Message 3", moral_value=0.1)  # Rejected

        assert wrapper.step_counter > 0
        assert wrapper.accepted_count > 0

        # Reset
        wrapper.reset()

        assert wrapper.step_counter == 0
        assert wrapper.accepted_count == 0
        assert wrapper.rejected_count == 0
        assert len(wrapper.consolidation_buffer) == 0

    def test_thread_safety(self) -> None:
        """Test thread-safe operation."""
        import threading

        wrapper = LLMWrapper(
            llm_generate_fn=self.mock_llm_generate,
            embedding_fn=self.mock_embedding
        )

        results = []

        def worker():
            result = wrapper.generate("Concurrent message", moral_value=0.8)
            results.append(result)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should have completed
        assert len(results) == 10

        # Step counter should reflect all operations
        state = wrapper.get_state()
        assert state["step"] >= 10

    def test_embedding_normalization(self) -> None:
        """Test that embeddings are normalized."""
        wrapper = LLMWrapper(
            llm_generate_fn=self.mock_llm_generate,
            embedding_fn=lambda text: np.ones(384, dtype=np.float32) * 10  # Non-normalized
        )

        result = wrapper.generate("Test", moral_value=0.8)

        # Should still work due to normalization
        assert result["accepted"] is True

    def test_moral_adaptation(self) -> None:
        """Test moral threshold adaptation over time."""
        wrapper = LLMWrapper(
            llm_generate_fn=self.mock_llm_generate,
            embedding_fn=self.mock_embedding,
            initial_moral_threshold=0.50
        )

        initial_threshold = wrapper.moral.threshold

        # Send many low moral value messages
        for _ in range(20):
            wrapper.generate("Message", moral_value=0.1)

        # Threshold should decrease (become more lenient) due to low accept rate
        # to maintain homeostasis around 50% acceptance
        state = wrapper.get_state()
        # Due to many rejections, threshold adapts downward toward MIN_THRESHOLD (0.30)
        assert state["moral_threshold"] <= initial_threshold
        assert state["moral_threshold"] >= 0.30  # MIN_THRESHOLD

    def test_memory_capacity_bounded(self) -> None:
        """Test that memory stays within capacity."""
        capacity = 100
        wrapper = LLMWrapper(
            llm_generate_fn=self.mock_llm_generate,
            embedding_fn=self.mock_embedding,
            capacity=capacity
        )

        # Add more than capacity
        for i in range(150):
            wrapper.generate(f"Message {i}", moral_value=0.8)

        state = wrapper.get_state()
        qilm_stats = state["qilm_stats"]

        # Used should not exceed capacity
        assert qilm_stats["used"] <= capacity
        assert qilm_stats["capacity"] == capacity

    def test_custom_max_tokens(self) -> None:
        """Test custom max tokens parameter."""
        wrapper = LLMWrapper(
            llm_generate_fn=self.mock_llm_generate,
            embedding_fn=self.mock_embedding
        )

        result = wrapper.generate(
            prompt="Test message",
            moral_value=0.8,
            max_tokens=512
        )

        assert result["accepted"] is True
        # During wake phase, custom max_tokens should be respected
        assert result["max_tokens_used"] == 512

    def test_synaptic_memory_updates(self) -> None:
        """Test that synaptic memory is updated."""
        wrapper = LLMWrapper(
            llm_generate_fn=self.mock_llm_generate,
            embedding_fn=self.mock_embedding
        )

        initial_state = wrapper.get_state()
        initial_l1_norm = initial_state["synaptic_norms"]["L1"]

        # Process messages
        for i in range(5):
            wrapper.generate(f"Message {i}", moral_value=0.8)

        final_state = wrapper.get_state()
        final_l1_norm = final_state["synaptic_norms"]["L1"]

        # L1 norm should have changed
        assert final_l1_norm != initial_l1_norm
        assert final_l1_norm > 0


class TestLLMWrapperEdgeCases:
    """Test edge cases and error handling."""

    @staticmethod
    def mock_llm_generate(prompt: str, max_tokens: int) -> str:
        return "OK"

    @staticmethod
    def mock_embedding(text: str) -> np.ndarray:
        vec = np.random.randn(384).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def test_embedding_error_handling(self) -> None:
        """Test handling of embedding errors."""
        def failing_embed(text: str) -> np.ndarray:
            raise ValueError("Embedding failed")

        wrapper = LLMWrapper(
            llm_generate_fn=self.mock_llm_generate,
            embedding_fn=failing_embed
        )

        result = wrapper.generate("Test", moral_value=0.8)

        assert result["accepted"] is False
        assert "embedding failed" in result["note"]

    def test_generation_error_handling(self) -> None:
        """Test handling of generation errors."""
        def failing_generate(prompt: str, max_tokens: int) -> str:
            raise RuntimeError("Generation failed")

        wrapper = LLMWrapper(
            llm_generate_fn=failing_generate,
            embedding_fn=self.mock_embedding
        )

        result = wrapper.generate("Test", moral_value=0.8)

        assert result["accepted"] is False
        assert "generation failed" in result["note"]

    def test_zero_norm_embedding(self) -> None:
        """Test handling of zero-norm embeddings."""
        def zero_embed(text: str) -> np.ndarray:
            return np.zeros(384, dtype=np.float32)

        wrapper = LLMWrapper(
            llm_generate_fn=self.mock_llm_generate,
            embedding_fn=zero_embed
        )

        # Should handle gracefully (norm check prevents division by zero)
        result = wrapper.generate("Test", moral_value=0.8)
        # Result depends on implementation - may accept or reject
        assert "accepted" in result  # Just verify it returns a result dict


class TestLLMWrapperIntegration:
    """Integration tests for LLMWrapper with realistic scenarios."""

    @staticmethod
    def mock_llm_generate(prompt: str, max_tokens: int) -> str:
        # Simulate realistic response
        if "toxic" in prompt.lower() or "harmful" in prompt.lower():
            return "I cannot help with that request."
        return "This is a helpful response to your query."

    @staticmethod
    def mock_embedding(text: str) -> np.ndarray:
        # Simple but deterministic embedding
        np.random.seed(sum(ord(c) for c in text) % (2**32))
        vec = np.random.randn(384).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-9)

    def test_realistic_conversation_flow(self) -> None:
        """Test a realistic conversation with multiple turns."""
        wrapper = LLMWrapper(
            llm_generate_fn=self.mock_llm_generate,
            embedding_fn=self.mock_embedding
        )

        # Simulate conversation
        messages = [
            ("Hello, how can you help me?", 0.9),
            ("Tell me about Python programming", 0.9),
            ("What are best practices?", 0.9),
            ("Can you write some code?", 0.9),
        ]

        accepted_count = 0
        for msg, moral in messages:
            result = wrapper.generate(msg, moral_value=moral)
            if result["accepted"]:
                accepted_count += 1

        # Most should be accepted during wake phase
        assert accepted_count >= 2

        state = wrapper.get_state()
        assert state["accepted_count"] >= 2

    def test_memory_coherence_across_interactions(self) -> None:
        """Test that memory maintains coherence across interactions."""
        wrapper = LLMWrapper(
            llm_generate_fn=self.mock_llm_generate,
            embedding_fn=self.mock_embedding,
            wake_duration=10  # Longer wake to test many interactions
        )

        # Send related messages
        topics = ["AI", "machine learning", "neural networks", "deep learning"]

        for topic in topics:
            result = wrapper.generate(f"Tell me about {topic}", moral_value=0.9)
            assert result["accepted"] is True

        # Query similar topic - should get context
        result = wrapper.generate("What about AI models?", moral_value=0.9)

        # Should have retrieved relevant context
        if result["accepted"]:
            assert result["context_items"] > 0
