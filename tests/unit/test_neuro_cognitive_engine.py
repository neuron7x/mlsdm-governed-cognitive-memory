"""
Unit tests for NeuroCognitiveEngine.

Tests cover:
1. Basic initialization with default config
2. Custom configuration
3. Generate method without FSLGS (fallback mode)
4. Mock FSLGS integration
5. get_last_states method
6. Error handling scenarios
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from mlsdm.engine import NeuroCognitiveEngine, NeuroEngineConfig


class TestNeuroEngineConfig:
    """Test NeuroEngineConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = NeuroEngineConfig()

        # MLSDM defaults
        assert config.dim == 384
        assert config.capacity == 20_000
        assert config.wake_duration == 8
        assert config.sleep_duration == 3
        assert config.initial_moral_threshold == 0.50
        assert config.llm_timeout == 30.0
        assert config.llm_retry_attempts == 3

        # FSLGS defaults
        assert config.enable_fslgs is True
        assert config.enable_universal_grammar is True
        assert config.grammar_strictness == 0.9
        assert config.association_threshold == 0.65
        assert config.enable_monitoring is True
        assert config.stress_threshold == 0.7
        assert config.fslgs_fractal_levels is None
        assert config.fslgs_memory_capacity == 0
        assert config.enable_entity_tracking is True
        assert config.enable_temporal_validation is True
        assert config.enable_causal_checking is True

        # Runtime defaults
        assert config.default_moral_value == 0.5
        assert config.default_context_top_k == 5
        assert config.default_cognitive_load == 0.5
        assert config.default_user_intent == "conversational"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = NeuroEngineConfig(
            dim=512,
            capacity=10_000,
            wake_duration=10,
            enable_fslgs=False,
            default_moral_value=0.7,
        )

        assert config.dim == 512
        assert config.capacity == 10_000
        assert config.wake_duration == 10
        assert config.enable_fslgs is False
        assert config.default_moral_value == 0.7


class TestNeuroCognitiveEngineInit:
    """Test NeuroCognitiveEngine initialization."""

    def test_init_with_default_config(self):
        """Test initialization with default config."""
        llm_fn = Mock(return_value="response")
        embed_fn = Mock(return_value=np.random.randn(384))

        engine = NeuroCognitiveEngine(
            llm_generate_fn=llm_fn,
            embedding_fn=embed_fn,
        )

        assert engine.config is not None
        assert engine.config.dim == 384
        assert engine._mlsdm is not None
        assert engine._last_mlsdm_state is None
        # FSLGS will be None since it's not installed
        assert engine._fslgs is None

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        llm_fn = Mock(return_value="response")
        embed_fn = Mock(return_value=np.random.randn(512))

        config = NeuroEngineConfig(
            dim=512,
            capacity=5_000,
            enable_fslgs=False,
        )

        engine = NeuroCognitiveEngine(
            llm_generate_fn=llm_fn,
            embedding_fn=embed_fn,
            config=config,
        )

        assert engine.config.dim == 512
        assert engine.config.capacity == 5_000
        assert engine.config.enable_fslgs is False
        assert engine._fslgs is None

    def test_init_without_fslgs_installed(self):
        """Test that engine works when FSLGS is not installed."""
        llm_fn = Mock(return_value="response")
        embed_fn = Mock(return_value=np.random.randn(384))

        # FSLGS should be None by default (not installed in test environment)
        engine = NeuroCognitiveEngine(
            llm_generate_fn=llm_fn,
            embedding_fn=embed_fn,
        )

        assert engine._fslgs is None


class TestNeuroCognitiveEngineGenerate:
    """Test NeuroCognitiveEngine.generate method."""

    def test_generate_without_fslgs(self):
        """Test generate method when FSLGS is not available (fallback mode)."""
        llm_fn = Mock(return_value="Hello, world!")
        embed_fn = Mock(return_value=np.random.randn(384))

        config = NeuroEngineConfig(enable_fslgs=False)
        engine = NeuroCognitiveEngine(
            llm_generate_fn=llm_fn,
            embedding_fn=embed_fn,
            config=config,
        )

        result = engine.generate("Test prompt", max_tokens=128)

        # Verify structure
        assert "response" in result
        assert "governance" in result
        assert "mlsdm" in result

        # Verify values
        assert result["response"] == "Hello, world!"
        assert result["governance"] is None
        assert result["mlsdm"] is not None
        assert "response" in result["mlsdm"]

    def test_generate_with_custom_parameters(self):
        """Test generate with custom moral_value and context_top_k."""
        llm_fn = Mock(return_value="Custom response")
        embed_fn = Mock(return_value=np.random.randn(384))

        config = NeuroEngineConfig(enable_fslgs=False)
        engine = NeuroCognitiveEngine(
            llm_generate_fn=llm_fn,
            embedding_fn=embed_fn,
            config=config,
        )

        result = engine.generate(
            "Test prompt",
            max_tokens=256,
            moral_value=0.8,
            context_top_k=10,
        )

        assert result["response"] == "Custom response"
        assert result["mlsdm"] is not None

    def test_generate_uses_default_parameters(self):
        """Test that generate uses config defaults when parameters not provided."""
        llm_fn = Mock(return_value="Default response")
        embed_fn = Mock(return_value=np.random.randn(384))

        config = NeuroEngineConfig(
            enable_fslgs=False,
            default_moral_value=0.6,
            default_context_top_k=7,
            default_user_intent="testing",
        )
        engine = NeuroCognitiveEngine(
            llm_generate_fn=llm_fn,
            embedding_fn=embed_fn,
            config=config,
        )

        result = engine.generate("Test prompt")

        assert result["response"] == "Default response"
        # Verify defaults were used (implicitly through successful generation)
        assert result["mlsdm"] is not None

    @patch("mlsdm.engine.neuro_cognitive_engine.FSLGSWrapper")
    def test_generate_with_fslgs_mock(self, mock_fslgs_class):
        """Test generate method with mocked FSLGS integration."""
        llm_fn = Mock(return_value="MLSDM response")
        embed_fn = Mock(return_value=np.random.randn(384))

        # Mock FSLGS instance and its generate method
        mock_fslgs_instance = Mock()
        mock_fslgs_instance.generate.return_value = {
            "response": "FSLGS enhanced response",
            "governance_data": {"dual_stream": "processed"},
        }
        mock_fslgs_class.return_value = mock_fslgs_instance

        config = NeuroEngineConfig(enable_fslgs=True)

        # Patch FSLGSWrapper at module level to simulate it being available
        with patch("mlsdm.engine.neuro_cognitive_engine.FSLGSWrapper", mock_fslgs_class):
            engine = NeuroCognitiveEngine(
                llm_generate_fn=llm_fn,
                embedding_fn=embed_fn,
                config=config,
            )

            # FSLGS should be initialized
            assert engine._fslgs is not None

            result = engine.generate("Test prompt", max_tokens=128)

            # Verify FSLGS was called
            mock_fslgs_instance.generate.assert_called_once()

            # Verify result structure
            assert result["response"] == "FSLGS enhanced response"
            assert result["governance"] is not None
            # mlsdm state may be None if governed_llm wasn't actually called by the mock
            assert "mlsdm" in result


class TestNeuroCognitiveEngineState:
    """Test NeuroCognitiveEngine state management."""

    def test_get_last_states_initial(self):
        """Test get_last_states returns correct initial state."""
        llm_fn = Mock(return_value="response")
        embed_fn = Mock(return_value=np.random.randn(384))

        engine = NeuroCognitiveEngine(
            llm_generate_fn=llm_fn,
            embedding_fn=embed_fn,
        )

        states = engine.get_last_states()

        assert "mlsdm" in states
        assert "has_fslgs" in states
        assert states["mlsdm"] is None  # No generation yet
        assert states["has_fslgs"] is False  # FSLGS not installed

    def test_get_last_states_after_generate(self):
        """Test get_last_states after a generation."""
        llm_fn = Mock(return_value="Test response")
        embed_fn = Mock(return_value=np.random.randn(384))

        config = NeuroEngineConfig(enable_fslgs=False)
        engine = NeuroCognitiveEngine(
            llm_generate_fn=llm_fn,
            embedding_fn=embed_fn,
            config=config,
        )

        # Generate to populate state
        engine.generate("Test prompt")

        states = engine.get_last_states()

        assert states["mlsdm"] is not None
        assert "response" in states["mlsdm"]
        assert states["has_fslgs"] is False


class TestNeuroCognitiveEngineErrorHandling:
    """Test error handling in NeuroCognitiveEngine."""

    def test_llm_error_propagates(self):
        """Test that LLM errors are handled and returned in response."""
        llm_fn = Mock(side_effect=RuntimeError("LLM error"))
        embed_fn = Mock(return_value=np.random.randn(384))

        config = NeuroEngineConfig(enable_fslgs=False, llm_retry_attempts=1)
        engine = NeuroCognitiveEngine(
            llm_generate_fn=llm_fn,
            embedding_fn=embed_fn,
            config=config,
        )

        # LLMWrapper catches errors and returns error response
        result = engine.generate("Test prompt")

        # Verify error is captured in response
        assert result["response"] == ""
        assert result["mlsdm"] is not None
        assert "error" in result["mlsdm"].get("note", "").lower()

    def test_embedding_error_handling(self):
        """Test that embedding errors are handled gracefully."""
        llm_fn = Mock(return_value="Response")
        embed_fn = Mock(side_effect=RuntimeError("Embedding error"))

        config = NeuroEngineConfig(enable_fslgs=False)
        engine = NeuroCognitiveEngine(
            llm_generate_fn=llm_fn,
            embedding_fn=embed_fn,
            config=config,
        )

        # The circuit breaker catches embedding errors and returns error response
        result = engine.generate("Test prompt")

        # Verify error response structure
        assert result["response"] == ""
        assert result["mlsdm"] is not None
        assert "error" in result["mlsdm"].get("note", "").lower()


class TestNeuroCognitiveEngineIntegration:
    """Integration-level tests for NeuroCognitiveEngine."""

    def test_end_to_end_without_fslgs(self):
        """Test complete flow without FSLGS."""

        def simple_llm(prompt: str, max_tokens: int) -> str:
            return f"Response to: {prompt[:20]}"

        def simple_embed(text: str) -> np.ndarray:
            # Simple deterministic embedding for testing
            return np.ones(384) * len(text)

        config = NeuroEngineConfig(
            dim=384,
            capacity=1000,
            enable_fslgs=False,
        )

        engine = NeuroCognitiveEngine(
            llm_generate_fn=simple_llm,
            embedding_fn=simple_embed,
            config=config,
        )

        # First generation
        result1 = engine.generate("Hello, how are you?", max_tokens=50)

        assert "response" in result1
        assert result1["response"].startswith("Response to:")
        assert result1["governance"] is None
        assert result1["mlsdm"] is not None

        # Second generation (should maintain state)
        result2 = engine.generate("What is the weather?", max_tokens=50)

        assert "response" in result2
        assert result2["mlsdm"] is not None

        # Verify state is tracked
        states = engine.get_last_states()
        assert states["mlsdm"] is not None
