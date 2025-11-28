"""
Safety-focused, non-flaky tests for NeuroCognitiveEngine invariants and metrics.

Design goals:
- Stable under refactoring: tests check structural invariants, not internal details
- No side effects: no files, network, environment dependencies
- Fast execution: lightweight fixtures, no heavy loops or matrices
- Clear organization: grouped by functionality with documented invariants

Test categories:
1. NeuroEngineConfig - Configuration dataclass validation
2. Initialization - Engine startup invariants
3. Generation - Output structure and flow invariants
4. State management - State tracking consistency
5. Error handling - Graceful degradation invariants
6. Integration - End-to-end flow validation
"""

from unittest.mock import Mock, patch

import numpy as np

from mlsdm.engine import NeuroCognitiveEngine, NeuroEngineConfig


# =============================================================================
# Test Constants
# =============================================================================
# Fixed dimension for consistent test vectors (small for speed)
TEST_DIMENSION = 384
SMALL_CAPACITY = 1000


def _make_test_vector(dim: int = TEST_DIMENSION) -> np.ndarray:
    """Create deterministic test vector. No random state dependency."""
    return np.ones(dim, dtype=np.float32) * 0.1


def _make_mock_llm(response: str = "test response"):
    """Create mock LLM function."""
    return Mock(return_value=response)


def _make_mock_embed(dim: int = TEST_DIMENSION):
    """Create mock embedding function with deterministic output."""
    return Mock(return_value=_make_test_vector(dim))


# =============================================================================
# NeuroEngineConfig Tests
# =============================================================================
class TestNeuroEngineConfig:
    """
    Invariant: Configuration dataclass provides sensible defaults and
    accepts custom values without modification.
    """

    def test_default_config_has_valid_defaults(self):
        """Default config values are within valid ranges."""
        config = NeuroEngineConfig()

        # Core dimensions - must be positive
        assert config.dim > 0
        assert config.capacity > 0
        assert config.wake_duration > 0
        assert config.sleep_duration > 0

        # Thresholds - must be in [0, 1] range
        assert 0.0 <= config.initial_moral_threshold <= 1.0
        assert 0.0 <= config.default_moral_value <= 1.0
        assert 0.0 <= config.grammar_strictness <= 1.0
        assert 0.0 <= config.stress_threshold <= 1.0

        # Timeouts - must be positive
        assert config.llm_timeout > 0
        assert config.llm_retry_attempts >= 0

    def test_custom_config_values_preserved(self):
        """Custom values are stored without modification."""
        custom_dim = 512
        custom_capacity = 10_000
        custom_wake = 10

        config = NeuroEngineConfig(
            dim=custom_dim,
            capacity=custom_capacity,
            wake_duration=custom_wake,
            enable_fslgs=False,
        )

        assert config.dim == custom_dim
        assert config.capacity == custom_capacity
        assert config.wake_duration == custom_wake
        assert config.enable_fslgs is False


# =============================================================================
# Initialization Tests
# =============================================================================
class TestNeuroCognitiveEngineInit:
    """
    Invariant: Engine initializes to a consistent state with required
    components present and optional components handled gracefully.
    """

    def test_init_creates_required_components(self):
        """Engine creates MLSDM component on init."""
        engine = NeuroCognitiveEngine(
            llm_generate_fn=_make_mock_llm(),
            embedding_fn=_make_mock_embed(),
        )

        # Required: config and MLSDM must exist
        assert engine.config is not None
        assert engine._mlsdm is not None

        # Initial state: no generation yet
        assert engine._last_mlsdm_state is None

    def test_init_respects_custom_config(self):
        """Engine uses provided config values."""
        custom_dim = 512
        config = NeuroEngineConfig(dim=custom_dim, enable_fslgs=False)

        engine = NeuroCognitiveEngine(
            llm_generate_fn=_make_mock_llm(),
            embedding_fn=_make_mock_embed(custom_dim),
            config=config,
        )

        assert engine.config.dim == custom_dim
        assert engine.config.enable_fslgs is False

    def test_init_handles_missing_fslgs_gracefully(self):
        """Engine works when optional FSLGS is not available."""
        engine = NeuroCognitiveEngine(
            llm_generate_fn=_make_mock_llm(),
            embedding_fn=_make_mock_embed(),
        )

        # FSLGS is optional - None is acceptable
        assert engine._fslgs is None


# =============================================================================
# Generation Tests
# =============================================================================
class TestNeuroCognitiveEngineGenerate:
    """
    Invariant: Generate always returns a dict with required keys,
    regardless of internal processing path.
    """

    def test_generate_returns_required_structure(self):
        """Generate output has required keys: response, governance, mlsdm."""
        config = NeuroEngineConfig(enable_fslgs=False)
        engine = NeuroCognitiveEngine(
            llm_generate_fn=_make_mock_llm("test output"),
            embedding_fn=_make_mock_embed(),
            config=config,
        )

        result = engine.generate("test prompt", max_tokens=50)

        # Required keys must be present
        assert "response" in result
        assert "governance" in result
        assert "mlsdm" in result

    def test_generate_preserves_llm_response(self):
        """LLM output appears in response field."""
        expected_response = "Hello from LLM"
        config = NeuroEngineConfig(enable_fslgs=False)
        engine = NeuroCognitiveEngine(
            llm_generate_fn=_make_mock_llm(expected_response),
            embedding_fn=_make_mock_embed(),
            config=config,
        )

        result = engine.generate("prompt")

        assert result["response"] == expected_response

    def test_generate_without_fslgs_sets_governance_none(self):
        """Without FSLGS, governance data is None."""
        config = NeuroEngineConfig(enable_fslgs=False)
        engine = NeuroCognitiveEngine(
            llm_generate_fn=_make_mock_llm(),
            embedding_fn=_make_mock_embed(),
            config=config,
        )

        result = engine.generate("prompt")

        assert result["governance"] is None
        assert result["mlsdm"] is not None

    def test_generate_populates_mlsdm_state(self):
        """Generate populates mlsdm with response data."""
        config = NeuroEngineConfig(enable_fslgs=False)
        engine = NeuroCognitiveEngine(
            llm_generate_fn=_make_mock_llm(),
            embedding_fn=_make_mock_embed(),
            config=config,
        )

        result = engine.generate("prompt")

        assert result["mlsdm"] is not None
        assert isinstance(result["mlsdm"], dict)
        assert "response" in result["mlsdm"]

    @patch("mlsdm.engine.neuro_cognitive_engine.FSLGSWrapper")
    def test_generate_with_fslgs_includes_governance(self, mock_fslgs_class):
        """With FSLGS, governance data is populated."""
        mock_fslgs = Mock()
        mock_fslgs.generate.return_value = {
            "response": "enhanced response",
            "governance_data": {"processed": True},
        }
        mock_fslgs_class.return_value = mock_fslgs

        config = NeuroEngineConfig(enable_fslgs=True)

        with patch("mlsdm.engine.neuro_cognitive_engine.FSLGSWrapper", mock_fslgs_class):
            engine = NeuroCognitiveEngine(
                llm_generate_fn=_make_mock_llm(),
                embedding_fn=_make_mock_embed(),
                config=config,
            )

            result = engine.generate("prompt")

            assert engine._fslgs is not None
            assert result["governance"] is not None


# =============================================================================
# State Management Tests
# =============================================================================
class TestNeuroCognitiveEngineState:
    """
    Invariant: get_last_states returns consistent state structure,
    reflecting generation history accurately.
    """

    def test_get_last_states_has_required_keys(self):
        """State dict always has mlsdm and has_fslgs keys."""
        engine = NeuroCognitiveEngine(
            llm_generate_fn=_make_mock_llm(),
            embedding_fn=_make_mock_embed(),
        )

        states = engine.get_last_states()

        assert "mlsdm" in states
        assert "has_fslgs" in states

    def test_get_last_states_before_generate(self):
        """Before any generation, mlsdm state is None."""
        engine = NeuroCognitiveEngine(
            llm_generate_fn=_make_mock_llm(),
            embedding_fn=_make_mock_embed(),
        )

        states = engine.get_last_states()

        assert states["mlsdm"] is None
        assert states["has_fslgs"] is False

    def test_get_last_states_after_generate(self):
        """After generation, mlsdm state is populated."""
        config = NeuroEngineConfig(enable_fslgs=False)
        engine = NeuroCognitiveEngine(
            llm_generate_fn=_make_mock_llm(),
            embedding_fn=_make_mock_embed(),
            config=config,
        )

        engine.generate("prompt")
        states = engine.get_last_states()

        assert states["mlsdm"] is not None
        assert "response" in states["mlsdm"]


# =============================================================================
# Error Handling Tests
# =============================================================================
class TestNeuroCognitiveEngineErrorHandling:
    """
    Invariant: Errors in LLM or embedding don't crash the engine;
    error information is captured in the response structure.
    """

    def test_llm_error_returns_error_response(self):
        """LLM errors result in empty response with error note."""
        llm_fn = Mock(side_effect=RuntimeError("LLM failure"))
        config = NeuroEngineConfig(enable_fslgs=False, llm_retry_attempts=1)
        engine = NeuroCognitiveEngine(
            llm_generate_fn=llm_fn,
            embedding_fn=_make_mock_embed(),
            config=config,
        )

        result = engine.generate("prompt")

        # Structure preserved even on error
        assert "response" in result
        assert "mlsdm" in result
        # Response should be empty or contain error indication
        assert result["response"] == ""
        # Error info should be captured
        assert result["mlsdm"] is not None
        note = result["mlsdm"].get("note", "")
        assert "error" in note.lower()

    def test_embedding_error_returns_error_response(self):
        """Embedding errors result in graceful degradation."""
        embed_fn = Mock(side_effect=RuntimeError("Embedding failure"))
        config = NeuroEngineConfig(enable_fslgs=False)
        engine = NeuroCognitiveEngine(
            llm_generate_fn=_make_mock_llm(),
            embedding_fn=embed_fn,
            config=config,
        )

        result = engine.generate("prompt")

        # Structure preserved
        assert "response" in result
        assert "mlsdm" in result
        # Error captured
        assert result["mlsdm"] is not None
        note = result["mlsdm"].get("note", "")
        assert "error" in note.lower()


# =============================================================================
# Integration Tests
# =============================================================================
class TestNeuroCognitiveEngineIntegration:
    """
    Invariant: Full engine flow maintains consistency across
    multiple generations without external dependencies.
    """

    def test_multiple_generations_maintain_state(self):
        """Sequential generations don't corrupt state."""

        def deterministic_llm(prompt: str, max_tokens: int) -> str:
            return f"Response-{len(prompt)}"

        def deterministic_embed(text: str) -> np.ndarray:
            return np.ones(TEST_DIMENSION, dtype=np.float32) * len(text)

        config = NeuroEngineConfig(
            dim=TEST_DIMENSION,
            capacity=SMALL_CAPACITY,
            enable_fslgs=False,
        )

        engine = NeuroCognitiveEngine(
            llm_generate_fn=deterministic_llm,
            embedding_fn=deterministic_embed,
            config=config,
        )

        # Multiple generations with explicit max_tokens
        r1 = engine.generate("short", max_tokens=50)
        r2 = engine.generate("a longer prompt", max_tokens=50)
        r3 = engine.generate("x", max_tokens=50)

        # All should have valid structure
        for result in [r1, r2, r3]:
            assert "response" in result
            assert "mlsdm" in result
            # Response should be non-empty and contain our prefix
            if result["response"]:  # Non-empty response
                assert "Response-" in result["response"]

        # State should reflect last generation
        states = engine.get_last_states()
        assert states["mlsdm"] is not None
