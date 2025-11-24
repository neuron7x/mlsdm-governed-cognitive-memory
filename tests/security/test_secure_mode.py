"""
Security tests for MLSDM secure mode functionality.

This test suite validates that when MLSDM_SECURE_MODE is enabled:
- NeuroLang training and checkpoint loading are disabled
- Aphasia repair is disabled (detection only)
- The system operates in a security-hardened mode
"""

import os
from unittest.mock import patch

import numpy as np
import pytest

from mlsdm.extensions.neuro_lang_extension import (
    NeuroLangWrapper,
    is_secure_mode_enabled,
)


def dummy_llm(prompt: str, max_tokens: int) -> str:
    """Dummy LLM for testing."""
    return "This is a test response with proper grammar and function words."


def dummy_embedder(text: str):
    """Generate deterministic embeddings for testing."""
    vec = np.ones(384, dtype=np.float32)
    return vec / np.linalg.norm(vec)


@pytest.mark.security
def test_is_secure_mode_enabled_returns_true_when_env_is_1():
    """Test that secure mode is detected when MLSDM_SECURE_MODE=1."""
    with patch.dict(os.environ, {"MLSDM_SECURE_MODE": "1"}):
        assert is_secure_mode_enabled() is True


@pytest.mark.security
def test_is_secure_mode_enabled_returns_true_when_env_is_true():
    """Test that secure mode is detected when MLSDM_SECURE_MODE=true."""
    with patch.dict(os.environ, {"MLSDM_SECURE_MODE": "true"}):
        assert is_secure_mode_enabled() is True


@pytest.mark.security
def test_is_secure_mode_enabled_returns_true_when_env_is_TRUE():
    """Test that secure mode is detected when MLSDM_SECURE_MODE=TRUE."""
    with patch.dict(os.environ, {"MLSDM_SECURE_MODE": "TRUE"}):
        assert is_secure_mode_enabled() is True


@pytest.mark.security
def test_is_secure_mode_enabled_returns_false_by_default():
    """Test that secure mode is disabled by default."""
    with patch.dict(os.environ, {}, clear=True):
        assert is_secure_mode_enabled() is False


@pytest.mark.security
def test_is_secure_mode_enabled_returns_false_when_env_is_0():
    """Test that secure mode is disabled when MLSDM_SECURE_MODE=0."""
    with patch.dict(os.environ, {"MLSDM_SECURE_MODE": "0"}):
        assert is_secure_mode_enabled() is False


@pytest.mark.security
def test_secure_mode_forces_neurolang_disabled():
    """Test that secure mode forces neurolang_mode to 'disabled'."""
    with patch.dict(os.environ, {"MLSDM_SECURE_MODE": "1"}):
        wrapper = NeuroLangWrapper(
            llm_generate_fn=dummy_llm,
            embedding_fn=dummy_embedder,
            dim=384,
            capacity=256,
            neurolang_mode="eager_train",  # Try to enable training
        )

        # Verify that secure mode overrode the setting
        assert wrapper.neurolang_mode == "disabled"
        assert wrapper.actor is None
        assert wrapper.critic is None
        assert wrapper.trainer is None


@pytest.mark.security
def test_secure_mode_ignores_checkpoint_path():
    """Test that secure mode ignores checkpoint_path even if provided."""
    with patch.dict(os.environ, {"MLSDM_SECURE_MODE": "1"}):
        wrapper = NeuroLangWrapper(
            llm_generate_fn=dummy_llm,
            embedding_fn=dummy_embedder,
            dim=384,
            capacity=256,
            neurolang_mode="eager_train",
            neurolang_checkpoint_path="config/neurolang_grammar.pt",
        )

        # Verify neurolang is disabled
        assert wrapper.neurolang_mode == "disabled"
        assert wrapper.actor is None
        assert wrapper.critic is None


@pytest.mark.security
def test_secure_mode_disables_aphasia_repair():
    """Test that secure mode disables aphasia repair (detection only)."""
    with patch.dict(os.environ, {"MLSDM_SECURE_MODE": "1"}):
        wrapper = NeuroLangWrapper(
            llm_generate_fn=dummy_llm,
            embedding_fn=dummy_embedder,
            dim=384,
            capacity=256,
            aphasia_repair_enabled=True,  # Try to enable repair
        )

        # Verify that secure mode disabled repair
        assert wrapper.aphasia_repair_enabled is False
        # Detection should still work
        assert wrapper.aphasia_detect_enabled is True


@pytest.mark.security
def test_secure_mode_generate_works_without_training():
    """Test that generate() works in secure mode without attempting to train."""
    with patch.dict(os.environ, {"MLSDM_SECURE_MODE": "1"}):
        wrapper = NeuroLangWrapper(
            llm_generate_fn=dummy_llm,
            embedding_fn=dummy_embedder,
            dim=384,
            capacity=256,
            neurolang_mode="eager_train",
        )

        # Should not raise any errors
        result = wrapper.generate(
            prompt="Test secure mode generation",
            moral_value=0.7,
            max_tokens=50
        )

        assert result is not None
        assert "response" in result
        assert result["accepted"] is True
        # Neuro enhancement should indicate disabled state
        assert "disabled" in result["neuro_enhancement"].lower()


@pytest.mark.security
def test_secure_mode_preserves_explicit_detect_disabled():
    """Test that if detection is explicitly disabled, secure mode respects it."""
    with patch.dict(os.environ, {"MLSDM_SECURE_MODE": "1"}):
        wrapper = NeuroLangWrapper(
            llm_generate_fn=dummy_llm,
            embedding_fn=dummy_embedder,
            dim=384,
            capacity=256,
            aphasia_detect_enabled=False,  # Explicitly disable detection
            aphasia_repair_enabled=True,
        )

        # Detection should remain disabled
        assert wrapper.aphasia_detect_enabled is False
        # Repair should be disabled by secure mode
        assert wrapper.aphasia_repair_enabled is False


@pytest.mark.security
def test_without_secure_mode_training_works_normally():
    """Test that without secure mode, normal training/checkpoint loading works."""
    with patch.dict(os.environ, {"MLSDM_SECURE_MODE": "0"}):
        wrapper = NeuroLangWrapper(
            llm_generate_fn=dummy_llm,
            embedding_fn=dummy_embedder,
            dim=384,
            capacity=256,
            neurolang_mode="eager_train",
            aphasia_repair_enabled=True,
        )

        # Normal mode should allow training and repair
        assert wrapper.neurolang_mode == "eager_train"
        assert wrapper.aphasia_repair_enabled is True
        assert wrapper.actor is not None
        assert wrapper.critic is not None
