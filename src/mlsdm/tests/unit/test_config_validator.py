"""
Tests for Configuration Validator

Author: neuron7x
License: MIT
"""

import pytest

from mlsdm.utils.config_validator import ConfigValidator, ValidationError, validate_config


class TestConfigValidator:
    """Test suite for ConfigValidator."""

    def test_validate_dimension_valid(self):
        """Test valid dimension validation."""
        assert ConfigValidator.validate_dimension(384) == 384
        assert ConfigValidator.validate_dimension(768) == 768
        assert ConfigValidator.validate_dimension(1) == 1

    def test_validate_dimension_invalid_type(self):
        """Test dimension validation with invalid type."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_dimension("384")
        assert "dim" in str(exc_info.value)
        assert "positive integer" in str(exc_info.value)

    def test_validate_dimension_negative(self):
        """Test dimension validation with negative value."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_dimension(-1)

    def test_validate_dimension_zero(self):
        """Test dimension validation with zero."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_dimension(0)

    def test_validate_dimension_too_large(self):
        """Test dimension validation with too large value."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_dimension(100000)

    def test_validate_capacity_valid(self):
        """Test valid capacity validation."""
        assert ConfigValidator.validate_capacity(20000) == 20000
        assert ConfigValidator.validate_capacity(1000) == 1000

    def test_validate_capacity_invalid_type(self):
        """Test capacity validation with invalid type."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_capacity("20000")

    def test_validate_capacity_negative(self):
        """Test capacity validation with negative value."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_capacity(-100)

    def test_validate_capacity_too_large(self):
        """Test capacity validation with too large value."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_capacity(2_000_000)

    def test_validate_threshold_valid(self):
        """Test valid threshold validation."""
        assert ConfigValidator.validate_threshold(0.5) == 0.5
        assert ConfigValidator.validate_threshold(0.0) == 0.0
        assert ConfigValidator.validate_threshold(1.0) == 1.0

    def test_validate_threshold_invalid_type(self):
        """Test threshold validation with invalid type."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_threshold("0.5")

    def test_validate_threshold_out_of_range_low(self):
        """Test threshold validation with value too low."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_threshold(-0.1)

    def test_validate_threshold_out_of_range_high(self):
        """Test threshold validation with value too high."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_threshold(1.5)

    def test_validate_threshold_custom_range(self):
        """Test threshold validation with custom range."""
        assert ConfigValidator.validate_threshold(0.5, 0.3, 0.9) == 0.5

        with pytest.raises(ValidationError):
            ConfigValidator.validate_threshold(0.2, 0.3, 0.9)

        with pytest.raises(ValidationError):
            ConfigValidator.validate_threshold(1.0, 0.3, 0.9)

    def test_validate_duration_valid(self):
        """Test valid duration validation."""
        assert ConfigValidator.validate_duration(8) == 8
        assert ConfigValidator.validate_duration(1) == 1

    def test_validate_duration_invalid_type(self):
        """Test duration validation with invalid type."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_duration(8.5)

    def test_validate_duration_negative(self):
        """Test duration validation with negative value."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_duration(-1)

    def test_validate_duration_zero(self):
        """Test duration validation with zero."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_duration(0)

    def test_validate_duration_too_large(self):
        """Test duration validation with too large value."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_duration(10000)

    def test_validate_rate_valid(self):
        """Test valid rate validation."""
        assert ConfigValidator.validate_rate(0.05) == 0.05
        assert ConfigValidator.validate_rate(1.0) == 1.0
        assert ConfigValidator.validate_rate(0.001) == 0.001

    def test_validate_rate_invalid_type(self):
        """Test rate validation with invalid type."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_rate("0.05")

    def test_validate_rate_zero(self):
        """Test rate validation with zero."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_rate(0.0)

    def test_validate_rate_negative(self):
        """Test rate validation with negative value."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_rate(-0.1)

    def test_validate_rate_too_large(self):
        """Test rate validation with value too large."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_rate(1.5)

    def test_validate_positive_int_valid(self):
        """Test valid positive int validation."""
        assert ConfigValidator.validate_positive_int(10, "param") == 10
        assert ConfigValidator.validate_positive_int(1, "param") == 1

    def test_validate_positive_int_with_max(self):
        """Test positive int validation with max value."""
        assert ConfigValidator.validate_positive_int(10, "param", max_val=20) == 10

        with pytest.raises(ValidationError):
            ConfigValidator.validate_positive_int(30, "param", max_val=20)

    def test_validate_positive_int_invalid_type(self):
        """Test positive int validation with invalid type."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_positive_int(10.5, "param")

    def test_validate_positive_int_negative(self):
        """Test positive int validation with negative value."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_positive_int(-5, "param")

    def test_validate_float_range_valid(self):
        """Test valid float range validation."""
        result = ConfigValidator.validate_float_range(0.5, "param", 0.0, 1.0)
        assert result == 0.5

    def test_validate_float_range_invalid_type(self):
        """Test float range validation with invalid type."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_float_range("0.5", "param", 0.0, 1.0)

    def test_validate_float_range_out_of_range(self):
        """Test float range validation with value out of range."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_float_range(-0.1, "param", 0.0, 1.0)

        with pytest.raises(ValidationError):
            ConfigValidator.validate_float_range(1.5, "param", 0.0, 1.0)

    def test_validate_llm_wrapper_config_minimal(self):
        """Test LLMWrapper config validation with minimal params."""

        def mock_llm(prompt, max_tokens):
            return "response"

        def mock_embed(text):
            import numpy as np

            return np.random.randn(384).astype(np.float32)

        config = {"llm_generate_fn": mock_llm, "embedding_fn": mock_embed}

        validated = ConfigValidator.validate_llm_wrapper_config(config)

        assert validated["llm_generate_fn"] == mock_llm
        assert validated["embedding_fn"] == mock_embed
        assert validated["dim"] == 384
        assert validated["capacity"] == 20000
        assert validated["wake_duration"] == 8
        assert validated["sleep_duration"] == 3
        assert validated["initial_moral_threshold"] == 0.50

    def test_validate_llm_wrapper_config_custom(self):
        """Test LLMWrapper config validation with custom params."""

        def mock_llm(prompt, max_tokens):
            return "response"

        def mock_embed(text):
            import numpy as np

            return np.random.randn(768).astype(np.float32)

        config = {
            "llm_generate_fn": mock_llm,
            "embedding_fn": mock_embed,
            "dim": 768,
            "capacity": 10000,
            "wake_duration": 10,
            "sleep_duration": 5,
            "initial_moral_threshold": 0.60,
        }

        validated = ConfigValidator.validate_llm_wrapper_config(config)

        assert validated["dim"] == 768
        assert validated["capacity"] == 10000
        assert validated["wake_duration"] == 10
        assert validated["sleep_duration"] == 5
        assert validated["initial_moral_threshold"] == 0.60

    def test_validate_llm_wrapper_config_missing_llm(self):
        """Test LLMWrapper config validation with missing LLM function."""

        def mock_embed(text):
            import numpy as np

            return np.random.randn(384).astype(np.float32)

        config = {"embedding_fn": mock_embed}

        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_llm_wrapper_config(config)

        assert "llm_generate_fn" in str(exc_info.value)

    def test_validate_llm_wrapper_config_missing_embedder(self):
        """Test LLMWrapper config validation with missing embedder."""

        def mock_llm(prompt, max_tokens):
            return "response"

        config = {"llm_generate_fn": mock_llm}

        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_llm_wrapper_config(config)

        assert "embedding_fn" in str(exc_info.value)

    def test_validate_llm_wrapper_config_non_callable_llm(self):
        """Test LLMWrapper config validation with non-callable LLM."""

        def mock_embed(text):
            import numpy as np

            return np.random.randn(384).astype(np.float32)

        config = {"llm_generate_fn": "not a function", "embedding_fn": mock_embed}

        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_llm_wrapper_config(config)

        assert "llm_generate_fn" in str(exc_info.value)
        assert "callable" in str(exc_info.value)

    def test_validate_moral_filter_config_defaults(self):
        """Test MoralFilter config validation with defaults."""
        config = {}
        validated = ConfigValidator.validate_moral_filter_config(config)

        assert validated["initial_threshold"] == 0.50
        assert validated["adapt_rate"] == 0.05
        assert validated["ema_alpha"] == 0.1

    def test_validate_moral_filter_config_custom(self):
        """Test MoralFilter config validation with custom values."""
        config = {"initial_threshold": 0.70, "adapt_rate": 0.10, "ema_alpha": 0.2}

        validated = ConfigValidator.validate_moral_filter_config(config)

        assert validated["initial_threshold"] == 0.70
        assert validated["adapt_rate"] == 0.10
        assert validated["ema_alpha"] == 0.2

    def test_validate_moral_filter_config_invalid_threshold(self):
        """Test MoralFilter config validation with invalid threshold."""
        config = {"initial_threshold": 1.5}

        with pytest.raises(ValidationError):
            ConfigValidator.validate_moral_filter_config(config)

    def test_validate_qilm_config_defaults(self):
        """Test QILM config validation with defaults."""
        config = {}
        validated = ConfigValidator.validate_qilm_config(config)

        assert validated["dim"] == 384
        assert validated["capacity"] == 20000

    def test_validate_qilm_config_custom(self):
        """Test QILM config validation with custom values."""
        config = {"dim": 512, "capacity": 50000}

        validated = ConfigValidator.validate_qilm_config(config)

        assert validated["dim"] == 512
        assert validated["capacity"] == 50000

    def test_validate_config_wrapper(self):
        """Test validate_config wrapper function."""

        def mock_llm(prompt, max_tokens):
            return "response"

        def mock_embed(text):
            import numpy as np

            return np.random.randn(384).astype(np.float32)

        config = {"llm_generate_fn": mock_llm, "embedding_fn": mock_embed}

        validated = validate_config(config, "llm_wrapper")
        assert "llm_generate_fn" in validated
        assert "embedding_fn" in validated

    def test_validate_config_unknown_type(self):
        """Test validate_config with unknown component type."""
        with pytest.raises(ValueError) as exc_info:
            validate_config({}, "unknown_component")

        assert "Unknown component type" in str(exc_info.value)

    def test_validation_error_message(self):
        """Test ValidationError message format."""
        error = ValidationError(
            parameter="test_param", value=42, expected="string", component="TestComponent"
        )

        msg = str(error)
        assert "TestComponent.test_param" in msg
        assert "42" in msg
        assert "string" in msg
