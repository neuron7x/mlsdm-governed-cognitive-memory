"""
Unit Tests for ConfigValidator

Tests configuration validation for MLSDM components.
"""

import pytest

from mlsdm.utils.config_validator import (
    ConfigValidator,
    ValidationError,
    validate_config,
)


class TestValidationError:
    """Test ValidationError exception."""

    def test_error_message_format(self):
        """Test error message formatting."""
        error = ValidationError(
            parameter="dim",
            value=-1,
            expected="positive integer",
            component="TestComponent"
        )

        message = str(error)
        assert "TestComponent" in message
        assert "dim" in message
        assert "-1" in message
        assert "positive integer" in message

    def test_error_with_string_value(self):
        """Test error with string value."""
        error = ValidationError(
            parameter="threshold",
            value="invalid",
            expected="float in range [0, 1]",
            component="MoralFilter"
        )

        message = str(error)
        assert "'invalid'" in message


class TestValidateDimension:
    """Test dimension validation."""

    def test_valid_dimension(self):
        """Test valid dimension passes."""
        result = ConfigValidator.validate_dimension(384)
        assert result == 384

    def test_dimension_at_minimum(self):
        """Test dimension at minimum value."""
        result = ConfigValidator.validate_dimension(1)
        assert result == 1

    def test_dimension_at_maximum(self):
        """Test dimension at maximum value."""
        result = ConfigValidator.validate_dimension(10000)
        assert result == 10000

    def test_dimension_zero_raises(self):
        """Test zero dimension raises error."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_dimension(0)

        assert exc_info.value.parameter == "dim"
        assert exc_info.value.value == 0

    def test_dimension_negative_raises(self):
        """Test negative dimension raises error."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_dimension(-5)

    def test_dimension_too_large_raises(self):
        """Test too large dimension raises error."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_dimension(10001)

    def test_dimension_non_integer_raises(self):
        """Test non-integer dimension raises error."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_dimension(384.5)

    def test_dimension_string_raises(self):
        """Test string dimension raises error."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_dimension("384")

    def test_dimension_custom_component(self):
        """Test custom component name in error."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_dimension(0, component="PELM")

        assert exc_info.value.component == "PELM"


class TestValidateCapacity:
    """Test capacity validation."""

    def test_valid_capacity(self):
        """Test valid capacity passes."""
        result = ConfigValidator.validate_capacity(20000)
        assert result == 20000

    def test_capacity_at_minimum(self):
        """Test capacity at minimum value."""
        result = ConfigValidator.validate_capacity(1)
        assert result == 1

    def test_capacity_at_maximum(self):
        """Test capacity at maximum value."""
        result = ConfigValidator.validate_capacity(1000000)
        assert result == 1000000

    def test_capacity_zero_raises(self):
        """Test zero capacity raises error."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_capacity(0)

    def test_capacity_negative_raises(self):
        """Test negative capacity raises error."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_capacity(-100)

    def test_capacity_too_large_raises(self):
        """Test too large capacity raises error."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_capacity(1000001)

    def test_capacity_non_integer_raises(self):
        """Test non-integer capacity raises error."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_capacity(20000.5)


class TestValidateThreshold:
    """Test threshold validation."""

    def test_valid_threshold(self):
        """Test valid threshold passes."""
        result = ConfigValidator.validate_threshold(0.5)
        assert result == 0.5

    def test_threshold_at_minimum(self):
        """Test threshold at minimum value."""
        result = ConfigValidator.validate_threshold(0.0)
        assert result == 0.0

    def test_threshold_at_maximum(self):
        """Test threshold at maximum value."""
        result = ConfigValidator.validate_threshold(1.0)
        assert result == 1.0

    def test_threshold_below_minimum_raises(self):
        """Test threshold below minimum raises error."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_threshold(-0.1)

    def test_threshold_above_maximum_raises(self):
        """Test threshold above maximum raises error."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_threshold(1.1)

    def test_threshold_custom_range(self):
        """Test threshold with custom range."""
        result = ConfigValidator.validate_threshold(0.6, min_val=0.3, max_val=0.9)
        assert result == 0.6

        with pytest.raises(ValidationError):
            ConfigValidator.validate_threshold(0.2, min_val=0.3, max_val=0.9)

    def test_threshold_from_integer(self):
        """Test threshold accepts integer and converts to float."""
        result = ConfigValidator.validate_threshold(1)
        assert result == 1.0
        assert isinstance(result, float)

    def test_threshold_string_raises(self):
        """Test string threshold raises error."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_threshold("0.5")


class TestValidateDuration:
    """Test duration validation."""

    def test_valid_duration(self):
        """Test valid duration passes."""
        result = ConfigValidator.validate_duration(8)
        assert result == 8

    def test_duration_at_minimum(self):
        """Test duration at minimum value."""
        result = ConfigValidator.validate_duration(1)
        assert result == 1

    def test_duration_at_maximum(self):
        """Test duration at maximum value."""
        result = ConfigValidator.validate_duration(1000)
        assert result == 1000

    def test_duration_zero_raises(self):
        """Test zero duration raises error."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_duration(0)

    def test_duration_negative_raises(self):
        """Test negative duration raises error."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_duration(-5)

    def test_duration_too_large_raises(self):
        """Test too large duration raises error."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_duration(1001)

    def test_duration_non_integer_raises(self):
        """Test non-integer duration raises error."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_duration(8.5)

    def test_duration_custom_parameter_name(self):
        """Test custom parameter name in error."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_duration(0, parameter_name="wake_duration")

        assert exc_info.value.parameter == "wake_duration"


class TestValidateRate:
    """Test rate validation."""

    def test_valid_rate(self):
        """Test valid rate passes."""
        result = ConfigValidator.validate_rate(0.05)
        assert result == 0.05

    def test_rate_at_maximum(self):
        """Test rate at maximum value."""
        result = ConfigValidator.validate_rate(1.0)
        assert result == 1.0

    def test_rate_near_zero(self):
        """Test rate near zero passes."""
        result = ConfigValidator.validate_rate(0.001)
        assert result == 0.001

    def test_rate_zero_raises(self):
        """Test zero rate raises error."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_rate(0)

    def test_rate_negative_raises(self):
        """Test negative rate raises error."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_rate(-0.05)

    def test_rate_above_maximum_raises(self):
        """Test rate above maximum raises error."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_rate(1.1)

    def test_rate_from_integer(self):
        """Test rate accepts integer."""
        result = ConfigValidator.validate_rate(1)
        assert result == 1.0

    def test_rate_custom_parameter_name(self):
        """Test custom parameter name in error."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_rate(0, parameter_name="adapt_rate")

        assert exc_info.value.parameter == "adapt_rate"


class TestValidatePositiveInt:
    """Test positive integer validation."""

    def test_valid_positive_int(self):
        """Test valid positive integer passes."""
        result = ConfigValidator.validate_positive_int(10, "count")
        assert result == 10

    def test_positive_int_at_minimum(self):
        """Test positive integer at minimum value."""
        result = ConfigValidator.validate_positive_int(1, "count")
        assert result == 1

    def test_positive_int_zero_raises(self):
        """Test zero raises error."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_positive_int(0, "count")

    def test_positive_int_negative_raises(self):
        """Test negative value raises error."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_positive_int(-5, "count")

    def test_positive_int_with_max(self):
        """Test positive integer with max value."""
        result = ConfigValidator.validate_positive_int(50, "count", max_val=100)
        assert result == 50

        with pytest.raises(ValidationError):
            ConfigValidator.validate_positive_int(150, "count", max_val=100)

    def test_positive_int_non_integer_raises(self):
        """Test non-integer raises error."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_positive_int(10.5, "count")


class TestValidateFloatRange:
    """Test float range validation."""

    def test_valid_float_range(self):
        """Test valid float in range passes."""
        result = ConfigValidator.validate_float_range(0.5, "param", 0.0, 1.0)
        assert result == 0.5

    def test_float_at_minimum(self):
        """Test float at minimum value."""
        result = ConfigValidator.validate_float_range(0.0, "param", 0.0, 1.0)
        assert result == 0.0

    def test_float_at_maximum(self):
        """Test float at maximum value."""
        result = ConfigValidator.validate_float_range(1.0, "param", 0.0, 1.0)
        assert result == 1.0

    def test_float_below_minimum_raises(self):
        """Test float below minimum raises error."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_float_range(-0.1, "param", 0.0, 1.0)

    def test_float_above_maximum_raises(self):
        """Test float above maximum raises error."""
        with pytest.raises(ValidationError):
            ConfigValidator.validate_float_range(1.1, "param", 0.0, 1.0)

    def test_float_from_integer(self):
        """Test float range accepts integer."""
        result = ConfigValidator.validate_float_range(1, "param", 0.0, 1.0)
        assert result == 1.0
        assert isinstance(result, float)


class TestValidateLLMWrapperConfig:
    """Test LLMWrapper configuration validation."""

    def test_valid_config(self):
        """Test valid configuration passes."""
        config = {
            "llm_generate_fn": lambda x, y: "response",
            "embedding_fn": lambda x: [0.1, 0.2, 0.3],
            "dim": 384,
            "capacity": 20000,
            "wake_duration": 8,
            "sleep_duration": 3,
            "initial_moral_threshold": 0.5,
        }

        validated = ConfigValidator.validate_llm_wrapper_config(config)

        assert validated["dim"] == 384
        assert validated["capacity"] == 20000
        assert validated["wake_duration"] == 8
        assert validated["sleep_duration"] == 3
        assert validated["initial_moral_threshold"] == 0.5

    def test_missing_llm_generate_fn_raises(self):
        """Test missing llm_generate_fn raises error."""
        config = {
            "embedding_fn": lambda x: [0.1, 0.2, 0.3],
        }

        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_llm_wrapper_config(config)

        assert exc_info.value.parameter == "llm_generate_fn"

    def test_non_callable_llm_generate_fn_raises(self):
        """Test non-callable llm_generate_fn raises error."""
        config = {
            "llm_generate_fn": "not a function",
            "embedding_fn": lambda x: [0.1, 0.2, 0.3],
        }

        with pytest.raises(ValidationError):
            ConfigValidator.validate_llm_wrapper_config(config)

    def test_missing_embedding_fn_raises(self):
        """Test missing embedding_fn raises error."""
        config = {
            "llm_generate_fn": lambda x, y: "response",
        }

        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_llm_wrapper_config(config)

        assert exc_info.value.parameter == "embedding_fn"

    def test_non_callable_embedding_fn_raises(self):
        """Test non-callable embedding_fn raises error."""
        config = {
            "llm_generate_fn": lambda x, y: "response",
            "embedding_fn": "not a function",
        }

        with pytest.raises(ValidationError):
            ConfigValidator.validate_llm_wrapper_config(config)

    def test_default_values_used(self):
        """Test default values are used when not provided."""
        config = {
            "llm_generate_fn": lambda x, y: "response",
            "embedding_fn": lambda x: [0.1, 0.2, 0.3],
        }

        validated = ConfigValidator.validate_llm_wrapper_config(config)

        assert validated["dim"] == 384  # Default
        assert validated["capacity"] == 20000  # Default
        assert validated["wake_duration"] == 8  # Default
        assert validated["sleep_duration"] == 3  # Default

    def test_invalid_initial_moral_threshold_raises(self):
        """Test invalid moral threshold raises error."""
        config = {
            "llm_generate_fn": lambda x, y: "response",
            "embedding_fn": lambda x: [0.1, 0.2, 0.3],
            "initial_moral_threshold": 0.1,  # Below 0.3 min
        }

        with pytest.raises(ValidationError):
            ConfigValidator.validate_llm_wrapper_config(config)


class TestValidateMoralFilterConfig:
    """Test MoralFilter configuration validation."""

    def test_valid_config(self):
        """Test valid configuration passes."""
        config = {
            "initial_threshold": 0.6,
            "adapt_rate": 0.1,
            "ema_alpha": 0.2,
        }

        validated = ConfigValidator.validate_moral_filter_config(config)

        assert validated["initial_threshold"] == 0.6
        assert validated["adapt_rate"] == 0.1
        assert validated["ema_alpha"] == 0.2

    def test_default_values_used(self):
        """Test default values are used when not provided."""
        config = {}

        validated = ConfigValidator.validate_moral_filter_config(config)

        assert validated["initial_threshold"] == 0.5  # Default
        assert validated["adapt_rate"] == 0.05  # Default
        assert validated["ema_alpha"] == 0.1  # Default

    def test_invalid_threshold_raises(self):
        """Test invalid threshold raises error."""
        config = {"initial_threshold": 0.1}  # Below 0.3 min

        with pytest.raises(ValidationError):
            ConfigValidator.validate_moral_filter_config(config)


class TestValidateQILMConfig:
    """Test QILM configuration validation."""

    def test_valid_config(self):
        """Test valid configuration passes."""
        config = {
            "dim": 256,
            "capacity": 10000,
        }

        validated = ConfigValidator.validate_qilm_config(config)

        assert validated["dim"] == 256
        assert validated["capacity"] == 10000

    def test_default_values_used(self):
        """Test default values are used when not provided."""
        config = {}

        validated = ConfigValidator.validate_qilm_config(config)

        assert validated["dim"] == 384  # Default
        assert validated["capacity"] == 20000  # Default


class TestValidateConfigFunction:
    """Test the validate_config helper function."""

    def test_llm_wrapper_type(self):
        """Test validation for llm_wrapper type."""
        config = {
            "llm_generate_fn": lambda x, y: "response",
            "embedding_fn": lambda x: [0.1],
        }

        validated = validate_config(config, "llm_wrapper")
        assert "dim" in validated

    def test_moral_filter_type(self):
        """Test validation for moral_filter type."""
        config = {}

        validated = validate_config(config, "moral_filter")
        assert "initial_threshold" in validated

    def test_qilm_type(self):
        """Test validation for qilm type."""
        config = {}

        validated = validate_config(config, "qilm")
        assert "dim" in validated

    def test_unknown_type_raises(self):
        """Test unknown component type raises error."""
        with pytest.raises(ValueError, match="Unknown component type"):
            validate_config({}, "unknown_component")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
