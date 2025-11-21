"""Tests for configuration schema validation.

Tests cover:
- Schema validation for all configuration parameters
- Range constraints and type checking
- Hierarchical constraints (decay rates, thresholds)
- Cross-field validation
- Error message clarity
"""

import pytest
from pydantic import ValidationError

from mlsdm.utils.config_schema import (
    CognitiveRhythmConfig,
    MoralFilterConfig,
    MultiLevelMemoryConfig,
    OntologyMatcherConfig,
    SystemConfig,
    get_default_config,
    validate_config_dict,
)


class TestSystemConfig:
    """Tests for SystemConfig validation."""

    def test_default_config_is_valid(self):
        """Default configuration should be valid."""
        config = get_default_config()
        assert config.dimension == 384
        assert config.strict_mode is False

    def test_valid_dimension_range(self):
        """Test valid dimension values."""
        config = SystemConfig(dimension=384)
        assert config.dimension == 384

        # When changing dimension, must provide matching ontology vectors
        config = SystemConfig(
            dimension=2,
            ontology_matcher=OntologyMatcherConfig(ontology_vectors=[[1.0, 0.0], [0.0, 1.0]])
        )
        assert config.dimension == 2

        # Test maximum dimension with matching ontology vectors
        large_vec = [1.0] + [0.0] * 4095
        config = SystemConfig(
            dimension=4096,
            ontology_matcher=OntologyMatcherConfig(ontology_vectors=[large_vec, large_vec])
        )
        assert config.dimension == 4096

    def test_invalid_dimension_too_small(self):
        """Dimension below minimum should raise error."""
        with pytest.raises(ValidationError) as exc_info:
            SystemConfig(dimension=1)
        assert "greater than or equal to 2" in str(exc_info.value)

    def test_invalid_dimension_too_large(self):
        """Dimension above maximum should raise error."""
        with pytest.raises(ValidationError) as exc_info:
            SystemConfig(dimension=5000)
        assert "less than or equal to 4096" in str(exc_info.value)

    def test_invalid_dimension_type(self):
        """Non-integer dimension should raise error."""
        # Pydantic v2 coerces strings to ints, so test with truly invalid type
        with pytest.raises(ValidationError):
            SystemConfig(dimension="not-a-number")

    def test_strict_mode_boolean(self):
        """Strict mode should accept boolean values."""
        config = SystemConfig(strict_mode=True)
        assert config.strict_mode is True

        config = SystemConfig(strict_mode=False)
        assert config.strict_mode is False

    def test_unknown_fields_rejected(self):
        """Unknown configuration fields should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SystemConfig(unknown_field="value")
        assert "Extra inputs are not permitted" in str(exc_info.value)


class TestMultiLevelMemoryConfig:
    """Tests for MultiLevelMemoryConfig validation."""

    def test_valid_decay_hierarchy(self):
        """Valid decay rate hierarchy should pass."""
        config = MultiLevelMemoryConfig(
            lambda_l1=0.5,
            lambda_l2=0.1,
            lambda_l3=0.01
        )
        assert config.lambda_l1 == 0.5
        assert config.lambda_l2 == 0.1
        assert config.lambda_l3 == 0.01

    def test_invalid_decay_hierarchy_l3_gt_l2(self):
        """lambda_l3 > lambda_l2 should raise error."""
        with pytest.raises(ValidationError) as exc_info:
            MultiLevelMemoryConfig(
                lambda_l1=0.5,
                lambda_l2=0.1,
                lambda_l3=0.2  # Invalid: > lambda_l2
            )
        assert "Decay rates must follow hierarchy" in str(exc_info.value)

    def test_invalid_decay_hierarchy_l2_gt_l1(self):
        """lambda_l2 > lambda_l1 should raise error."""
        with pytest.raises(ValidationError) as exc_info:
            MultiLevelMemoryConfig(
                lambda_l1=0.1,
                lambda_l2=0.5,  # Invalid: > lambda_l1
                lambda_l3=0.01
            )
        assert "Decay rates must follow hierarchy" in str(exc_info.value)

    def test_decay_rates_equal_allowed(self):
        """Equal decay rates should be allowed."""
        config = MultiLevelMemoryConfig(
            lambda_l1=0.5,
            lambda_l2=0.5,
            lambda_l3=0.5
        )
        assert config.lambda_l1 == config.lambda_l2 == config.lambda_l3

    def test_decay_rates_range_valid(self):
        """Decay rates within [0.0, 1.0] should pass."""
        # Must maintain hierarchy: l3 <= l2 <= l1
        config = MultiLevelMemoryConfig(lambda_l1=0.0, lambda_l2=0.0, lambda_l3=0.0)
        assert config.lambda_l1 == 0.0

        config = MultiLevelMemoryConfig(lambda_l1=1.0, lambda_l2=0.5, lambda_l3=0.1)
        assert config.lambda_l1 == 1.0

    def test_decay_rates_range_invalid(self):
        """Decay rates outside [0.0, 1.0] should fail."""
        with pytest.raises(ValidationError):
            MultiLevelMemoryConfig(lambda_l1=-0.1)

        with pytest.raises(ValidationError):
            MultiLevelMemoryConfig(lambda_l1=1.1)

    def test_valid_threshold_hierarchy(self):
        """theta_l2 > theta_l1 should pass."""
        config = MultiLevelMemoryConfig(
            theta_l1=1.0,
            theta_l2=2.0
        )
        assert config.theta_l1 == 1.0
        assert config.theta_l2 == 2.0

    def test_invalid_threshold_hierarchy(self):
        """theta_l2 <= theta_l1 should raise error."""
        with pytest.raises(ValidationError) as exc_info:
            MultiLevelMemoryConfig(
                theta_l1=2.0,
                theta_l2=1.0  # Invalid: <= theta_l1
            )
        assert "threshold hierarchy violated" in str(exc_info.value)

    def test_gating_factors_range(self):
        """Gating factors should be in [0.0, 1.0]."""
        config = MultiLevelMemoryConfig(gating12=0.0, gating23=1.0)
        assert config.gating12 == 0.0
        assert config.gating23 == 1.0

        with pytest.raises(ValidationError):
            MultiLevelMemoryConfig(gating12=-0.1)

        with pytest.raises(ValidationError):
            MultiLevelMemoryConfig(gating23=1.1)


class TestMoralFilterConfig:
    """Tests for MoralFilterConfig validation."""

    def test_valid_threshold_bounds(self):
        """Valid threshold configuration should pass."""
        config = MoralFilterConfig(
            threshold=0.5,
            min_threshold=0.3,
            max_threshold=0.9
        )
        assert config.threshold == 0.5
        assert config.min_threshold == 0.3
        assert config.max_threshold == 0.9

    def test_invalid_min_greater_than_max(self):
        """min_threshold >= max_threshold should fail."""
        with pytest.raises(ValidationError) as exc_info:
            MoralFilterConfig(
                min_threshold=0.9,
                max_threshold=0.3
            )
        assert "must be <" in str(exc_info.value)

    def test_invalid_threshold_below_min(self):
        """threshold < min_threshold should fail."""
        with pytest.raises(ValidationError) as exc_info:
            MoralFilterConfig(
                threshold=0.2,
                min_threshold=0.3
            )
        assert "must be >=" in str(exc_info.value)

    def test_invalid_threshold_above_max(self):
        """threshold > max_threshold should fail."""
        with pytest.raises(ValidationError) as exc_info:
            MoralFilterConfig(
                threshold=0.95,
                max_threshold=0.9
            )
        # Pydantic v2 uses "less than or equal" in error messages
        assert "less than or equal" in str(exc_info.value).lower()

    def test_threshold_range_constraints(self):
        """Thresholds should be in valid ranges."""
        with pytest.raises(ValidationError):
            MoralFilterConfig(threshold=0.05)  # Below 0.1

        with pytest.raises(ValidationError):
            MoralFilterConfig(threshold=0.95)  # Above 0.9

    def test_adapt_rate_range(self):
        """Adapt rate should be in [0.0, 0.5]."""
        config = MoralFilterConfig(adapt_rate=0.0)
        assert config.adapt_rate == 0.0

        config = MoralFilterConfig(adapt_rate=0.5)
        assert config.adapt_rate == 0.5

        with pytest.raises(ValidationError):
            MoralFilterConfig(adapt_rate=-0.1)

        with pytest.raises(ValidationError):
            MoralFilterConfig(adapt_rate=0.6)


class TestOntologyMatcherConfig:
    """Tests for OntologyMatcherConfig validation."""

    def test_valid_vectors_and_labels(self):
        """Valid ontology configuration should pass."""
        config = OntologyMatcherConfig(
            ontology_vectors=[[1.0, 0.0], [0.0, 1.0]],
            ontology_labels=["cat1", "cat2"]
        )
        assert len(config.ontology_vectors) == 2
        assert len(config.ontology_labels) == 2

    def test_vectors_same_dimension(self):
        """All vectors should have same dimension."""
        with pytest.raises(ValidationError) as exc_info:
            OntologyMatcherConfig(
                ontology_vectors=[[1.0, 0.0], [1.0, 0.0, 0.0]]
            )
        assert "same dimension" in str(exc_info.value)

    def test_empty_vectors_rejected(self):
        """Empty ontology_vectors should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            OntologyMatcherConfig(ontology_vectors=[])
        assert "cannot be empty" in str(exc_info.value)

    def test_labels_mismatch_count(self):
        """Number of labels must match vectors."""
        with pytest.raises(ValidationError) as exc_info:
            OntologyMatcherConfig(
                ontology_vectors=[[1.0, 0.0], [0.0, 1.0]],
                ontology_labels=["cat1"]  # Only 1 label for 2 vectors
            )
        assert "must match" in str(exc_info.value)

    def test_labels_optional(self):
        """Labels should be optional."""
        config = OntologyMatcherConfig(
            ontology_vectors=[[1.0, 0.0], [0.0, 1.0]],
            ontology_labels=None
        )
        assert config.ontology_labels is None


class TestCognitiveRhythmConfig:
    """Tests for CognitiveRhythmConfig validation."""

    def test_valid_durations(self):
        """Valid wake/sleep durations should pass."""
        config = CognitiveRhythmConfig(
            wake_duration=8,
            sleep_duration=3
        )
        assert config.wake_duration == 8
        assert config.sleep_duration == 3

    def test_duration_range_constraints(self):
        """Durations should be in [1, 100]."""
        config = CognitiveRhythmConfig(wake_duration=1)
        assert config.wake_duration == 1

        config = CognitiveRhythmConfig(sleep_duration=100)
        assert config.sleep_duration == 100

        with pytest.raises(ValidationError):
            CognitiveRhythmConfig(wake_duration=0)

        with pytest.raises(ValidationError):
            CognitiveRhythmConfig(sleep_duration=101)


class TestCrossFieldValidation:
    """Tests for cross-field validation in SystemConfig."""

    def test_ontology_dimension_mismatch(self):
        """Ontology vector dimension must match system dimension."""
        with pytest.raises(ValidationError) as exc_info:
            SystemConfig(
                dimension=10,
                ontology_matcher=OntologyMatcherConfig(
                    ontology_vectors=[[1.0, 0.0], [0.0, 1.0]]  # dim=2, not 10
                )
            )
        assert "must match system dimension" in str(exc_info.value)

    def test_ontology_dimension_match(self):
        """Matching ontology and system dimensions should pass."""
        config = SystemConfig(
            dimension=2,
            ontology_matcher=OntologyMatcherConfig(
                ontology_vectors=[[1.0, 0.0], [0.0, 1.0]]
            )
        )
        assert config.dimension == 2


class TestConfigLoaderIntegration:
    """Tests for validate_config_dict function."""

    def test_valid_dict(self):
        """Valid configuration dictionary should pass."""
        config_dict = {
            "dimension": 384,
            "moral_filter": {
                "threshold": 0.5
            },
            "strict_mode": False
        }
        config = validate_config_dict(config_dict)
        assert config.dimension == 384

    def test_invalid_dict(self):
        """Invalid configuration dictionary should raise ValueError."""
        config_dict = {
            "dimension": -1  # Invalid
        }
        with pytest.raises(ValueError) as exc_info:
            validate_config_dict(config_dict)
        assert "Configuration validation failed" in str(exc_info.value)

    def test_empty_dict_uses_defaults(self):
        """Empty dictionary should use all defaults."""
        config = validate_config_dict({})
        assert config.dimension == 384  # Default
        assert config.strict_mode is False  # Default


class TestConfigSerialization:
    """Tests for configuration serialization/deserialization."""

    def test_model_dump(self):
        """Config should serialize to dictionary."""
        # Use default dimension to avoid ontology mismatch
        config = SystemConfig(dimension=384)
        data = config.model_dump()
        assert isinstance(data, dict)
        assert data["dimension"] == 384

    def test_model_dump_json(self):
        """Config should serialize to JSON."""
        config = SystemConfig(dimension=384)
        json_str = config.model_dump_json()
        assert isinstance(json_str, str)
        assert "384" in json_str

    def test_round_trip(self):
        """Config should round-trip through dict."""
        config1 = SystemConfig(
            dimension=384,
            moral_filter=MoralFilterConfig(threshold=0.7)
        )
        data = config1.model_dump()
        config2 = SystemConfig(**data)
        assert config2.dimension == config1.dimension
        assert config2.moral_filter.threshold == config1.moral_filter.threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
