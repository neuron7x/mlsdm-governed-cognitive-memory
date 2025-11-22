"""Configuration schema and validation for MLSDM Governed Cognitive Memory.

This module defines the configuration schema using Pydantic models for
type safety and validation. It ensures all configuration parameters are
properly validated before use.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class MultiLevelMemoryConfig(BaseModel):
    """Multi-level synaptic memory configuration.

    Defines decay rates and gating parameters for the three-level
    memory hierarchy (L1, L2, L3).
    """
    lambda_l1: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="L1 decay rate (short-term memory). Higher = faster decay."
    )
    lambda_l2: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="L2 decay rate (medium-term memory). Should be < lambda_l1."
    )
    lambda_l3: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="L3 decay rate (long-term memory). Should be < lambda_l2."
    )
    theta_l1: float = Field(
        default=1.0,
        ge=0.0,
        description="L1 threshold for memory consolidation to L2."
    )
    theta_l2: float = Field(
        default=2.0,
        ge=0.0,
        description="L2 threshold for memory consolidation to L3."
    )
    gating12: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Gating factor for L1 to L2 consolidation."
    )
    gating23: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Gating factor for L2 to L3 consolidation."
    )

    @model_validator(mode='after')
    def validate_decay_hierarchy(self):
        """Ensure decay rates follow hierarchy: lambda_l3 < lambda_l2 < lambda_l1."""
        l1, l2, l3 = self.lambda_l1, self.lambda_l2, self.lambda_l3
        if not (l3 <= l2 <= l1):
            raise ValueError(
                f"Decay rates must follow hierarchy: lambda_l3 ({l3}) <= "
                f"lambda_l2 ({l2}) <= lambda_l1 ({l1})"
            )
        return self

    @model_validator(mode='after')
    def validate_threshold_hierarchy(self):
        """Ensure theta_l2 > theta_l1 for proper consolidation."""
        t1, t2 = self.theta_l1, self.theta_l2
        if t2 <= t1:
            raise ValueError(
                f"Consolidation threshold hierarchy violated: "
                f"theta_l2 ({t2}) must be > theta_l1 ({t1})"
            )
        return self


class MoralFilterConfig(BaseModel):
    """Moral filter configuration for content governance.

    Adaptive moral threshold system that adjusts based on content
    quality to maintain homeostatic balance.
    """
    threshold: float = Field(
        default=0.5,
        ge=0.1,
        le=0.9,
        description="Initial moral threshold. Values [0.0-1.0], higher = stricter."
    )
    adapt_rate: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Adaptation rate for threshold adjustment. Higher = faster adaptation."
    )
    min_threshold: float = Field(
        default=0.3,
        ge=0.1,
        le=0.9,
        description="Minimum allowed moral threshold."
    )
    max_threshold: float = Field(
        default=0.9,
        ge=0.1,
        le=0.99,
        description="Maximum allowed moral threshold."
    )

    @model_validator(mode='after')
    def validate_threshold_bounds(self):
        """Ensure min <= threshold <= max."""
        min_t = self.min_threshold
        max_t = self.max_threshold
        threshold = self.threshold

        if min_t is not None and max_t is not None and min_t >= max_t:
            raise ValueError(
                f"min_threshold ({min_t}) must be < max_threshold ({max_t})"
            )

        if threshold is not None:
            if min_t is not None and threshold < min_t:
                raise ValueError(
                    f"threshold ({threshold}) must be >= min_threshold ({min_t})"
                )
            if max_t is not None and threshold > max_t:
                raise ValueError(
                    f"threshold ({threshold}) must be <= max_threshold ({max_t})"
                )

        return self


class OntologyMatcherConfig(BaseModel):
    """Ontology matcher configuration for semantic categorization."""
    ontology_vectors: list[list[float]] = Field(
        default_factory=lambda: [[1.0] + [0.0] * 383, [0.0, 1.0] + [0.0] * 382],
        description="List of ontology category vectors. Must match dimension."
    )
    ontology_labels: list[str] | None = Field(
        default=None,
        description="Human-readable labels for ontology categories."
    )

    @field_validator('ontology_vectors')
    @classmethod
    def validate_vectors(cls, v):
        """Ensure all vectors have same dimension and are non-empty."""
        if not v:
            raise ValueError("ontology_vectors cannot be empty")

        dims = [len(vec) for vec in v]
        if len(set(dims)) > 1:
            raise ValueError(
                f"All ontology vectors must have same dimension. Found: {set(dims)}"
            )

        return v

    @model_validator(mode='after')
    def validate_labels_match(self):
        """Ensure labels match number of vectors if provided."""
        vectors = self.ontology_vectors
        labels = self.ontology_labels

        if labels is not None and len(labels) != len(vectors):
            raise ValueError(
                f"Number of labels ({len(labels)}) must match "
                f"number of vectors ({len(vectors)})"
            )

        return self


class CognitiveRhythmConfig(BaseModel):
    """Cognitive rhythm configuration for wake/sleep cycles.

    Controls the circadian-like rhythm that governs processing modes.
    """
    wake_duration: int = Field(
        default=8,
        ge=1,
        le=100,
        description="Duration of wake phase (in cycles). Typical: 5-10."
    )
    sleep_duration: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Duration of sleep phase (in cycles). Typical: 2-5."
    )

    @model_validator(mode='after')
    def validate_durations(self):
        """Warn if unusual wake/sleep ratio."""
        wake = self.wake_duration
        sleep = self.sleep_duration

        if wake is not None and sleep is not None:
            ratio = wake / sleep
            if ratio < 1.0 or ratio > 10.0:
                # Note: This is a warning, not an error
                # In production, consider logging this
                pass

        return self


class SystemConfig(BaseModel):
    """Complete system configuration.

    Root configuration object that encompasses all subsystem configurations.
    """
    dimension: int = Field(
        default=384,
        ge=2,
        le=4096,
        description="Vector dimension for embeddings. Common values: 384, 768, 1536."
    )
    multi_level_memory: MultiLevelMemoryConfig = Field(
        default_factory=MultiLevelMemoryConfig,
        description="Multi-level synaptic memory configuration."
    )
    moral_filter: MoralFilterConfig = Field(
        default_factory=MoralFilterConfig,
        description="Moral filter configuration for content governance."
    )
    ontology_matcher: OntologyMatcherConfig = Field(
        default_factory=OntologyMatcherConfig,
        description="Ontology matcher configuration."
    )
    cognitive_rhythm: CognitiveRhythmConfig = Field(
        default_factory=CognitiveRhythmConfig,
        description="Cognitive rhythm (wake/sleep cycle) configuration."
    )
    strict_mode: bool = Field(
        default=False,
        description="Enable strict mode for enhanced validation. Not recommended for production."
    )

    @model_validator(mode='after')
    def validate_ontology_dimension(self):
        """Ensure ontology vectors match system dimension."""
        dim = self.dimension
        onto_cfg = self.ontology_matcher

        if dim is not None and onto_cfg is not None:
            vectors = onto_cfg.ontology_vectors
            if vectors:
                vec_dim = len(vectors[0])
                if vec_dim != dim:
                    raise ValueError(
                        f"Ontology vector dimension ({vec_dim}) must match "
                        f"system dimension ({dim})"
                    )

        return self

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",  # Reject unknown fields
        json_schema_extra={
            "examples": [
                {
                    "dimension": 384,
                    "multi_level_memory": {
                        "lambda_l1": 0.5,
                        "lambda_l2": 0.1,
                        "lambda_l3": 0.01,
                        "theta_l1": 1.0,
                        "theta_l2": 2.0,
                        "gating12": 0.5,
                        "gating23": 0.3
                    },
                    "moral_filter": {
                        "threshold": 0.5,
                        "adapt_rate": 0.05,
                        "min_threshold": 0.3,
                        "max_threshold": 0.9
                    },
                    "cognitive_rhythm": {
                        "wake_duration": 8,
                        "sleep_duration": 3
                    },
                    "strict_mode": False
                }
            ]
        }
    )


def validate_config_dict(config_dict: dict[str, Any]) -> SystemConfig:
    """Validate a configuration dictionary against the schema.

    Args:
        config_dict: Dictionary containing configuration parameters

    Returns:
        Validated SystemConfig instance

    Raises:
        ValueError: If configuration is invalid
    """
    try:
        return SystemConfig(**config_dict)
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {str(e)}") from e


def get_default_config() -> SystemConfig:
    """Get default system configuration.

    Returns:
        SystemConfig with all default values
    """
    return SystemConfig()
