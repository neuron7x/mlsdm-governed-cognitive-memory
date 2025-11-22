# Configuration Guide

**Document Version:** 1.0.0  
**Project Version:** 1.0.0  
**Last Updated:** November 2025  
**Status:** Production

Complete guide to configuring MLSDM Governed Cognitive Memory for different deployment scenarios.

## Table of Contents

- [Overview](#overview)
- [Configuration Files](#configuration-files)
- [Configuration Schema](#configuration-schema)
- [Environment Variables](#environment-variables)
- [Validation](#validation)
- [Examples](#examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

MLSDM supports multiple configuration methods with the following precedence (highest to lowest):

1. **Environment variables** (prefixed with `MLSDM_`)
2. **Configuration file** (YAML or INI format)
3. **Default values** (defined in schema)

All configurations are validated against a strict schema to prevent runtime errors.

## Configuration Files

### Available Templates

- `config/default_config.yaml` - Development/testing configuration
- `config/production.yaml` - Production-ready configuration template
- `env.example` - Environment variable template

### File Format

Configuration files support YAML format (recommended) or INI format:

```yaml
# config/custom.yaml
dimension: 384

multi_level_memory:
  lambda_l1: 0.5
  lambda_l2: 0.1
  lambda_l3: 0.01
```

### Loading Configuration

```python
from mlsdm.utils.config_loader import ConfigLoader

# Load with validation (recommended)
config = ConfigLoader.load_config("config/production.yaml", validate=True)

# Load without validation (not recommended)
config = ConfigLoader.load_config("config/custom.yaml", validate=False)

# Get validated config object
config_obj = ConfigLoader.load_validated_config("config/production.yaml")

# Extract aphasia config for NeuroLangWrapper (convenience helper)
aphasia_params = ConfigLoader.get_aphasia_config_from_dict(config)
# Returns: {
#   "aphasia_detect_enabled": bool,
#   "aphasia_repair_enabled": bool,
#   "aphasia_severity_threshold": float
# }

# Extract NeuroLang config for NeuroLangWrapper (convenience helper)
neurolang_params = ConfigLoader.get_neurolang_config_from_dict(config)
# Returns: {
#   "neurolang_mode": str,
#   "neurolang_checkpoint_path": str | None
# }
```

## Configuration Schema

### System Configuration

#### `dimension` (integer)

Vector dimension for embeddings.

- **Type**: Integer
- **Range**: 2 to 4096
- **Default**: 384
- **Common values**: 
  - 384 (sentence-transformers/all-MiniLM-L6-v2)
  - 768 (BERT-base)
  - 1536 (OpenAI text-embedding-ada-002)

**Must match your embedding model's output dimension.**

```yaml
dimension: 384
```

### Multi-Level Memory Configuration

Three-level memory hierarchy with decay rates and consolidation thresholds.

#### `lambda_l1` (float)

L1 (short-term) memory decay rate.

- **Type**: Float
- **Range**: 0.0 to 1.0
- **Default**: 0.5
- **Description**: Higher values = faster decay. Controls how quickly short-term memories fade.

#### `lambda_l2` (float)

L2 (medium-term) memory decay rate.

- **Type**: Float
- **Range**: 0.0 to 1.0
- **Default**: 0.1
- **Constraint**: Must be ≤ lambda_l1

#### `lambda_l3` (float)

L3 (long-term) memory decay rate.

- **Type**: Float
- **Range**: 0.0 to 1.0
- **Default**: 0.01
- **Constraint**: Must be ≤ lambda_l2

**Hierarchy constraint**: `lambda_l3 ≤ lambda_l2 ≤ lambda_l1`

#### `theta_l1` (float)

Threshold for L1→L2 memory consolidation.

- **Type**: Float
- **Range**: ≥ 0.0
- **Default**: 1.0

#### `theta_l2` (float)

Threshold for L2→L3 memory consolidation.

- **Type**: Float
- **Range**: ≥ 0.0
- **Default**: 2.0
- **Constraint**: Must be > theta_l1

#### `gating12` (float)

Gating factor for L1→L2 consolidation.

- **Type**: Float
- **Range**: 0.0 to 1.0
- **Default**: 0.5

#### `gating23` (float)

Gating factor for L2→L3 consolidation.

- **Type**: Float
- **Range**: 0.0 to 1.0
- **Default**: 0.3

**Example:**

```yaml
multi_level_memory:
  lambda_l1: 0.5
  lambda_l2: 0.1
  lambda_l3: 0.01
  theta_l1: 1.0
  theta_l2: 2.0
  gating12: 0.5
  gating23: 0.3
```

### Moral Filter Configuration

Adaptive moral threshold system for content governance.

#### `threshold` (float)

Initial moral threshold.

- **Type**: Float
- **Range**: 0.1 to 0.9
- **Default**: 0.5
- **Description**: Higher values = stricter filtering. Scale: 0.0 (accept all) to 1.0 (reject all).

#### `adapt_rate` (float)

Threshold adaptation rate.

- **Type**: Float
- **Range**: 0.0 to 0.5
- **Default**: 0.05
- **Description**: Controls how quickly the threshold adjusts. Higher = faster adaptation.

#### `min_threshold` (float)

Minimum allowed threshold.

- **Type**: Float
- **Range**: 0.1 to 0.9
- **Default**: 0.3
- **Description**: Lower bound for adaptive threshold.

#### `max_threshold` (float)

Maximum allowed threshold.

- **Type**: Float
- **Range**: 0.1 to 0.99
- **Default**: 0.9
- **Description**: Upper bound for adaptive threshold.

**Constraints**: `min_threshold < initial threshold < max_threshold`

**Example:**

```yaml
moral_filter:
  threshold: 0.5
  adapt_rate: 0.05
  min_threshold: 0.3
  max_threshold: 0.9
```

### Ontology Matcher Configuration

Semantic categorization using ontology vectors.

#### `ontology_vectors` (list of lists)

Category definition vectors.

- **Type**: List of float lists
- **Constraint**: All vectors must have same dimension as system `dimension`
- **Default**: Two identity vectors

#### `ontology_labels` (list of strings, optional)

Human-readable category labels.

- **Type**: List of strings
- **Constraint**: Must match number of vectors if provided
- **Default**: None

**Example:**

```yaml
ontology_matcher:
  ontology_vectors:
    - [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    - [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  ontology_labels:
    - "technical"
    - "social"
    - "creative"
```

### Cognitive Rhythm Configuration

Wake/sleep cycle controlling processing modes.

#### `wake_duration` (integer)

Duration of wake phase in cycles.

- **Type**: Integer
- **Range**: 1 to 100
- **Default**: 8
- **Typical**: 5-10

#### `sleep_duration` (integer)

Duration of sleep phase in cycles.

- **Type**: Integer
- **Range**: 1 to 100
- **Default**: 3
- **Typical**: 2-5

**Recommended ratio**: wake_duration = 2-3× sleep_duration

**Example:**

```yaml
cognitive_rhythm:
  wake_duration: 8
  sleep_duration: 3
```

### System Behavior

#### `strict_mode` (boolean)

Enable enhanced validation.

- **Type**: Boolean
- **Default**: false
- **Production**: Set to false (performance impact)
- **Development**: Can be enabled for debugging

**Example:**

```yaml
strict_mode: false
```

### Aphasia-Broca Configuration

Configuration for Aphasia-Broca detection and repair in NeuroLangWrapper.

**Note:** These settings only apply when using `NeuroLangWrapper` from `mlsdm.extensions`. The base `LLMWrapper` does not use these settings.

#### `detect_enabled` (boolean)

Enable/disable aphasia detection.

- **Type**: Boolean
- **Default**: true
- **Description**: When enabled, analyzes LLM responses for Broca-like aphasia symptoms (telegraphic speech, broken syntax)

**When disabled:**
- No aphasia analysis is performed
- `aphasia_flags` in response will be `None`
- Response text is returned as-is from LLM

#### `repair_enabled` (boolean)

Enable/disable automatic repair of detected aphasia.

- **Type**: Boolean
- **Default**: true
- **Requires**: `detect_enabled` must be true
- **Description**: When enabled, automatically regenerates responses that show aphasia symptoms

**When disabled (monitoring mode):**
- Aphasia is detected and reported in `aphasia_flags`
- Response text is NOT modified
- Useful for observability without altering outputs

#### `severity_threshold` (float)

Minimum severity score required to trigger repair.

- **Type**: Float
- **Range**: 0.0 to 1.0
- **Default**: 0.3
- **Description**: Only repairs responses with severity >= threshold

**Severity interpretation:**
- `0.0`: No aphasia symptoms
- `0.0-0.3`: Mild symptoms (default: no repair)
- `0.3-0.7`: Moderate symptoms (default: repair)
- `0.7-1.0`: Severe symptoms (default: repair)

**Tuning guidance:**
- Lower threshold (e.g., 0.1): More aggressive repair, fewer telegraphic responses
- Higher threshold (e.g., 0.7): Less aggressive repair, only fix severe cases
- Default (0.3): Balanced approach validated in effectiveness testing

**Example:**

```yaml
aphasia:
  detect_enabled: true      # Enable detection
  repair_enabled: true      # Enable repair
  severity_threshold: 0.3   # Repair moderate-to-severe cases
```

**Common Configurations:**

1. **Full Protection (Default):**
   ```yaml
   aphasia:
     detect_enabled: true
     repair_enabled: true
     severity_threshold: 0.3
   ```

2. **Monitoring Only:**
   ```yaml
   aphasia:
     detect_enabled: true
     repair_enabled: false
     severity_threshold: 0.3  # Ignored when repair disabled
   ```

3. **Disabled:**
   ```yaml
   aphasia:
     detect_enabled: false
     repair_enabled: false  # Ignored when detect disabled
     severity_threshold: 0.3  # Ignored when detect disabled
   ```

4. **Aggressive Repair:**
   ```yaml
   aphasia:
     detect_enabled: true
     repair_enabled: true
     severity_threshold: 0.1  # Repair even mild symptoms
   ```

**Metrics:**

All aphasia metrics in `EFFECTIVENESS_VALIDATION_REPORT.md` were measured with default configuration (detect=true, repair=true, threshold=0.3).

### NeuroLang Performance Configuration

Configuration for NeuroLang training behavior and resource usage in NeuroLangWrapper.

**Note:** These settings only apply when using `NeuroLangWrapper` from `mlsdm.extensions`. The base `LLMWrapper` does not use these settings.

#### `mode` (string)

Training mode for NeuroLang grammar models.

- **Type**: String (enum)
- **Allowed values**: `"eager_train"`, `"lazy_train"`, `"disabled"`
- **Default**: `"eager_train"`
- **Production recommendation**: `"disabled"` (minimal resource usage)

**Mode descriptions:**

1. **`eager_train`** - Trains models immediately at initialization
   - **Use case**: R&D, development, experimentation
   - **Resource impact**: High initial CPU/GPU usage during startup
   - **Best for**: Local development, model research

2. **`lazy_train`** - Trains models on first generation call
   - **Use case**: Demo servers, testing environments
   - **Resource impact**: Delayed training until first request
   - **Best for**: Delayed initialization scenarios

3. **`disabled`** - Skips NeuroLang entirely (recommended for production)
   - **Use case**: Production deployments, low-resource environments
   - **Resource impact**: Zero NeuroLang overhead
   - **What's included**: Cognitive controller + Aphasia-Broca detection
   - **What's excluded**: NeuroLang grammar models, training
   - **Best for**: Production with minimal resource footprint

#### `checkpoint_path` (string or null)

Path to pre-trained checkpoint file.

- **Type**: String (file path) or null
- **Default**: null
- **Description**: When provided, loads pre-trained model weights instead of training
- **Requires**: `mode` must be `"eager_train"` or `"lazy_train"` (ignored if `"disabled"`)

**Creating checkpoints:**

```bash
# Train models offline and save checkpoint
python scripts/train_neurolang_grammar.py --epochs 3 --output config/neurolang_grammar.pt
```

**When checkpoint is provided:**
- No training occurs at initialization or runtime
- Models are loaded from checkpoint file
- Significantly faster startup
- Recommended for production if not using `disabled` mode

**Example:**

```yaml
neurolang:
  mode: "disabled"              # Recommended for production
  checkpoint_path: "config/neurolang_grammar.pt"
```

**Common Configurations:**

1. **Production (Recommended):**
   ```yaml
   neurolang:
     mode: "disabled"
     checkpoint_path: null  # Not used when disabled
   ```

2. **Production with NeuroLang (Pre-trained):**
   ```yaml
   neurolang:
     mode: "eager_train"
     checkpoint_path: "config/neurolang_grammar.pt"
   ```

3. **Development (Train on startup):**
   ```yaml
   neurolang:
     mode: "eager_train"
     checkpoint_path: null  # Train from scratch
   ```

4. **Demo (Lazy loading):**
   ```yaml
   neurolang:
     mode: "lazy_train"
     checkpoint_path: null  # Train on first request
   ```

**Resource Comparison:**

| Mode | Startup Time | Runtime Overhead | Memory Usage | Production Ready |
|------|-------------|------------------|--------------|------------------|
| `disabled` | Instant | Zero | Minimal | ✅ Yes |
| `eager_train` (checkpoint) | Fast | Low | Moderate | ⚠️ Maybe |
| `eager_train` (no checkpoint) | Slow | Low | Moderate | ❌ No |
| `lazy_train` | Instant | Medium (first call) | Moderate | ❌ No |

**Using with ConfigLoader:**

```python
from mlsdm.utils.config_loader import ConfigLoader

config = ConfigLoader.load_config("config/production.yaml")
neurolang_params = ConfigLoader.get_neurolang_config_from_dict(config)
# Returns: {
#   "neurolang_mode": str,
#   "neurolang_checkpoint_path": str | None
# }
```

## Environment Variables

All configuration parameters can be overridden using environment variables with the `MLSDM_` prefix.

### Naming Convention

- Top-level keys: `MLSDM_<KEY>`
- Nested keys: `MLSDM_<SECTION>__<KEY>`
- Use uppercase for environment variables
- Use double underscores (`__`) for nesting

### Examples

```bash
# System configuration
export MLSDM_DIMENSION=768
export MLSDM_STRICT_MODE=false

# Multi-level memory
export MLSDM_MULTI_LEVEL_MEMORY__LAMBDA_L1=0.6
export MLSDM_MULTI_LEVEL_MEMORY__LAMBDA_L2=0.15
export MLSDM_MULTI_LEVEL_MEMORY__LAMBDA_L3=0.02

# Moral filter
export MLSDM_MORAL_FILTER__THRESHOLD=0.7
export MLSDM_MORAL_FILTER__ADAPT_RATE=0.1

# Cognitive rhythm
export MLSDM_COGNITIVE_RHYTHM__WAKE_DURATION=10
export MLSDM_COGNITIVE_RHYTHM__SLEEP_DURATION=4

# Aphasia-Broca configuration (NeuroLangWrapper only)
export MLSDM_APHASIA__DETECT_ENABLED=true
export MLSDM_APHASIA__REPAIR_ENABLED=true
export MLSDM_APHASIA__SEVERITY_THRESHOLD=0.3

# NeuroLang performance configuration (NeuroLangWrapper only)
export MLSDM_NEUROLANG__MODE=disabled
export MLSDM_NEUROLANG__CHECKPOINT_PATH=config/neurolang_grammar.pt
```

### Using .env File

Create `.env` from `env.example`:

```bash
cp env.example .env
# Edit .env with your values
```

## Validation

### Schema Validation

All configurations are validated against the schema defined in `src/utils/config_schema.py`.

**Validation checks:**

- ✅ Type checking (int, float, bool, list)
- ✅ Range constraints (min/max values)
- ✅ Hierarchical constraints (e.g., lambda_l3 ≤ lambda_l2 ≤ lambda_l1)
- ✅ Cross-field validation (e.g., ontology vectors match dimension)
- ✅ Unknown field rejection

### Error Messages

Validation errors provide clear, actionable messages:

```
Configuration validation failed for 'config/custom.yaml':
Decay rates must follow hierarchy: lambda_l3 (0.1) <= lambda_l2 (0.05) <= lambda_l1 (0.5)

Please check your configuration file against the schema documentation 
in src/utils/config_schema.py
```

### Disable Validation

```python
# Not recommended for production
config = ConfigLoader.load_config("config/custom.yaml", validate=False)
```

## Examples

### Production Deployment

```yaml
# config/production.yaml
dimension: 384

multi_level_memory:
  lambda_l1: 0.5
  lambda_l2: 0.1
  lambda_l3: 0.01
  theta_l1: 1.0
  theta_l2: 2.0
  gating12: 0.5
  gating23: 0.3

moral_filter:
  threshold: 0.5
  adapt_rate: 0.05
  min_threshold: 0.3
  max_threshold: 0.9

cognitive_rhythm:
  wake_duration: 8
  sleep_duration: 3

strict_mode: false
```

### High-Throughput Configuration

Optimized for maximum throughput:

```yaml
dimension: 384

multi_level_memory:
  lambda_l1: 0.7  # Faster decay
  lambda_l2: 0.2
  lambda_l3: 0.05
  theta_l1: 1.5   # Higher consolidation thresholds
  theta_l2: 3.0
  gating12: 0.3   # Lower gating = less consolidation
  gating23: 0.2

moral_filter:
  threshold: 0.4  # More permissive
  adapt_rate: 0.1 # Faster adaptation
  min_threshold: 0.2
  max_threshold: 0.8

cognitive_rhythm:
  wake_duration: 10  # Longer wake cycles
  sleep_duration: 2

strict_mode: false
```

### Strict Governance Configuration

Optimized for strict content control:

```yaml
dimension: 384

multi_level_memory:
  lambda_l1: 0.3  # Slower decay = longer memory
  lambda_l2: 0.05
  lambda_l3: 0.005
  theta_l1: 0.8
  theta_l2: 1.5
  gating12: 0.7   # More consolidation
  gating23: 0.5

moral_filter:
  threshold: 0.7  # Stricter filtering
  adapt_rate: 0.03 # Slower adaptation
  min_threshold: 0.5
  max_threshold: 0.95

cognitive_rhythm:
  wake_duration: 6
  sleep_duration: 4  # More consolidation time

strict_mode: false  # Still false for performance
```

## Best Practices

### Production Deployments

1. **Always use validation** in production to catch configuration errors early
2. **Use environment variables** for secrets and environment-specific settings
3. **Start with production.yaml** template and customize
4. **Keep strict_mode=false** for performance
5. **Monitor moral filter threshold** adjustments over time

### Configuration Management

1. **Version control** configuration files (except .env)
2. **Document custom values** with inline comments
3. **Test configuration changes** in staging before production
4. **Use secrets management** for API keys (AWS Secrets Manager, etc.)
5. **Validate after changes** using the schema

### Tuning Guidelines

**Memory hierarchy:**
- Start with defaults
- Increase lambda values for faster forgetting
- Adjust theta values to control consolidation frequency

**Moral filter:**
- Monitor acceptance rate (target: 50-70%)
- Increase threshold for stricter governance
- Adjust adapt_rate based on content variability

**Cognitive rhythm:**
- Typical ratio: wake = 2-3× sleep
- Increase wake duration for more processing
- Increase sleep duration for more consolidation

## Troubleshooting

### Common Issues

#### "Configuration file not found"

```
FileNotFoundError: Configuration file not found: config/my_config.yaml
```

**Solution**: Check file path and ensure file exists.

#### "Decay rates must follow hierarchy"

```
ValueError: Decay rates must follow hierarchy: lambda_l3 (0.2) <= lambda_l2 (0.1) <= lambda_l1 (0.5)
```

**Solution**: Ensure lambda_l3 ≤ lambda_l2 ≤ lambda_l1.

#### "Ontology vector dimension mismatch"

```
ValueError: Ontology vector dimension (768) must match system dimension (384)
```

**Solution**: Update ontology vectors to match system dimension or change dimension.

#### "Unknown configuration field"

```
ValidationError: Extra inputs are not permitted
```

**Solution**: Remove unknown fields or check spelling against schema.

### Validation Troubleshooting

To see full schema and constraints:

```python
from mlsdm.utils.config_schema import SystemConfig, validate_config_dict

# Print schema
print(SystemConfig.model_json_schema())

# Get default config
default = validate_config_dict({})
print(default.model_dump_json(indent=2))
```

### Debug Configuration Loading

```python
from mlsdm.utils.config_loader import ConfigLoader

# Load with validation to see detailed errors
try:
    config = ConfigLoader.load_config("config/custom.yaml", validate=True)
    print("Configuration loaded successfully")
except ValueError as e:
    print(f"Validation error: {e}")
```

## See Also

- [Deployment Guide](DEPLOYMENT_GUIDE.md) - Production deployment patterns
- [API Reference](API_REFERENCE.md) - API documentation
- [Testing Strategy](TESTING_STRATEGY.md) - Testing configurations
- Schema source: `mlsdm/utils/config_schema.py`
- Loader source: `mlsdm/utils/config_loader.py`

---

**Need help?** Open an issue on [GitHub](https://github.com/neuron7x/mlsdm-governed-cognitive-memory/issues).
