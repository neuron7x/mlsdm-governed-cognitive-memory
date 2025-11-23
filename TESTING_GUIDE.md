# Testing Guide

**Document Version:** 1.0.0  
**Project Version:** 1.0.0  
**Last Updated:** November 2025  
**Test Coverage:** 97.63%

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Test Structure](#test-structure)
- [Coverage Requirements](#coverage-requirements)
- [Writing New Tests](#writing-new-tests)
- [Coverage Analysis](#coverage-analysis)
- [Continuous Integration](#continuous-integration)
- [Troubleshooting](#troubleshooting)
- [Standards Compliance](#standards-compliance)

---

## Overview

This guide provides comprehensive instructions for testing MLSDM Governed Cognitive Memory. The project maintains high test coverage (97.63%) and follows industry best practices for unit, integration, and validation testing.

### Testing Philosophy

1. **Comprehensive Coverage**: Maintain ≥90% code coverage
2. **Test Pyramid**: More unit tests, fewer integration tests
3. **Fast Execution**: Unit tests < 1ms, integration tests < 1s
4. **Reproducibility**: Deterministic tests with fixed seeds
5. **Clear Assertions**: Each test validates specific behavior

### Test Categories

| Category | Purpose | Location | Count | Execution Time |
|----------|---------|----------|-------|----------------|
| **Unit Tests** | Test individual components | `src/tests/unit/` | 150+ | <2s |
| **Integration Tests** | Test component interactions | `tests/integration/` | 10+ | <5s |
| **Validation Tests** | Test effectiveness claims | `tests/validation/` | 8+ | <10s |
| **Property Tests** | Test invariants with fuzzing | `src/tests/unit/test_property_based.py` | 20+ | <3s |

---

## Quick Start

### Running Tests

```bash
# Run all tests with coverage
pytest --cov=src --cov-report=html --cov-report=term tests/ src/tests/unit/

# Run only unit tests
pytest src/tests/unit/ -v

# Run only integration tests
pytest tests/integration/ -v

# Run only validation tests
pytest tests/validation/ -v

# Run specific test file
pytest src/tests/unit/test_coherence_safety_metrics.py -v
```

### Viewing Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html tests/ src/tests/unit/

# Open in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Test Structure

```
tests/
├── integration/          # Integration tests for end-to-end workflows
│   ├── test_end_to_end.py
│   └── test_llm_wrapper_integration.py
└── validation/          # Effectiveness validation tests
    ├── test_moral_filter_effectiveness.py
    └── test_wake_sleep_effectiveness.py

src/tests/unit/          # Unit tests for individual components
├── test_coherence_safety_metrics.py
├── test_api.py
├── test_cognitive_controller.py
├── test_config_loader.py
├── test_config_validator.py
├── test_data_serializer.py
├── test_llm_wrapper.py
├── test_llm_wrapper_unit.py
├── test_memory_manager.py
├── test_moral_filter_v2.py
├── test_performance.py
├── test_property_based.py
├── test_qilm_v2.py
└── test_security.py
```

## Coverage Requirements

- **Minimum Coverage:** 90% (enforced by pyproject.toml)
- **Current Coverage:** 97.63%
- **Critical Modules:** 100% coverage required for:
  - `src/core/cognitive_controller.py`
  - `src/core/memory_manager.py`
  - `src/cognition/moral_filter.py`
  - `src/utils/coherence_safety_metrics.py`
  - `src/utils/security_logger.py`

## Writing New Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*` (e.g., `TestCoherenceMetrics`)
- Test methods: `test_*` (e.g., `test_measure_temporal_consistency`)

### Example Test Structure

```python
import pytest
import numpy as np
from src.utils.coherence_safety_metrics import CoherenceSafetyAnalyzer

class TestMyComponent:
    """Test suite for MyComponent"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return CoherenceSafetyAnalyzer()
    
    def test_basic_functionality(self, analyzer):
        """Test basic functionality"""
        result = analyzer.measure_temporal_consistency([])
        assert result == 1.0
    
    def test_edge_case_empty_input(self, analyzer):
        """Test edge case with empty input"""
        result = analyzer.measure_semantic_coherence([], [])
        assert result == 0.0
```

### Edge Cases to Test

1. **Empty Inputs** - Test with empty lists, None values
2. **Boundary Values** - Test with 0, 1, min, max values
3. **Large Datasets** - Test with 1000+ samples
4. **Numerical Precision** - Test for floating-point errors
5. **Random Data** - Use fixtures with seeded random generators

### Example: Testing with Random Data

```python
class TestEdgeCases:
    @pytest.fixture(autouse=True)
    def set_random_seed(self):
        """Set random seed for reproducible tests"""
        np.random.seed(42)
    
    def test_with_random_data(self):
        """Test with random data"""
        data = np.random.randn(100, 128)
        # Your test logic here
```

## Coverage Analysis

### Checking Coverage for Specific Module

```bash
pytest --cov=src/utils/coherence_safety_metrics --cov-report=term-missing tests/ src/tests/unit/
```

### Finding Uncovered Lines

The coverage report shows uncovered lines:

```
src/utils/coherence_safety_metrics.py    169      0   100%
```

If there are missing lines, they will be listed:

```
src/utils/input_validator.py    87     12    86%   42, 59-60, 95-96, 131, 168, 206, 211, 241, 245-246
```

## Continuous Integration

### GitHub Actions Workflows

The project uses three specialized CI workflows plus a comprehensive release pipeline:

#### 1. **CI - Neuro Cognitive Engine** (`.github/workflows/ci-neuro-cognitive-engine.yml`)
- **When it runs**: Every push/PR to `main` or `feature/*` branches
- **What it tests**:
  - `tests-core`: Unit, integration, eval, validation, security, aphasia tests (Python 3.10 & 3.11)
  - Property tests run separately in their own workflow
  - `tests-benchmarks`: Performance benchmarks (advisory)
  - `tests-eval-sapolsky`: Cognitive safety evaluation (advisory)
  - `security-trivy`: Docker vulnerability scanning (advisory)
- **Required for PR merge**: `tests-core` for both Python versions
- **Status badge**: Shows overall health of core functionality

#### 2. **Property-Based Tests** (`.github/workflows/property-tests.yml`)
- **When it runs**: Every push/PR to `main` or `feature/*` branches (runs in parallel with core tests)
- **What it tests**:
  - `property-tests`: Hypothesis-based invariant verification (Python 3.10 & 3.11)
  - `counterexamples-regression`: Known issue regression tests
  - `invariant-coverage`: Invariant documentation validation
- **Required for PR merge**: `property-tests` for both Python versions
- **Determinism**: Uses `PYTHONHASHSEED=0` and `HYPOTHESIS_PROFILE=ci`
- **Separation rationale**: Property tests are separated to provide distinct status signal and avoid duplication

#### 3. **Aphasia / NeuroLang CI** (`.github/workflows/aphasia-ci.yml`)
- **When it runs**: Push/PR when Aphasia/NeuroLang code changes
- **What it tests**:
  - Aphasia detection and speech processing tests
  - NeuroLang optional dependency packaging
  - AphasiaEvalSuite quality gate
- **Required for PR merge**: Core tests (quality gate is advisory)
- **Specialized**: Only runs when related paths change

#### 4. **Release Pipeline** (`.github/workflows/release.yml`)
- **When it runs**: Push of version tags (e.g., `v1.2.0`)
- **What it does**:
  - Runs comprehensive test suite with coverage
  - Validates benchmarks and cognitive safety
  - Builds and pushes Docker image
  - Publishes to TestPyPI
  - Creates GitHub release with changelog
- **Required**: All tests must pass before release artifacts are built

### Pre-commit Checks

Run the same commands that CI uses to catch issues early:

```bash
# Core tests (same as CI tests-core job)
export PYTHONHASHSEED=0
pytest -q --ignore=tests/load --ignore=benchmarks --ignore=tests/property

# Property tests (same as CI property-tests job)
export PYTHONHASHSEED=0
export HYPOTHESIS_PROFILE=ci
pytest tests/property/ -q --maxfail=5

# Full suite with coverage (excluding property tests which run separately)
pytest --cov=src --cov-fail-under=90 --cov-report=term-missing -v --ignore=tests/load --ignore=tests/property

# Run both core and property tests together
export PYTHONHASHSEED=0
export HYPOTHESIS_PROFILE=ci
pytest -q --ignore=tests/load --ignore=benchmarks && pytest tests/property/ -q --maxfail=5
```

### CI Status Checks

Each PR shows status checks for:
- ✅ `tests-core (3.10)` - Core tests on Python 3.10 (required)
- ✅ `tests-core (3.11)` - Core tests on Python 3.11 (required)
- ✅ `property-tests (3.10)` - Property tests on Python 3.10 (required)
- ✅ `property-tests (3.11)` - Property tests on Python 3.11 (required)
- ℹ️ `tests-benchmarks` - Performance validation (informational)
- ℹ️ `tests-eval-sapolsky` - Cognitive safety (informational)
- ℹ️ `security-trivy` - Security scan (informational)

### Coverage Threshold

The coverage threshold is configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = "--cov=src --cov-report=html --cov-fail-under=90"
```

### Viewing CI Results

1. **In Pull Requests**: Check the "Checks" tab to see all workflow runs
2. **Artifacts**: Download test reports, benchmark results, and evaluation outputs
3. **Security**: View Trivy scan results in the "Security" tab
4. **Logs**: Click on failed jobs to see detailed test output

For detailed CI architecture and design principles, see [TESTING_STRATEGY.md](TESTING_STRATEGY.md#4-continuous-integration-test-pipelines).

## Troubleshooting

### Tests Fail to Import Modules

Make sure you're running tests from the repository root:

```bash
cd /path/to/mlsdm-governed-cognitive-memory
pytest --cov=src tests/ src/tests/unit/
```

### Coverage Report Not Generated

Install pytest-cov:

```bash
pip install pytest-cov
```

### Test Warnings

Deprecation warnings are informational and don't affect test results. To hide them:

```bash
pytest --cov=src -W ignore::DeprecationWarning tests/ src/tests/unit/
```

## ESA Standards Compliance

For TRL 5-6 compliance:

1. **Run full test suite before releases**
2. **Ensure ≥90% coverage on all PRs**
3. **Document any coverage regressions**
4. **Review COVERAGE_REPORT.md quarterly**
5. **Update tests when adding new features**

## References

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [ESA Standards 2025](../COVERAGE_REPORT.md)
- [MLSDM Coverage Report](../COVERAGE_REPORT.md)
