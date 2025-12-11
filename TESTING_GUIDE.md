# Testing Guide

**Document Version:** 1.2.0  
**Project Version:** 1.2.0  
**Last Updated:** December 2025  
**Test Coverage:** 70.85% (Overall) | 90%+ (Core Modules)

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

This guide provides comprehensive instructions for testing MLSDM Governed Cognitive Memory. The project maintains solid test coverage (70.85% overall, 90%+ for core modules) and follows industry best practices for unit, integration, and validation testing.

### Testing Philosophy

1. **Pragmatic Coverage**: Maintain ≥68% overall coverage (enforced), 90%+ for core modules
2. **Test Pyramid**: More unit tests, fewer integration tests
3. **Fast Execution**: Unit tests < 1ms, integration tests < 1s
4. **Reproducibility**: Deterministic tests with fixed seeds
5. **Clear Assertions**: Each test validates specific behavior

### Test Categories

| Category | Purpose | Location | Count | Execution Time |
|----------|---------|----------|-------|----------------|
| **Unit Tests** | Test individual components | `tests/unit/` | ~1,200 | <30s |
| **State Tests** | Test state persistence | `tests/state/` | ~31 | <5s |
| **Integration Tests** | Test component interactions | `tests/integration/` | ~50 | <10s |
| **Validation Tests** | Test effectiveness claims | `tests/validation/` | ~4 | <10s |
| **Property Tests** | Test invariants with Hypothesis | `tests/property/` | ~50 | <15s |
| **Security Tests** | Test STRIDE controls | `tests/security/` | ~38 | <5s |
| **E2E Tests** | Test full prod-level scenarios | `tests/e2e/` | ~28 | <10s |
| **Load Tests** | Test throughput (requires server) | `tests/load/` | ~3 | N/A |

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

# Run only E2E tests
pytest tests/e2e/ -v

# Run E2E tests (fast, production scenarios)
pytest tests/e2e -v -m "not slow"

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

### Test Organization

The MLSDM test suite is organized by scope and purpose to support different validation needs:

| Directory | Scope | Test Count | Coverage Focus |
|-----------|-------|------------|----------------|
| `tests/unit/` | Unit tests for all modules | ~1,200 | All source modules |
| `tests/state/` | State persistence unit tests | ~31 | State management |
| `tests/integration/` | Component integration | ~50 | Module interactions |
| `tests/validation/` | Effectiveness validation | ~4 | Key metrics |
| `tests/eval/` | Evaluation suites | ~9 | Aphasia, Sapolsky |
| `tests/property/` | Property-based tests | ~50 | Invariants |
| `tests/security/` | Security tests | ~38 | STRIDE controls |
| `tests/e2e/` | End-to-end scenarios | ~28 | Full workflows |
| `tests/load/` | Load tests (Locust) | ~3 | Throughput |
| **Total** | **Full Suite** | **~1,587** | **70.85% coverage** |

### Test Scopes

**Core Cognitive Modules Only** (used in CORE_IMPLEMENTATION_VALIDATION.md):
```bash
pytest tests/unit/test_cognitive_controller.py \
       tests/unit/test_llm_wrapper.py \
       tests/unit/test_*_memory.py \
       tests/unit/test_moral_filter*.py \
       tests/unit/test_cognitive_rhythm.py \
       --co -q
# Result: ~577 tests
```

**Full Coverage Suite** (used in coverage_gate.sh):
```bash
pytest tests/unit/ tests/state/ --cov=src/mlsdm
# Result: 1,587 tests, 70.85% coverage
```

### Directory Structure

```
tests/
├── unit/                # Unit tests for individual modules (~1,200 tests)
│   ├── test_cognitive_controller.py
│   ├── test_llm_wrapper.py
│   ├── test_*_memory.py
│   ├── test_moral_filter*.py
│   └── ... (all module tests)
├── state/               # State persistence tests (~31 tests)
│   └── test_system_state_integrity.py
├── integration/         # Component integration tests (~50 tests)
│   ├── test_end_to_end.py
│   └── test_llm_wrapper_integration.py
├── validation/          # Effectiveness validation tests (~4 tests)
│   ├── test_moral_filter_effectiveness.py
│   └── test_wake_sleep_effectiveness.py
├── eval/                # Evaluation suites (~9 tests)
│   ├── test_aphasia_eval_suite.py
│   └── test_sapolsky_suite.py
├── property/            # Property-based tests (~50 tests)
│   ├── test_pelm_phase_behavior.py
│   └── test_invariants_memory.py
├── security/            # Security tests (~38 tests)
│   ├── test_guardrails_stride.py
│   └── test_payload_scrubber.py
├── e2e/                 # End-to-end tests (~28 tests)
│   ├── test_e2e_scenarios.py
│   └── test_neuro_cognitive_engine_stub_backend.py
└── load/                # Load tests (~3 tests, requires server)
    └── locust_load_test.py
```

## Coverage Requirements

- **Minimum Coverage Threshold:** 68% (enforced by coverage_gate.sh)
- **Current Overall Coverage:** 70.85% (measured on tests/unit/ + tests/state/)
- **Core Modules Coverage:** 90%+ (cognitive controller, memory, moral filter)
- **Critical Modules:** Near 100% coverage achieved for:
  - `src/mlsdm/core/cognitive_controller.py` (97.05%)
  - `src/mlsdm/core/memory_manager.py` (100%)
  - `src/mlsdm/cognition/moral_filter.py` (100%)
  - `src/mlsdm/cognition/moral_filter_v2.py` (100%)
  - `src/mlsdm/utils/coherence_safety_metrics.py` (99.56%)

**Note**: Coverage measurement focuses on core modules. See [COVERAGE_REPORT_2025.md](COVERAGE_REPORT_2025.md) for detailed breakdown by module.

## Writing New Tests

### Test Naming Convention

- Test files: `test_*.py`
- Test classes: `Test*` (e.g., `TestCoherenceMetrics`)
- Test methods: `test_*` (e.g., `test_measure_temporal_consistency`)

### Example Test Structure

```python
import pytest
import numpy as np
from mlsdm.utils.coherence_safety_metrics import CoherenceSafetyAnalyzer

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
pytest --cov=src/mlsdm/utils/coherence_safety_metrics --cov-report=term-missing tests/ src/tests/unit/
```

### Finding Uncovered Lines

The coverage report shows uncovered lines:

```
src/mlsdm/utils/coherence_safety_metrics.py    169      0   100%
```

If there are missing lines, they will be listed:

```
src/mlsdm/utils/input_validator.py    87     12    86%   42, 59-60, 95-96, 131, 168, 206, 211, 241, 245-246
```

## Continuous Integration

### Pre-commit Checks

```bash
# Run before committing
pytest --cov=src --cov-fail-under=90 tests/ src/tests/unit/
```

### Coverage Threshold

The coverage threshold is configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = "--cov=src --cov-report=html --cov-fail-under=90"
```

## Troubleshooting

### Tests Fail to Import Modules

Make sure you're running tests from the repository root:

```bash
cd /path/to/mlsdm
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
4. **Review COVERAGE_REPORT_2025.md quarterly**
5. **Update tests when adding new features**

## E2E Test Suite

### Overview

The End-to-End (E2E) test suite validates that MLSDM works correctly as a complete system, exercising the full pipeline through external interfaces (HTTP API/Python API). These tests ensure production-level reliability.

### Running E2E Tests

```bash
# Run all E2E tests
pytest tests/e2e -v

# Run E2E tests excluding slow tests
pytest tests/e2e -m "not slow" -v

# Run with verbose output and timing
pytest tests/e2e -v --tb=short
```

### E2E Test Scenarios

The E2E test suite covers the following scenarios:

| Scenario | Test File | Purpose |
|----------|-----------|---------|
| **Happy Path Governed Chat** | `test_e2e_scenarios.py` | Validates non-toxic prompts are accepted with responses |
| **Toxic Rejection** | `test_e2e_scenarios.py` | Validates moral filter correctly rejects harmful prompts |
| **Aphasia Detection & Repair** | `test_e2e_scenarios.py` | Validates telegraphic speech detection and repair |
| **Secure Mode** | `test_e2e_scenarios.py` | Validates secure mode operation without training |
| **Memory Phase Rhythm** | `test_e2e_scenarios.py` | Validates wake/sleep phase alternation |
| **Metrics Exposed** | `test_e2e_scenarios.py` | Validates `/health/metrics` endpoint returns data |
| **Core Happy Path** | `test_end_to_end_core.py` | Validates basic CognitiveController flow |
| **Engine Stub Backend** | `test_neuro_cognitive_engine_stub_backend.py` | Validates complete NeuroCognitiveEngine pipeline |

### Key E2E Invariants Tested

1. **Moral Filter**: Toxic prompts with high moral threshold are rejected
2. **Aphasia Repair**: Telegraphic text is detected with severity > 0
3. **Rhythm**: Wake/sleep phases alternate correctly
4. **Memory**: Events in sleep phase are handled (not stored)
5. **Telemetry/Metrics**: `/health/metrics` returns Prometheus-formatted data
6. **Latency**: Responses under 100ms (soft upper bound)

### CI Integration

E2E tests are mandatory for release:
- Job: `e2e-tests` in `ci-neuro-cognitive-engine.yml`
- Gate: Part of `All CI Checks Passed` job
- Runtime: < 60 seconds (currently ~2-3s)

### When to Run E2E Tests

- **Always**: Before merging PRs
- **Always**: Before releases
- **Recommended**: After changing core modules
- **Optional**: During local development (fast enough for frequent runs)

## References

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [ESA Standards 2025](../COVERAGE_REPORT_2025.md)
- [MLSDM Coverage Report](../COVERAGE_REPORT_2025.md)
