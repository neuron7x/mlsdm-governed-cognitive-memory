# Test Strategy for MLSDM

This document describes the test organization, coverage strategy, and test execution tiers for the MLSDM (Multi-Level Synaptic-Driven Memory) project.

## Test Organization

### Directory Structure

Tests are organized by purpose and scope:

- **`tests/unit/`** - Pure unit tests for individual modules
  - Fast, deterministic, no I/O
  - Mock external dependencies
  - Target: All critical business logic

- **`tests/state/`** - Unit-level tests for state persistence
  - Schema validation
  - Store/load operations
  - Migration testing
  - Organized separately for clarity but counted as unit tests in coverage

- **`tests/security/`** - Security-focused integration tests
  - RBAC, mTLS, OIDC, signing flows
  - Tests actual security module behavior
  - Includes adversarial input testing

- **`tests/integration/`** - Component integration tests
  - Multiple modules working together
  - May use test doubles for external services
  - Focus on API contracts and data flows

- **`tests/e2e/`** - End-to-end scenario tests
  - Full system integration
  - Realistic user workflows
  - May be slower due to setup/teardown

- **`tests/perf/`** - Performance and SLO tests
  - Latency measurements
  - Throughput validation
  - Memory usage checks

- **`tests/property/`** - Property-based tests (Hypothesis)
  - Invariant checking
  - Fuzz testing for edge cases
  - Can be computationally expensive

- **`tests/load/`** - Load and stress tests (Locust)
  - Concurrent user simulation
  - System behavior under stress
  - Excluded from standard CI runs

## Coverage Strategy

### Coverage Target: **68% minimum, 71% baseline**

The coverage gate enforces a minimum of **68%** to allow for minor fluctuations while maintaining high quality. Current baseline with unit+state tests is approximately **71%**.

### Module Classification

#### CRITICAL Modules (Target: ≥90% coverage)
- `src/mlsdm/core/*` - Core cognitive controller and state management
- `src/mlsdm/memory/*` (except experimental) - Memory systems
- `src/mlsdm/cognition/*` - Moral filter and decision-making
- `src/mlsdm/state/*` - State persistence and migrations
- `src/mlsdm/utils/config_*` - Configuration validation
- `src/mlsdm/utils/coherence_safety_metrics.py` - Safety metrics
- `src/mlsdm/api/health.py` - Health check endpoints

**Current Status**: ✅ All critical modules have ≥85% coverage

#### IMPORTANT Modules (Target: ≥70% coverage)
- `src/mlsdm/security/*` - Security modules (mTLS, OIDC, signing, RBAC, payload scrubbing)
- `src/mlsdm/observability/*` - Logging, metrics, tracing
- `src/mlsdm/router/*` - LLM routing logic
- `src/mlsdm/sdk/*` - SDK client

**Current Status**: ✅ Most important modules have 60-85% coverage, with active tests

#### NON-CRITICAL Modules (Coverage not strictly enforced)

These modules are **excluded** from strict coverage requirements because:

1. **Entrypoints** (`src/mlsdm/entrypoints/*`, `src/mlsdm/main.py`, `src/mlsdm/service/*`)
   - Thin wrappers around core functionality
   - Tested indirectly via E2E tests
   - Minimal logic to test in isolation
   - Justification: CLI/service entry points are glue code

2. **Experimental Modules** (`src/mlsdm/memory/experimental/*`)
   - GPU-accelerated memory (optional PyTorch dependency)
   - Research/benchmarking purposes
   - Not part of stable API
   - Justification: Experimental features require optional dependencies

3. **Optional Extensions** (`src/mlsdm/extensions/neuro_lang_extension.py`)
   - NeuroLang is an optional feature requiring PyTorch
   - Has dedicated integration tests when dependencies present
   - Justification: Optional feature with complex dependencies

### Coverage Measurement

Coverage is measured using:
```bash
pytest tests/unit/ tests/state/ --cov=src/mlsdm --cov-report=term-missing
```

The coverage gate script (`coverage_gate.sh`) enforces the minimum threshold and is run in CI.

## Test Tiers

Tests are organized into tiers for efficient feedback:

### Tier 0: Fast Smoke Tests (< 2 minutes)
```bash
pytest tests/unit -m "not slow" -q --maxfail=3
```
- Runs on every commit
- Fast unit tests only
- Fails fast on first few failures

### Tier 1: Standard CI Tests (< 10 minutes)
```bash
pytest -q --ignore=tests/load
```
- Runs on every PR and push to main
- Includes: unit, state, integration, security, perf, e2e (non-slow)
- Excludes: load tests (require special setup)

### Tier 2: Extended Validation (< 30 minutes)
```bash
pytest tests/property -v --maxfail=10
```
- Property-based tests (Hypothesis)
- May run nightly or on merge to main
- More comprehensive but slower

### Tier 3: Load & Stress Tests (manual or scheduled)
```bash
locust -f tests/load/locust_load_test.py
```
- Heavy concurrent load simulation
- Requires dedicated test environment
- Run before releases or on schedule

## Reproducing CI Failures Locally

### Basic Setup
```bash
# Install dependencies
pip install -e ".[dev,test]"

# Run the same tests as CI
pytest -q --ignore=tests/load
```

### Coverage Gate
```bash
./coverage_gate.sh
```

### Specific Test Failures
```bash
# Run a specific test file
pytest tests/unit/test_cognitive_controller.py -vv

# Run a specific test
pytest tests/unit/test_cognitive_controller.py::TestCognitiveControllerMemoryLeak -vv

# Run with verbose output
pytest tests/perf/test_slo_api_endpoints.py -vv -s
```

### Static Analysis
```bash
# Linting
ruff check src tests

# Type checking
mypy src/mlsdm

# Security scanning
bandit -r src/mlsdm
```

## Test Markers

Tests can be marked with pytest markers for selective execution:

- `@pytest.mark.slow` - Tests that take >5 seconds
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.unit` - Pure unit tests
- `@pytest.mark.property` - Property-based tests
- `@pytest.mark.security` - Security-focused tests

Example:
```bash
# Run only fast tests
pytest -m "not slow"

# Run only security tests
pytest -m security
```

## CI Workflow Mapping

- **`ci-neuro-cognitive-engine.yml`**: Tier 0 + Tier 1 tests
- **`property-tests.yml`**: Tier 2 property tests
- **`prod-gate.yml`**: Full validation before release
- **`sast-scan.yml`**: Security static analysis

## SLO and Performance Tests

Performance tests (`tests/perf/test_slo_api_endpoints.py`) validate:
- API latency (P50, P95, P99)
- Memory stability
- Error rates

These tests use:
- Warm-up phases to stabilize measurements
- Noise tolerance for CI variability
- Relative thresholds (e.g., readiness < liveness * factor)

## Memory Leak Tests

Memory leak tests (`tests/unit/test_cognitive_controller.py::TestCognitiveControllerMemoryLeak`) validate:
- No unbounded memory growth over time
- Stable RSS (Resident Set Size) after warm-up
- Deterministic behavior with fixed random seeds

## Maintenance Notes

### Adding New Tests
1. Choose appropriate directory based on test scope
2. Add appropriate markers (`@pytest.mark.slow`, etc.)
3. Ensure tests are deterministic and can run in parallel
4. Update this document if adding new test categories

### Updating Coverage Thresholds
1. Measure current coverage: `pytest --cov=src/mlsdm --cov-report=term`
2. Update `COVERAGE_MIN` in `coverage_gate.sh` if justified
3. Document reason in commit message and update `COVERAGE_REPORT_2025.md`

### Excluding Modules from Coverage
1. Only exclude non-critical glue code or optional features
2. Add `# pragma: no cover` to specific lines if needed
3. Document exclusion reason in this file
4. Consider adding smoke tests for excluded modules

## Future Improvements

- [ ] Add mutation testing for critical paths
- [ ] Implement contract testing for API boundaries
- [ ] Add visual regression tests for documentation
- [ ] Improve property test coverage for security modules
