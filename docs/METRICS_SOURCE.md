# Metrics Source of Truth

**Last Updated**: December 18, 2025
**Purpose**: Single source for test coverage and quality metrics to prevent documentation drift.

---

## Coverage Metrics

| Metric | Value | Source |
|--------|-------|--------|
| **CI Coverage Threshold** | 65% | [ci-neuro-cognitive-engine.yml](../.github/workflows/ci-neuro-cognitive-engine.yml#L150) |
| **Actual Coverage** | ~86% | [Latest CI run coverage.xml artifact](https://github.com/neuron7x/mlsdm/actions/workflows/ci-neuro-cognitive-engine.yml) |
| **Core Modules Coverage** | 90%+ | Critical modules (`core/`, `memory/`, `cognition/`) |

### Why 65% Threshold When 86% Actual?

The CI coverage threshold (65%) is intentionally set below actual coverage (~86%) for several reasons:

1. **Anti-flap**: Prevents spurious CI failures from minor fluctuations in test coverage
2. **Developer Experience**: Allows focused work without CI blocking every small change
3. **Incremental Growth**: Threshold is increased when coverage consistently exceeds it by 5%+ for 2+ releases
4. **Safety Margin**: Provides ~21% headroom for refactoring without breaking CI

**Policy**: The threshold represents the *minimum acceptable* coverage, not the *target* coverage.

---

## Test Metrics

| Metric | Value | Source |
|--------|-------|--------|
| **Total Tests** | ~3,600 | [Latest CI run](https://github.com/neuron7x/mlsdm/actions/workflows/ci-neuro-cognitive-engine.yml) |
| **Unit Tests** | ~1,900 | `tests/unit/` |
| **Integration Tests** | ~50 | `tests/integration/` |
| **E2E Tests** | ~28 | `tests/e2e/` |
| **Property Tests** | ~50 | `tests/property/` |
| **Security Tests** | ~38 | `tests/security/` |

---

## CI Coverage Command

The canonical coverage command used in CI:

```bash
# From .github/workflows/ci-neuro-cognitive-engine.yml (coverage job)
pytest --cov=src/mlsdm --cov-report=xml --cov-report=term-missing \
  --cov-fail-under=65 --ignore=tests/load -m "not slow and not benchmark" -v
```

**Use this exact command for local verification to match CI behavior.**

---

## Updating This Document

When coverage improves significantly:

1. Run the CI and check the coverage artifact
2. Update the "Actual Coverage" value above with the new figure
3. Update the "Last Updated" date
4. If coverage exceeds threshold by 5%+ for 2+ releases, consider raising the threshold

---

## Related Documentation

- [TESTING_GUIDE.md](../TESTING_GUIDE.md) - How to write and run tests
- [CI_GUIDE.md](../CI_GUIDE.md) - CI/CD configuration overview
- [TEST_STRATEGY.md](../TEST_STRATEGY.md) - Test organization and priorities
