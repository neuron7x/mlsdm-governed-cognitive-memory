# Metrics Source of Truth

**Last Updated**: December 26, 2025
**Purpose**: Single source for test coverage and quality metrics to prevent documentation drift.

---

## Evidence Snapshots

Metrics are sourced from **committed evidence snapshots** in the repository for reproducibility.

| Artifact | Path |
|----------|------|
| **Coverage Report** | `artifacts/evidence/<date>/<sha>/coverage/coverage.xml` |
| **JUnit Test Results** | `artifacts/evidence/<date>/<sha>/pytest/junit.xml` |
| **Benchmark Metrics** | `artifacts/evidence/<date>/<sha>/benchmarks/benchmark-metrics.json` |
| **Memory Footprint** | `artifacts/evidence/<date>/<sha>/memory/memory_footprint.json` |
| **Manifest** | `artifacts/evidence/<date>/<sha>/manifest.json` |

To regenerate evidence locally:

```bash
make evidence
```

---

## Coverage Metrics

| Metric | Value | Source |
|--------|-------|--------|
| **CI Coverage Threshold** | 65% | [ci-neuro-cognitive-engine.yml](../.github/workflows/ci-neuro-cognitive-engine.yml#L149) |
| **Actual Coverage** | See evidence snapshot | `artifacts/evidence/<date>/<sha>/coverage/coverage.xml` |
| **Core Modules Coverage** | 90%+ | Critical modules (`core/`, `memory/`, `cognition/`) |

### Why 65% Threshold When Actual is Higher?

The CI coverage threshold (65%) is intentionally set below actual coverage for several reasons:

1. **Anti-flap**: Prevents spurious CI failures from minor fluctuations in test coverage
2. **Developer Experience**: Allows focused work without CI blocking every small change
3. **Incremental Growth**: Threshold is increased when coverage consistently exceeds it by 5%+ for 2+ releases
4. **Safety Margin**: Provides headroom for refactoring without breaking CI

**Policy**: The threshold represents the *minimum acceptable* coverage, not the *target* coverage.

---

## Test Metrics

Test counts are derived from the committed JUnit evidence:

| Metric | Source |
|--------|--------|
| **Test Results** | `artifacts/evidence/<date>/<sha>/pytest/junit.xml` |

To get exact counts, parse the JUnit XML or run `make evidence`.

---

## Benchmark Metrics

Performance metrics are captured in the evidence snapshot:

| Metric | Source |
|--------|--------|
| **Benchmark Results** | `artifacts/evidence/<date>/<sha>/benchmarks/benchmark-metrics.json` |
| **Raw Latency Data** | `artifacts/evidence/<date>/<sha>/benchmarks/raw_neuro_engine_latency.json` |
| **Baseline** | `benchmarks/baseline.json` |

To check for benchmark drift:

```bash
python scripts/check_benchmark_drift.py artifacts/evidence/<date>/<sha>/benchmarks/benchmark-metrics.json
```

---

## CI Coverage Command

The canonical coverage command used in CI:

```bash
# From .github/workflows/ci-neuro-cognitive-engine.yml (coverage job, lines 148-149)
pytest --cov=src/mlsdm --cov-report=xml --cov-report=term-missing \
  --cov-fail-under=65 --ignore=tests/load -m "not slow and not benchmark" -v
```

**Use this exact command for local verification to match CI behavior.**

---

## Updating This Document

When evidence is regenerated:

1. Run `make evidence` to capture a new snapshot
2. Commit the new evidence folder under `artifacts/evidence/`
3. Update the "Last Updated" date above
4. If coverage exceeds threshold by 5%+ for 2+ releases, consider raising the threshold

---

## Related Documentation

- [TESTING_GUIDE.md](../TESTING_GUIDE.md) - How to write and run tests
- [CI_GUIDE.md](../CI_GUIDE.md) - CI/CD configuration overview
- [TEST_STRATEGY.md](../TEST_STRATEGY.md) - Test organization and priorities
- [Evidence README](../artifacts/evidence/README.md) - Evidence snapshot policy
