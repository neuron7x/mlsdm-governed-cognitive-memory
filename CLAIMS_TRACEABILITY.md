# Claims Traceability Matrix

**Document Version:** 1.0.0  
**Last Updated:** November 2025  
**Status:** Production

This document maps theoretical claims and documented metrics to their validation evidence. Use this for quick verification that all stated metrics are backed by tests, benchmarks, or appropriately marked as future work.

---

## Table of Contents

- [Status Legend](#status-legend)
- [A. Fully Backed Claims (Code + Tests + Metrics)](#a-fully-backed-claims)
- [B. Partially Backed Claims (Code Exists, Needs Clarification)](#b-partially-backed-claims)
- [C. Future Work / Hypotheses](#c-future-work--hypotheses)
- [Metric Sources](#metric-sources)

---

## Status Legend

| Status | Description |
|--------|-------------|
| ✅ **Backed** | Claim is verified by tests/benchmarks with reproducible results |
| ⚠️ **Partial** | Code exists but metrics may vary or corpus is limited |
| 📝 **Future Work** | Clearly labeled as planned/experimental, not production claim |

---

## A. Fully Backed Claims

### Safety & Moral Filtering

| Claim | Value | Source Test/Benchmark | Notes |
|-------|-------|----------------------|-------|
| Toxic Content Rejection Rate | 93.3% | `tests/validation/test_moral_filter_effectiveness.py::test_moral_filter_toxic_rejection` | Reproducible with seed=42 |
| Comprehensive Toxic Rejection | 97.8% | `tests/validation/test_moral_filter_effectiveness.py::test_comprehensive_safety_metrics` | Aggregated safety metrics |
| False Positive Rate | 37.5% | `tests/validation/test_moral_filter_effectiveness.py::test_moral_filter_false_positive_rate` | Acceptable trade-off for safety |
| Bounded Drift Under Attack | 0.33 | `tests/validation/test_moral_filter_effectiveness.py::test_moral_drift_stability` | 70% toxic bombardment test |
| Threshold Range | [0.30, 0.90] | `tests/unit/test_moral_filter.py` | Formal invariant, property-tested |

### Wake/Sleep Cycles

| Claim | Value | Source Test/Benchmark | Notes |
|-------|-------|----------------------|-------|
| Processing Load Reduction | 89.5% | `tests/validation/test_wake_sleep_effectiveness.py::test_wake_sleep_resource_efficiency` | During sleep phase |
| Coherence Improvement | 5.5% | `tests/validation/test_wake_sleep_effectiveness.py::test_comprehensive_coherence_metrics` | Overall score improvement |
| Phase Separation Score | 0.51 vs 0.38 | `tests/validation/test_wake_sleep_effectiveness.py::test_comprehensive_coherence_metrics` | With vs without rhythm |
| Wake Duration | 8 steps (configurable) | `tests/validation/test_rhythm_state_machine.py` | Default configuration |
| Sleep Duration | 3 steps (configurable) | `tests/validation/test_rhythm_state_machine.py` | Default configuration |

### Memory System

| Claim | Value | Source Test/Benchmark | Notes |
|-------|-------|----------------------|-------|
| Memory Footprint | 29.37 MB | `tests/property/test_invariants_memory.py::test_pelm_capacity_enforcement` | Fixed allocation |
| PELM Capacity | 20,000 vectors | `tests/property/test_invariants_memory.py` | Hard limit enforced |
| Zero Allocation After Init | Yes | Property tests | Circular buffer, no heap growth |

### Performance

| Claim | Value | Source Test/Benchmark | Notes |
|-------|-------|----------------------|-------|
| Pre-flight Latency P95 | < 20ms (actual: < 1ms) | `benchmarks/test_neuro_engine_performance.py::test_benchmark_pre_flight_latency` | Stub backend |
| End-to-End Latency P95 | < 500ms (actual: ~23ms) | `benchmarks/test_neuro_engine_performance.py::test_benchmark_end_to_end_small_load` | Stub backend |
| Concurrent Requests | 1,000+ | `tests/load/locust_load_test.py` | Requires running server |
| Thread Safety | Zero data races | `tests/property/test_concurrency_safety.py` | Lock-based protection |

---

## B. Partially Backed Claims

### Aphasia-Broca Detection

| Claim | Value | Source Test/Benchmark | Status | Notes |
|-------|-------|----------------------|--------|-------|
| Telegraphic Response Reduction | 87.2% | `APHASIA_SPEC.md` | ⚠️ Partial | Based on empirical study (1,000 samples) not in repo. Test corpus has 5 samples. |
| True Positive Rate | 100% | `tests/eval/test_aphasia_eval_suite.py` | ⚠️ Partial | Verified but corpus limited (5 telegraphic samples) |
| True Negative Rate | 80% | `tests/eval/test_aphasia_eval_suite.py` | ⚠️ Partial | 20% false positive rate on small corpus |
| Mean Severity | 0.89 | `tests/eval/aphasia_eval_suite.py` | ✅ Backed | From corpus evaluation |
| Detection Thresholds | avg_len≥6, func≥0.15, frag≤0.5 | `tests/validation/test_aphasia_detection.py` | ✅ Backed | Configurable via constructor |

**Clarification**: The 87.2% metric comes from a reported internal study on 1,000 LLM responses. The repository contains a validation test suite with a smaller corpus (5 telegraphic + 5 normal samples) that verifies detection logic works correctly. The detection algorithm is sound but the 87.2% figure requires external validation with a larger corpus.

### Throughput Claims

| Claim | Value | Source Test/Benchmark | Status | Notes |
|-------|-------|----------------------|--------|-------|
| Maximum Verified RPS | 5,500 ops/sec | Documentation | ⚠️ Partial | Mentioned in ARCHITECTURE_SPEC.md, requires load test with server |
| Sustained Target | 1,000 RPS | `SLO_SPEC.md` | ⚠️ Partial | SLO target, verified via Locust but requires server deployment |

**Clarification**: These throughput claims require running the actual server and load testing infrastructure. The Locust test file exists (`tests/load/locust_load_test.py`) but cannot be run in CI without server deployment.

---

## C. Future Work / Hypotheses

The following items are clearly marked as future work or hypotheses in the documentation:

### Planned Enhancements (Not Implemented)

| Item | Status | Location | Notes |
|------|--------|----------|-------|
| RAG Hallucination Rate (ragas) | 📝 Future Work | `EFFECTIVENESS_VALIDATION_REPORT.md` | Requires retrieval setup |
| TLA+ Formal Verification | 📝 Future Work | `EFFECTIVENESS_VALIDATION_REPORT.md` | Requires formal methods expertise |
| Coq Proofs | 📝 Future Work | `EFFECTIVENESS_VALIDATION_REPORT.md` | Requires formal methods expertise |
| Chaos Engineering Suite | 📝 Future Work | `EFFECTIVENESS_VALIDATION_REPORT.md` | Requires staging environment |
| Soak Testing (48-72h) | 📝 Future Work | `EFFECTIVENESS_VALIDATION_REPORT.md` | Long-duration testing |
| OpenTelemetry Distributed Tracing | 📝 Future Work | `SLO_SPEC.md` | v1.3+ planned |
| Grafana Dashboards | 📝 Future Work | `SLO_SPEC.md` | v1.3+ planned |
| Error Budget Tracking | 📝 Future Work | `SLO_SPEC.md` | v1.3+ planned |

### Terminology Clarifications

| Term | Clarification | Location |
|------|---------------|----------|
| "Phase-Entangled" | Mathematical metaphor for phase-based associations, NOT quantum entanglement | `ARCHITECTURE_SPEC.md`, `docs/NEURO_FOUNDATIONS.md` |
| "Quantum-Inspired" | Follows Masuyama et al. (2014) notation - classical implementation inspired by quantum math | `docs/NEURO_FOUNDATIONS.md` |
| "Circadian 8+3" | Configurable wake(8)/sleep(3) steps - bio-inspired but not 24-hour cycle | `ARCHITECTURE_SPEC.md` |
| "Neurobiological" | Computational principles inspired by neuroscience, NOT neural simulation | `docs/SCIENTIFIC_RATIONALE.md` |

---

## Metric Sources

### How to Reproduce Key Metrics

```bash
# Moral Filter Effectiveness (93.3% toxic rejection)
pytest tests/validation/test_moral_filter_effectiveness.py -v -s

# Wake/Sleep Effectiveness (89.5% resource reduction)
pytest tests/validation/test_wake_sleep_effectiveness.py -v -s

# Performance Benchmarks (P50/P95/P99 latency)
pytest benchmarks/test_neuro_engine_performance.py -v -s

# Aphasia Detection (TPR/TNR/severity)
pytest tests/eval/test_aphasia_eval_suite.py -v

# Memory Bounds (29.37 MB)
pytest tests/property/test_invariants_memory.py -v

# Full Validation Suite
pytest tests/validation/ tests/eval/ -v
```

### Seed Values for Reproducibility

All validation tests use:
- **Random Seed**: 42
- **Vector Dimension**: 384
- **Sample Sizes**: 100-500 events per test

---

## Document Maintenance

This traceability matrix should be updated when:
1. New metrics are added to documentation
2. Test/benchmark implementations change
3. Future work items are implemented
4. Terminology is clarified

**Reviewer Workflow**: Before approving documentation changes, verify that any new numerical claims have corresponding entries in this matrix with valid test references.

---

**Document Status:** Production  
**Review Cycle:** Per PR with metric changes  
**Owner:** Principal System Architect
