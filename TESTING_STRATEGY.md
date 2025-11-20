# TESTING_STRATEGY

Principal-level system & AI verification approach for the governed ML-SDM cognitive memory framework.

---
## 1. Philosophy
We do not only test whether the code works; we test how the system degrades, how it can lie, and whether its behavior obeys declared mathematical invariants. Reliability, safety, and formal correctness are treated as first-class features.

---
## 2. Pillars
1. Invariant Verification (Property-Based + Formal Specs)
2. Resilience & Chaos Robustness
3. AI Governance & Safety Hardening
4. Performance & Saturation Profiling
5. Drift & Alignment Stability
6. Tail Failure Mode Observability

---
## 3. Invariant & Property-Based Testing
Tool: Hypothesis
Focus:
- Moral threshold clamp: T ∈ [0.1, 0.9]
- Episodic graph acyclicity
- Address selection monotonicity: similarity(addr(query), addr(neighbor)) ≥ configured_min
- State machine legal transitions: Sleep → Wake → Processing → (Consolidation|Idle)
Approach:
- Hypothesis strategies generate random high-dimensional vectors, toxic score distributions, and temporal event sequences.
- Shrinking used to derive minimal counterexamples → stored under /test/property/counterexamples/.

Example:
```python
@given(vec=vector_strategy(), toxic=st.floats(0,1))
def test_threshold_stability(vec, toxic):
    t = compute_threshold(vec, toxic)
    assert 0.1 <= t <= 0.9
```

---
## 4. Formal Specification
Tools: TLA+ (system lifecycle), Coq (critical algorithms)
Targets:
- Liveness: every authorized request eventually receives a policy decision.
- Safety: consolidation never deletes critical flagged memory nodes.
- Acyclic episodic timeline.
Integration:
- /spec/tla: run `make tlc` in CI (bounded model check).
- /spec/coq: proofs for neighbor threshold lemma; CI verifies `coqc` passes.
- Generated runtime assertions mirror TLA invariants.

---
## 5. Resilience & Chaos Engineering
Scenarios:
1. Fault Injection: kill vector DB (or simulate 5s latency).
2. Network Partition: drop 20% packets between memory and policy components.
3. Disk Pressure: fill 90% storage; verify graceful read-only mode.
4. Clock Skew: offset scheduler time by +7m.
Graceful Degradation Goals:
- Retrieval fallback returns structured envelope with degraded flag.
- Policy decisions still deterministic under partial state.
Tooling: chaos-toolkit scripts in /scripts/chaos/, Kubernetes pod disruption budgets.
Metrics: chaos_recovery_seconds, degraded_response_ratio.

---
## 6. Soak & Endurance
48–72h sustained RPS to expose leaks.
- Monitor: RSS memory growth < 5% after steady state.
- GC cycle times stable.
Tools: Locust/K6 scenario, Prometheus retention.

---
## 7. Load Shedding & Backpressure
Overload Simulation: send 10,000 RPS when limit=100.
Expectations:
- Immediate rejection of excess with HTTP 429 / gRPC RESOURCE_EXHAUSTED.
- Queue depth metric capped.
- No latency inflation for accepted requests.
Metrics: rejected_rps, accepted_latency_p95.

---
## 8. Performance & Saturation
Goals:
- Identify inflection where P95 retrieval jumps (capacity planning baseline).
- Track P99 and P99.9 latencies for memory + policy evaluation.
Tools: OpenTelemetry traces, Prometheus histograms.
SLIs:
- retrieval_latency_ms
- policy_eval_ms
- consolidation_duration_ms

---
## 9. Tail Latency Audits
Weekly job computes quantile drift.
Alert if P99 > (SLO_P99 * 1.15) for 3 consecutive windows.
Remediation: analyze trace exemplars; perform index compaction or cache warm.

---
## 10. AI Safety & Governance
### Adversarial Red Teaming
Automated prompts (jailbreak corpus) vs MoralFilter.
Metric: jailbreak_success_rate < 0.5%.
### Cognitive Drift Testing
Inject 10k toxic queries; measure Δ(moral_threshold) < 0.05.
### RAG Hallucination / Faithfulness
Tool: ragas — track hallucination_rate < 0.15.
### Ethical Override Traceability
Every override emits event_policy_override with justification.

---
## 11. Drift & Alignment Monitoring
Vectors: track embedding centroid shifts; anomaly if cosine distance from baseline > 0.1.
Periodic recalibration during circadian Consolidation phase.

---
## 12. Observability
Events:
- event_formal_violation
- event_drift_alert
- event_chaos_fault
Metrics:
- moral_filter_eval_ms
- drift_vector_magnitude
- ethical_block_rate
Traces: MemoryRetrieve → SemanticMerge → PolicyCheck.

---
## 13. Toolchain Summary
| Purpose | Tool |
|---------|------|
| Property Testing | Hypothesis |
| Formal Specs | TLA+, Coq |
| Chaos | chaos-toolkit |
| Load / Soak | Locust, K6 |
| Safety (RAG) | ragas |
| Tracing | OpenTelemetry |
| Metrics | Prometheus |
| Proof CI | GitHub Actions |

---
## 14. CI Integration

### GitHub Actions Workflows (Implemented)

#### 1. PR Validation Workflow (`pr-validation.yml`)
Runs on every pull request to main/develop branches:
- **Lint & Type Check**: Ruff linter + MyPy strict type checking
- **Unit Tests**: Full unit test suite with 90% coverage requirement
- **Integration Tests**: End-to-end and validation tests
- **Property-Based Tests**: Hypothesis-driven invariant verification
- **Security Scan**: Bandit, Safety, pip-audit vulnerability scanning
- **Dependency Check**: Validate dependencies and check for conflicts
- **Test Matrix**: Python 3.10, 3.11, 3.12 compatibility

**Quality Gates**: All checks must pass before merge. Coverage threshold: 90%.

#### 2. Continuous Integration Workflow (`ci.yml`)
Runs on push to main/develop and nightly:
- **Full Test Suite**: Complete test coverage including unit, integration, validation
- **Chaos Engineering (Nightly)**: Fault injection and resilience tests
- **Performance Baseline**: Wake/sleep effectiveness and moral filter validation
- **Memory Leak Detection (Nightly)**: Extended memory profiling
- **Build Validation**: Package building and validation
- **Docker Build Test**: Container image verification

**Artifacts**: Coverage reports, performance metrics, build artifacts

#### 3. CodeQL Security Analysis (`codeql.yml`)
Runs on push, PR, and weekly schedule:
- **Static Analysis**: Security-extended queries
- **Vulnerability Detection**: CVE scanning and SARIF reporting
- **Code Quality**: Security and quality combined analysis

**Upload**: Results to GitHub Security tab

#### 4. Performance Testing Workflow (`performance-tests.yml`)
Runs weekly and on-demand:
- **Load Testing**: Locust-based RPS testing with configurable duration
- **Stress Testing**: 50 concurrent workers, memory monitoring
- **Latency Profiling**: P50, P95, P99, P99.9 measurements
- **Memory Profiling**: Fixed memory bounds validation (≤1.4GB)

**SLOs Validated**:
- P95 latency < 120ms
- P99 latency < 200ms
- Throughput > 1000 ops/sec
- Memory increase < 100MB during 10k events

#### 5. Dependency Scanning Workflow (`dependency-scan.yml`)
Runs daily and on dependency changes:
- **Security Audit**: pip-audit for known vulnerabilities
- **Safety Check**: Safety database scanning
- **License Check**: pip-licenses for compliance
- **Vulnerability Scan**: Bandit SARIF reports
- **Outdated Check**: Track package updates
- **Dependency Graph**: pipdeptree visualization

**Alerts**: Automated reports on security issues

### Test Categories

#### Unit Tests (182 tests, 90.48% coverage)
Location: `src/tests/unit/`
- Component tests for all core modules
- Property-based tests with Hypothesis
- State machine transition verification
- Thread safety and concurrency tests

#### Integration Tests
Location: `tests/integration/`
- End-to-end cognitive processing flow
- Multi-component interaction validation

#### Validation Tests
Location: `tests/validation/`
- Wake/sleep effectiveness (89.5% efficiency improvement)
- Moral filter effectiveness (93.3% toxic rejection)
- Coherence improvement (5.5%)

#### Chaos Engineering Tests
Location: `tests/chaos/`
- High concurrency race conditions (5000 events, 50 workers)
- Invalid input handling (NaN, Inf, zero vectors)
- Extreme moral value boundaries
- Rapid phase transitions
- Toxic bombardment resilience
- Concurrent phase transitions
- Memory stability under sustained load

#### Adversarial Tests
Location: `tests/adversarial/`
- Threshold manipulation resistance
- Gradient-based attacks
- High-frequency toggle attacks
- Boundary probing
- Sustained toxic siege (500 events)
- Mixed attack patterns
- EMA stability validation

#### Performance Benchmarks
Location: `tests/performance/`
- P95/P99 latency SLO validation
- Throughput baseline (single and concurrent)
- Memory footprint verification
- Latency stability over time

### Running Tests Locally

```bash
# Full test suite
pytest src/tests/ tests/ -v --cov=src --cov-report=html

# Unit tests only
pytest src/tests/unit/ -v

# Integration tests
python tests/integration/test_end_to_end.py

# Validation tests
python tests/validation/test_moral_filter_effectiveness.py
python tests/validation/test_wake_sleep_effectiveness.py

# Chaos engineering
python tests/chaos/test_fault_injection.py

# Adversarial tests
python tests/adversarial/test_jailbreak_resistance.py

# Performance benchmarks
python tests/performance/test_benchmarks.py

# Property-based tests
pytest src/tests/unit/test_property_based.py --hypothesis-show-statistics
```

### Pre-commit Hooks

Install pre-commit hooks for automatic validation:
```bash
pip install pre-commit
pre-commit install
```

Hooks include:
- Trailing whitespace removal
- YAML/JSON/TOML validation
- Ruff linting and formatting
- MyPy type checking
- Bandit security scanning
- Pytest unit tests

### CI/CD Workflow Stages Summary
1. **PR Validation**: Comprehensive quality gates before merge
2. **CI Build**: Full suite on main branch + nightly extended tests
3. **Security Scan**: Weekly CodeQL + daily dependency scanning
4. **Performance**: Weekly load/stress/latency profiling
5. **Deployment Gate**: All checks must pass for production release

Failure in security scans or performance SLO violations gates deployment.

---
## 15. Exit Criteria for "Production-Ready"
- All core invariants hold (no Hypothesis counterexamples for 10k runs each).
- Chaos suite passes with ≤ 5% degraded responses & zero uncaught panics.
- Tail latency P99 within SLO for 7 consecutive days.
- Jailbreak success rate below threshold for 3 consecutive weekly runs.
- No formal invariant violations in last 30 CI cycles.

---
## 16. Future Extensions
- Symbolic execution for critical moral logic paths.
- Stateful fuzzing of consolidation algorithm.
- Multi-agent interaction fairness audits.

---
## 17. Glossary (Key Terms for Resume / Docs)
- Invariant Verification
- Chaos Engineering
- Adversarial Red Teaming
- Cognitive Drift Testing
- Load Shedding / Backpressure
- Saturation & Tail Latency Analysis
- Formal Specification (TLA+, Coq)
- RAG Hallucination/Faithfulness Assessment

---
Maintainer: neuron7x
