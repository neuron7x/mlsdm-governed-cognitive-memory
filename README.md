# mlsdm-governed-cognitive-memory

Governed cognitive memory system (ML-SDM) with multi-level synaptic memory, moral filtering,
ontology matching, quantum-inspired learning and cognitive rhythm.

## Installation

```bash
pip install -r requirements.txt
```

## Run simulation

```bash
python -m src.main --steps 100 --plot
```

## Run API

```bash
python -m src.main --api
```

## Tests (Quick Start)

```bash
make test            # run unit + property tests
pytest -k property   # run property-based invariants (Hypothesis)
pytest -k state      # run state machine transition tests
```

## Testing & Verification Strategy (Principal System Architect Level)
This project incorporates advanced system reliability, mathematical correctness, AI safety, and performance validation methodologies expected at Principal / Staff engineering levels.

### Verification Pillars
1. Invariant Verification (Formal & Property-Based)
2. Resilience & Chaos Robustness
3. AI Governance & Safety Hardening
4. Performance & Saturation Profiling
5. Drift & Alignment Stability
6. Observability of Tail Failure Modes

### Methodologies Implemented / Planned
| Category | Method | Purpose | Tooling |
|----------|--------|---------|---------|
| Resilience | Chaos Engineering / Fault Injection | Validate graceful degradation under component failure (DB/network/pod kill) | chaos-toolkit, custom fault scripts |
| Resilience | Soak Testing (48–72h) | Detect memory leaks, latent resource exhaustion | Locust / custom harness + Prometheus export |
| Resilience | Load Shedding & Backpressure Testing | Ensure overload results in fast rejection, not collapse | Rate limit middleware + stress generators |
| Correctness | Property-Based Testing | Assert mathematical invariants across wide input space | Hypothesis |
| Correctness | State Machine Verification | Enforce legal cognitive rhythm transitions (Sleep→Wake→Processing) | pytest state model + TLA+ spec alignment |
| AI Safety | Adversarial Red Teaming | Jailbreak & prompt-injection resistance for MoralFilter | Attack LLM harness + curated attack corpus |
| AI Safety | Cognitive Drift Testing | Ensure moral thresholds remain stable under toxic sequence bombardment | Drift probes + statistical monitoring |
| AI Safety | RAG Hallucination / Faithfulness Testing | Quantify grounding vs fabrication | ragas + retrieval audit logs |
| Performance | Saturation Testing | Identify RPS inflection where latency spikes | Locust/K6 + SLO dashboards |
| Performance | Tail Latency (P99/P99.9) Audits | Guarantee upper-bound latency SLOs | OpenTelemetry + Prometheus histograms |
| Formal | Formal Specification (TLA+) | Prove liveness/safety of memory lifecycle | TLC model checker |
| Formal | Algorithm Proof Fragments (Coq) | Prove correctness of address selection / neighbor threshold | Coq scripts |
| Governance | Ethical Override Traceability | Ensure explainable policy decisions | Structured event logging |
| Reliability | Drift & Anomaly Injection | Validate detection pipeline reaction | Synthetic anomaly injectors |

### Core Invariants (Examples)
- Moral filter threshold T always ∈ [0.1, 0.9].
- Episodic graph remains acyclic (no circular temporal references).
- State cannot jump directly from Sleep → Processing without Wake.
- Retrieval under corruption degrades to stateless fallback but always returns a syntactically valid response envelope.

### Sample Property-Based Invariant (Hypothesis Sketch)
```python
from hypothesis import given, strategies as st
from src.moral import clamp_threshold

@given(t=st.floats(min_value=-10, max_value=10))
def test_moral_threshold_clamped(t):
    clamped = clamp_threshold(t)
    assert 0.1 <= clamped <= 0.9
```

### Chaos Scenarios (Initial Set)
1. Kill vector DB container mid high-RPS retrieval.
2. Introduce 3000ms network latency between memory and policy service.
3. Randomly corrupt 0.5% of episodic entries → verify integrity alarms trigger & quarantine.
4. Simulated clock skew in circadian scheduler.

### Performance SLO Focus
- P95 composite memory retrieval < 120ms.
- P99 policy decision < 60ms.
- Error budget: ≤ 2% degraded cycles per 24h.

### AI Safety Metrics
- Drift Δ(moral_threshold) over toxic storm < 0.05 absolute.
- Hallucination rate (ragas) < 0.15.
- Successful jailbreak attempts < 0.5% of adversarial batch.

### Observability Hooks
- event_formal_violation, event_drift_alert, event_chaos_fault.
- Prometheus histograms: retrieval_latency_bucket, moral_filter_eval_ms.
- OpenTelemetry trace: MemoryRetrieve → SemanticMerge → PolicyCheck.

### Toolchain
- Hypothesis, pytest, chaos-toolkit, Locust/K6, ragas, TLA+, Coq, OpenTelemetry, Prometheus.

## Contributing
PRs & issues welcome. Add tests (property/state/chaos) for new logic.

## License
TBD