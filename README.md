<div align="center">

# 🧠 MLSDM

### Multi-Level Synaptic Dynamic Memory

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/mlsdm-hero.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/mlsdm-hero.svg">
  <img src="assets/mlsdm-hero.svg" alt="MLSDM Neural Architecture diagram with core components" width="1200" height="600" style="max-width: 100%; height: auto; display: block; margin: 0 auto; image-rendering: crisp-edges;">
</picture>

**Beta-stage neurobiologically-inspired cognitive governance for LLMs**

*Phase-based memory • Adaptive moral filtering • Aphasia detection & repair*

---

[![CI](https://img.shields.io/github/actions/workflow/status/neuron7x/mlsdm/ci-neuro-cognitive-engine.yml?style=for-the-badge&logo=github-actions&logoColor=white&label=CI)](https://github.com/neuron7x/mlsdm/actions/workflows/ci-neuro-cognitive-engine.yml)
[![Tests](https://img.shields.io/github/actions/workflow/status/neuron7x/mlsdm/property-tests.yml?style=for-the-badge&logo=pytest&logoColor=white&label=Tests)](https://github.com/neuron7x/mlsdm/actions/workflows/property-tests.yml)
[![Security](https://img.shields.io/github/actions/workflow/status/neuron7x/mlsdm/sast-scan.yml?style=for-the-badge&logo=shield&logoColor=white&label=Security)](https://github.com/neuron7x/mlsdm/actions/workflows/sast-scan.yml)
[![Coverage](https://img.shields.io/badge/coverage-71%25%20(gate:%2065%25)-green?style=for-the-badge)](COVERAGE_REPORT_2025.md)
[![Python](https://img.shields.io/badge/python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ghcr.io-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://github.com/neuron7x/mlsdm/pkgs/container/mlsdm-neuro-engine)
[![Status](https://img.shields.io/badge/status-beta-orange?style=for-the-badge)](CHANGELOG.md)

[🚀 Getting Started](GETTING_STARTED.md) •
[Quick Start](#-quick-start) •
[Documentation](#-documentation) •
[Architecture](#-architecture) •
[Metrics](#-validated-metrics) •
[Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [What is MLSDM?](#-what-is-mlsdm)
- [Core Value Proposition](#-core-value-proposition)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Validated Metrics](#-validated-metrics)
- [Engineering & Production Readiness](#-engineering--production-readiness)
- [Documentation](#-documentation)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

> [!NOTE]
> **🆕 Latest Updates:** OpenTelemetry is now optional, reducing installation complexity. See [Getting Started](GETTING_STARTED.md) for the simplified setup.

## 🧬 What is MLSDM?

**MLSDM (Multi-Level Synaptic Dynamic Memory)** is a governed cognitive wrapper for Large Language Models that enforces biological constraints inspired by neuroscience.

<table>
<tr>
<td width="60%">

### The Problem

LLMs lack built-in mechanisms for:
- ❌ Memory bounded constraints
- ❌ Adaptive safety filtering without RLHF
- ❌ Cognitive rhythm management (wake/sleep cycles)
- ❌ Speech quality detection and repair

### The Solution

MLSDM wraps **any LLM** with a neurobiologically-grounded cognitive layer that provides:
- ✅ **Fixed memory footprint** (29.37 MB)
- ✅ **Adaptive moral filtering** (93.3% toxic rejection)
- ✅ **Wake/sleep cycles** (89.5% resource reduction)
- ✅ **Aphasia detection** (telegraphic speech repair)

</td>
<td width="40%">

```text
┌─────────────────────────┐
│      Your LLM           │
│  (OpenAI, Anthropic,    │
│   Local, Custom...)     │
└───────────┬─────────────┘
            │
    ┌───────▼───────┐
    │    MLSDM      │
    │   Wrapper     │
    │               │
    │ • Memory      │
    │ • Moral       │
    │ • Rhythm      │
    │ • Speech      │
    └───────┬───────┘
            │
    ┌───────▼───────┐
    │   Governed    │
    │   Response    │
    └───────────────┘
```

</td>
</tr>
</table>

---

## 💡 Core Value Proposition

| Feature | Description |
|:--------|:------------|
| 🔒 **Safety Without RLHF** | Adaptive moral filtering with EMA-based threshold adjustment. No expensive fine-tuning required. |
| 📊 **Bounded Resources** | Fixed 29.37 MB memory with zero-allocation after init. Perfect for production. |
| 🌙 **Cognitive Rhythm** | Wake/sleep cycles reduce resource usage by 89.5% during consolidation phases. |
| 🗣️ **Speech Quality** | Detects telegraphic patterns and triggers automatic repair for coherent output. |

---

## ✨ Key Features

### Cognitive Governance

| Feature | Description | Metric |
|:--------|:------------|:-------|
| **Moral Filter** | EMA-based adaptive threshold [0.30, 0.90] | 93.3% toxic rejection |
| **PELM Memory** | Phase-entangled lattice with 20k vector capacity | 29.37 MB fixed |
| **Wake/Sleep Cycles** | 8 wake + 3 sleep steps with memory consolidation | 89.5% resource savings |
| **Aphasia Detection** | Broca-model for telegraphic speech detection | 100% TPR, 80% TNR* |
| **Thread Safety** | Lock-based synchronization for concurrent requests | 1,000+ RPS verified |
| **Observability** | Prometheus metrics + structured JSON logging | Full pipeline visibility |

<details>
<summary><strong>View Detailed Feature Breakdown</strong></summary>

#### Multi-Level Synaptic Memory

```text
L1 (Short-term):  λ = 0.95  │ Fast decay, immediate context
L2 (Medium-term): λ = 0.98  │ Balanced retention, gated transfer
L3 (Long-term):   λ = 0.99  │ Slow decay, consolidated memories
```

#### Phase-Entangled Lattice Memory (PELM)

- **Capacity**: 20,000 vectors × 384 dimensions
- **Footprint**: 29.37 MB (pre-allocated, zero-growth)
- **Retrieval**: Cosine similarity with phase tolerance
- **Eviction**: Circular buffer (FIFO)

#### Moral Homeostasis Algorithm

```python
# EMA update (α = 0.1)
ema = α * signal + (1 - α) * ema_prev

# Threshold adaptation
error = ema - 0.5  # target equilibrium
if abs(error) > 0.05:  # dead-band
    threshold += 0.05 * sign(error)
    threshold = clip(threshold, 0.30, 0.90)
```

</details>

---

## 🏗️ Architecture

### System Overview

```mermaid
flowchart TB
    subgraph Client["Client Layer"]
        U[User Prompt]
        SDK[SDK Client]
        API[HTTP API]
    end

    subgraph Wrapper["MLSDM Wrapper"]
        LW[LLMWrapper]
        NLW[NeuroLangWrapper]
    end

    subgraph Controller["Cognitive Controller"]
        MF[Moral Filter V2]
        CR[Cognitive Rhythm]
        OM[Ontology Matcher]
    end

    subgraph Memory["Memory System"]
        PELM[Phase-Entangled<br/>Lattice Memory]
        MLM[Multi-Level<br/>Synaptic Memory]
    end

    subgraph Speech["Speech Governance"]
        ABD[Aphasia-Broca<br/>Detector]
        ASG[Aphasia Speech<br/>Governor]
    end

    subgraph LLM["LLM Provider"]
        OpenAI[OpenAI]
        Local[Local/Custom]
    end

    U --> SDK & API
    SDK & API --> LW & NLW
    LW & NLW --> Controller
    Controller --> Memory
    NLW --> Speech
    LW & NLW --> LLM
    LLM --> Response[Governed Response]

    style Wrapper fill:#e1f5fe,stroke:#01579b
    style Controller fill:#f3e5f5,stroke:#4a148c
    style Memory fill:#e8f5e9,stroke:#1b5e20
    style Speech fill:#fff3e0,stroke:#e65100
```

### Request Flow

```mermaid
sequenceDiagram
    participant U as User
    participant W as LLMWrapper
    participant MF as MoralFilter
    participant CR as CognitiveRhythm
    participant M as Memory
    participant LLM as LLM Provider
    participant A as AphasiaDetector

    U->>W: generate(prompt, moral_value)
    W->>W: Create embedding
    W->>MF: Evaluate moral_value

    alt Rejected
        MF-->>W: Rejected
        W-->>U: {accepted: false}
    else Accepted
        MF-->>W: Accepted
        W->>CR: Advance phase
        W->>M: Store & retrieve context
        W->>LLM: Generate with context
        LLM-->>W: Response

        opt NeuroLangWrapper
            W->>A: Analyze response
            alt Aphasic
                A-->>W: Repair needed
                W->>LLM: Regenerate
            end
        end

        W-->>U: {response, phase, metadata}
    end
```

### Invariants

| Invariant | Constraint | Enforcement |
|:----------|:-----------|:------------|
| Moral Threshold | [0.30, 0.90] | Bounded clipping in MoralFilterV2 |
| Memory Capacity | 20,000 vectors | Circular buffer eviction |
| Memory Footprint | ≤ 29.37 MB | Pre-allocated, zero-growth |
| Non-Aphasic Output | `avg_sentence_len ≥ 6` | AphasiaBrocaDetector |
| Function Words | `ratio ≥ 0.15` | Speech quality check |

For complete system design, see [ARCHITECTURE_SPEC.md](ARCHITECTURE_SPEC.md).

---

## 🚀 Quick Start

> **New to MLSDM?** Start with our [**Getting Started Guide**](GETTING_STARTED.md) for a streamlined introduction.

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/neuron7x/mlsdm.git
cd mlsdm

# Recommended: Install from source with pip
pip install -e .

# OR: Install with all dependencies (includes OpenTelemetry)
pip install -r requirements.txt

# Optional extras:
pip install -e ".[observability]"  # Add OpenTelemetry tracing
pip install -r requirements-neurolang.txt  # Add Aphasia/NeuroLang support
```

> **Note:** For detailed installation options and minimal dependencies, see [GETTING_STARTED.md](GETTING_STARTED.md).

**Note:** OpenTelemetry is now optional. MLSDM works perfectly without it if you don't need distributed tracing.

### Basic Usage

```python
from mlsdm.core.llm_wrapper import LLMWrapper
import numpy as np

# Define your LLM function
def my_llm(prompt: str, max_tokens: int) -> str:
    # Replace with your LLM (OpenAI, Anthropic, local, etc.)
    return "Your LLM response here"

# Define your embedding function
def my_embedder(text: str) -> np.ndarray:
    # Replace with your embedding model
    return np.random.randn(384).astype(np.float32)

# Create governed wrapper
wrapper = LLMWrapper(
    llm_generate_fn=my_llm,
    embedding_fn=my_embedder,
    dim=384,                        # Embedding dimension
    capacity=20_000,                # Memory capacity
    wake_duration=8,                # Wake phase steps
    sleep_duration=3,               # Sleep phase steps
    initial_moral_threshold=0.50    # Starting threshold
)

# Generate with governance
result = wrapper.generate(
    prompt="Explain quantum computing",
    moral_value=0.8
)

print(f"Response: {result['response']}")
print(f"Accepted: {result['accepted']}")
print(f"Phase: {result['phase']}")
print(f"Threshold: {result['moral_threshold']}")
```

### Run Tests

```bash
# Full test suite
pytest tests/ -v

# Effectiveness validation
pytest tests/validation/ -v

# Property-based tests
pytest tests/property/ -v

# Coverage gate (enforces minimum coverage threshold)
./coverage_gate.sh                 # Default threshold: 65%
COVERAGE_MIN=80 ./coverage_gate.sh # Custom threshold
```

### Policy Checks (Governance)

```bash
# Install conftest (OPA policy testing)
brew install conftest  # macOS
# or download from https://github.com/open-policy-agent/conftest

# Run policy checks on CI workflows
conftest test .github/workflows/*.yml -p policies/ci/
```

### Runtime Modes

MLSDM supports three runtime profiles for different deployment scenarios:

| Mode | Command | Use Case |
|:-----|:--------|:---------|
| **Development** | `make run-dev` | Local development with hot reload |
| **Cloud** | `make run-cloud-local` | Docker/k8s production deployment |
| **Agent/API** | `make run-agent` | External LLM/client integration |

```bash
# Development mode (hot reload, debug logging, no rate limit)
make run-dev

# Cloud production mode (multiple workers, secure mode)
make run-cloud-local

# Agent/API mode (for LLM platform integration)
make run-agent

# Health check
make health-check
```

Or run directly with Python:

```bash
# Development mode
python -m mlsdm.entrypoints.dev

# Cloud mode
python -m mlsdm.entrypoints.cloud

# Agent mode
python -m mlsdm.entrypoints.agent
```

See [env.dev.example](env.dev.example), [env.cloud.example](env.cloud.example), and [env.agent.example](env.agent.example) for configuration options.

---

## 📖 Usage Examples

<details>
<summary><strong>OpenAI Integration</strong></summary>

```python
from openai import OpenAI
import numpy as np
from mlsdm.core.llm_wrapper import LLMWrapper

# Initialize OpenAI client
client = OpenAI(api_key="your-api-key")

def openai_generate(prompt: str, max_tokens: int) -> str:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def openai_embed(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

# Create governed wrapper
wrapper = LLMWrapper(
    llm_generate_fn=openai_generate,
    embedding_fn=openai_embed,
    dim=1536  # Ada embedding dimension
)
```

</details>

<details>
<summary><strong>Local Model Integration</strong></summary>

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from mlsdm.core.llm_wrapper import LLMWrapper
import numpy as np

# Load local models
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def local_generate(prompt: str, max_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def local_embed(text: str) -> np.ndarray:
    return embedder.encode(text).astype(np.float32)

wrapper = LLMWrapper(
    llm_generate_fn=local_generate,
    embedding_fn=local_embed,
    dim=384
)
```

</details>

<details>
<summary><strong>Aphasia Detection & Repair</strong></summary>

```python
from mlsdm.extensions.neuro_lang_extension import NeuroLangWrapper

# Use NeuroLangWrapper for aphasia detection
wrapper = NeuroLangWrapper(
    llm_generate_fn=my_llm,
    embedding_fn=my_embedder,
    dim=384
)

result = wrapper.generate(
    prompt="Describe the scientific method",
    moral_value=0.9
)

# Check aphasia analysis
if result.get("aphasia_flags"):
    print(f"Aphasia detected: {result['aphasia_flags']}")
    print("Original response was repaired")
```

</details>

<details>
<summary><strong>FastAPI Service</strong></summary>

```python
from fastapi import FastAPI
from pydantic import BaseModel
from mlsdm.core.llm_wrapper import LLMWrapper

app = FastAPI()
wrapper = LLMWrapper(...)

class GenerateRequest(BaseModel):
    prompt: str
    moral_value: float = 0.8

@app.post("/generate")
async def generate(request: GenerateRequest):
    return wrapper.generate(
        prompt=request.prompt,
        moral_value=request.moral_value
    )

@app.get("/health")
async def health():
    state = wrapper.get_state()
    return {"status": "ok", "phase": state["phase"]}
```

</details>

---

## 📊 Validated Metrics

All metrics are backed by reproducible tests with full traceability.

### Safety & Governance

| Metric | Value | Test Location |
|:-------|:------|:--------------|
| Toxic Rejection Rate | 93.3% | `tests/validation/test_moral_filter_effectiveness.py` |
| Comprehensive Safety | 97.8% | `tests/validation/test_moral_filter_effectiveness.py` |
| False Positive Rate | 37.5% | Trade-off for safety |
| Drift Under Attack | 0.33 max | 70% toxic bombardment scenario |

### Performance

| Metric | Value | Test Location |
|:-------|:------|:--------------|
| Throughput | 1,000+ RPS* | `tests/load/` (requires server) |
| P50 Latency | ~2ms | `benchmarks/` |
| P95 Latency | ~10ms | `benchmarks/` |
| Memory | 29.37 MB fixed | `tests/unit/` |

### Cognitive Effectiveness

| Metric | Value | Test Location |
|:-------|:------|:--------------|
| Resource Reduction | 89.5% | `tests/validation/test_wake_sleep_effectiveness.py` |
| Coherence Improvement | 5.5% | `tests/validation/test_wake_sleep_effectiveness.py` |
| Aphasia TPR | 100%* | `tests/eval/aphasia_eval_suite.py` |
| Aphasia TNR | 80%* | `tests/eval/aphasia_eval_suite.py` |

**\*Aphasia Note**: Metrics measured on evaluation corpus of 100 samples (50 telegraphic + 50 normal). See `tests/eval/aphasia_corpus.json`.

**\*\*Performance Note**: Throughput tested with Locust load tests. The 5,500 ops/sec estimate from earlier documentation requires server deployment and is marked as "Partial" in [CLAIMS_TRACEABILITY.md](CLAIMS_TRACEABILITY.md). The 1,000+ RPS figure represents the verified SLO target.

For detailed validation results, see:
- [EFFECTIVENESS_VALIDATION_REPORT.md](EFFECTIVENESS_VALIDATION_REPORT.md)
- [CLAIMS_TRACEABILITY.md](CLAIMS_TRACEABILITY.md)

---

## ⚙️ Engineering & Production Readiness

> [!NOTE]
> MLSDM is designed as an infrastructure component with comprehensive testing, observability, security controls, and production deployment patterns.

### 🧪 Quality & Reliability Matrix

| Dimension | Status | Implementation | Key References |
|:----------|:-------|:---------------|:---------------|
| **Test Coverage** | 70.85%* | `pytest`, `pytest-cov`, unit/integration/e2e/property | [TESTING_GUIDE.md](TESTING_GUIDE.md), [COVERAGE_REPORT_2025.md](COVERAGE_REPORT_2025.md) |
| **Test Types** | Unit, Integration, E2E, Property, Load, Security | `tests/unit/`, `tests/integration/`, `tests/e2e/`, `tests/property/`, `tests/load/`, `tests/security/` | [tests/](tests/) |
| **Type Safety** | Strict mypy | Configured in `pyproject.toml` with strict mode | [pyproject.toml](pyproject.toml) |
| **Static Analysis** | ruff, bandit | Pre-commit hooks and CI checks | [.pre-commit-config.yaml](.pre-commit-config.yaml) |
| **CI/CD** | GitHub Actions | Multi-workflow pipeline (CI, property tests, release) | [.github/workflows/](.github/workflows/) |
| **Security** | Policy + Implementation | Rate limiting, input validation, audit logging, threat model | [SECURITY_POLICY.md](SECURITY_POLICY.md), [THREAT_MODEL.md](THREAT_MODEL.md) |
| **Observability** | Prometheus + OpenTelemetry | Metrics, structured logging, distributed tracing | [OBSERVABILITY_GUIDE.md](OBSERVABILITY_GUIDE.md), [SLO_SPEC.md](SLO_SPEC.md) |

**\*Coverage Note**: Full codebase coverage is 70.85% (measured on `tests/unit/` + `tests/state/`). Core cognitive modules achieve 90%+ coverage. See [COVERAGE_REPORT_2025.md](COVERAGE_REPORT_2025.md) for detailed breakdown.

### 🚀 Deployment Topologies

MLSDM supports multiple deployment patterns:

| Topology | Description | Key Files |
|:---------|:------------|:----------|
| **Local/Dev** | Single container or bare metal | [`docker/Dockerfile`](docker/Dockerfile), [`docker/docker-compose.yaml`](docker/docker-compose.yaml) |
| **Service Image** | Production-ready container | [`Dockerfile.neuro-engine-service`](Dockerfile.neuro-engine-service) |
| **Kubernetes** | Full k8s manifests with monitoring | [`deploy/k8s/`](deploy/k8s/) |
| **Production** | Hardened deployment with security contexts | [`deploy/k8s/production-deployment.yaml`](deploy/k8s/production-deployment.yaml) |

**Kubernetes artifacts:**
- [`deploy/k8s/deployment.yaml`](deploy/k8s/deployment.yaml) — Base deployment configuration
- [`deploy/k8s/service.yaml`](deploy/k8s/service.yaml) — Service definition
- [`deploy/k8s/configmap.yaml`](deploy/k8s/configmap.yaml) — Configuration
- [`deploy/k8s/secrets.yaml`](deploy/k8s/secrets.yaml) — Secrets template
- [`deploy/k8s/service-monitor.yaml`](deploy/k8s/service-monitor.yaml) — Prometheus ServiceMonitor
- [`deploy/k8s/network-policy.yaml`](deploy/k8s/network-policy.yaml) — Network policies

### 📈 Observability & Ops

| Category | Description | Reference |
|:---------|:------------|:----------|
| **Metrics** | Prometheus-compatible metrics at `/health/metrics` | [OBSERVABILITY_GUIDE.md](OBSERVABILITY_GUIDE.md) |
| **Logging** | Structured JSON logs with correlation IDs, PII scrubbing | [OBSERVABILITY_GUIDE.md](OBSERVABILITY_GUIDE.md) |
| **Tracing** | OpenTelemetry distributed tracing (optional) | [OBSERVABILITY_GUIDE.md](OBSERVABILITY_GUIDE.md) |
| **Dashboards** | Grafana JSON dashboard | [`deploy/grafana/mlsdm_observability_dashboard.json`](deploy/grafana/mlsdm_observability_dashboard.json) |
| **Alerting** | Alertmanager rules | [`deploy/monitoring/alertmanager-rules.yaml`](deploy/monitoring/alertmanager-rules.yaml) |
| **SLOs** | Availability ≥99.9%, P95 latency <120ms, memory ≤50MB | [SLO_SPEC.md](SLO_SPEC.md) |
| **Runbook** | Operational procedures for incidents | [RUNBOOK.md](RUNBOOK.md) |

### 🛡️ Safety-by-Design

MLSDM implements defense-in-depth security:

| Control | Implementation | Reference |
|:--------|:---------------|:----------|
| **Rate Limiting** | 5 RPS per client (leaky bucket) | [SECURITY_IMPLEMENTATION.md](SECURITY_IMPLEMENTATION.md) |
| **Input Validation** | Type, range, dimension, sanitization | [SECURITY_POLICY.md](SECURITY_POLICY.md) |
| **Authentication** | Bearer token (OAuth2 scheme) | [SECURITY_POLICY.md](SECURITY_POLICY.md) |
| **Memory Bounds** | Fixed 29.37 MB, zero-growth | [ARCHITECTURE_SPEC.md](ARCHITECTURE_SPEC.md) |
| **Threat Model** | STRIDE analysis | [THREAT_MODEL.md](THREAT_MODEL.md) |
| **Risk Register** | AI safety risks tracked | [RISK_REGISTER.md](RISK_REGISTER.md) |
| **Secure Mode** | `MLSDM_SECURE_MODE=1` disables training in production | [SECURITY_POLICY.md](SECURITY_POLICY.md) |

For detailed deployment instructions, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md).

---

## 📖 Documentation

### Core Documentation

| Document | Description |
|:---------|:------------|
| [Architecture Spec](ARCHITECTURE_SPEC.md) | Full system design and component interactions |
| [Usage Guide](USAGE_GUIDE.md) | Detailed usage patterns and best practices |
| [Configuration Guide](CONFIGURATION_GUIDE.md) | All configuration options explained |
| [API Reference](API_REFERENCE.md) | Complete API documentation |
| [Deployment Guide](DEPLOYMENT_GUIDE.md) | Production deployment instructions |

### Validation & Testing

| Document | Description |
|:---------|:------------|
| [Implementation Summary](IMPLEMENTATION_SUMMARY.md) | What was built and how |
| [Effectiveness Report](EFFECTIVENESS_VALIDATION_REPORT.md) | Quantitative validation results |
| [Coverage Report](COVERAGE_REPORT_2025.md) | 90.26% test coverage details |
| [Testing Guide](TESTING_GUIDE.md) | How to run and write tests |

### Scientific Foundation

| Document | Description |
|:---------|:------------|
| [Scientific Rationale](docs/SCIENTIFIC_RATIONALE.md) | Core hypothesis and theory |
| [Neuro Foundations](docs/NEURO_FOUNDATIONS.md) | Neuroscience basis for each module |
| [Safety Foundations](docs/ALIGNMENT_AND_SAFETY_FOUNDATIONS.md) | AI safety principles |
| [Bibliography](BIBLIOGRAPHY.md) | Peer-reviewed references |

### Operations & Support

| Document | Description |
|:---------|:------------|
| [Getting Started](GETTING_STARTED.md) | **5-minute quickstart guide for new users** |
| [Troubleshooting](TROUBLESHOOTING.md) | **Common issues and solutions** |
| [CI Guide](CI_GUIDE.md) | **CI/CD configuration and workflows** |
| [Observability Guide](OBSERVABILITY_GUIDE.md) | Metrics, logging, tracing setup |
| [Runbook](RUNBOOK.md) | Operational procedures |
| [Security Policy](SECURITY_POLICY.md) | Security guidelines |

---

## 🗺️ Roadmap

### Stable (v1.x) — Current

- [x] Universal LLM wrapper with moral governance
- [x] Phase-entangled memory (PELM, 20k capacity)
- [x] Wake/sleep cognitive rhythm
- [x] Aphasia-Broca detection and repair
- [x] Prometheus metrics and structured logging
- [x] 90%+ test coverage with property-based tests
- [x] Thread-safe concurrent access

### Recent Improvements (v1.2+)

- [x] **OpenTelemetry is now optional** - Core system works without tracing dependencies
- [x] **Reduced entry barrier** - New Getting Started guide with 5-minute quickstart
- [x] **Improved documentation** - Added Troubleshooting and CI guides
- [x] **Graceful degradation** - All observability features work without OTEL

### In Progress

- [ ] Enhanced Grafana dashboards
- [ ] Additional usage examples

### Future Work

| Feature | Requirement |
|:--------|:------------|
| Stress testing at 10k+ RPS | Load infrastructure |
| Chaos engineering suite | Staging environment |
| TLA+/Coq formal verification | Formal methods expertise |
| RAG hallucination testing | Retrieval setup with ragas |

### Known Limitations

> [!WARNING]
> Understand these constraints before deploying to production.

| Limitation | Details |
|:-----------|:--------|
| No hallucination prevention | Wraps LLM but cannot improve factual accuracy |
| Imperfect filtering | 93.3% toxic rejection (6.7% may pass); 37.5% false positive rate |
| Beta status | Additional hardening needed for mission-critical production |
| Not a compliance substitute | Requires domain-specific security audit |

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for:

- Development setup
- Coding standards
- Pull request process
- Testing requirements

### Quick Contribution Commands

```bash
# Setup development environment
git clone https://github.com/neuron7x/mlsdm.git
cd mlsdm
pip install -r requirements.txt

# Run tests before submitting
pytest tests/ -v

# Check linting
ruff check src/
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with 🧠 for the future of AI safety**

[↑ Back to Top](#-mlsdm)

</div>
