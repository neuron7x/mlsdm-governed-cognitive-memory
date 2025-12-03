# Production Gaps

**Version**: 1.2.0  
**Last Updated**: December 2025  
**Purpose**: Prioritized task list for production-readiness improvements

---

## Summary

| Block | Blockers | High | Medium | Low |
|-------|----------|------|--------|-----|
| Core Reliability | 0 | 0 | 0 | 0 |
| Observability | 0 | 2 | 3 | 1 |
| Security | 0 | 2 | 2 | 2 |
| Performance | 0 | 1 | 2 | 1 |
| CI/CD | 0 | 4 | 2 | 1 |
| Docs | 0 | 1 | 2 | 2 |
| **Total** | **0** | **10** | **11** | **7** |

---

## Core Reliability ✅ COMPLETE

All Core Reliability tasks have been completed with verified implementations and tests:

- **REL-001**: Automated health-based recovery (time-based and step-based auto-recovery)
  - Code: `src/mlsdm/core/cognitive_controller.py::CognitiveController._try_auto_recovery`, `_enter_emergency_shutdown`
  - Config: `config/calibration.py::CognitiveControllerCalibration` (auto_recovery_enabled, auto_recovery_cooldown_seconds)
  - Tests: `tests/unit/test_cognitive_controller.py::TestCognitiveControllerAutoRecovery`, `TestCognitiveControllerTimeBasedRecovery`
  - Logs: `auto-recovery succeeded`, `Emergency shutdown entered`

- **REL-002**: Bulkhead pattern for request isolation (BulkheadMiddleware with metrics)
  - Code: `src/mlsdm/api/middleware.py::BulkheadMiddleware`, `BulkheadSemaphore`
  - Metrics: `mlsdm_bulkhead_queue_depth`, `mlsdm_bulkhead_active_requests`, `mlsdm_bulkhead_rejected_total`
  - Tests: `tests/api/test_middleware_reliability.py::TestBulkheadMetricsIntegration`, `tests/resilience/test_bulkhead_integration.py`

- **REL-003**: Chaos engineering tests in CI (memory pressure, slow LLM, network timeout)
  - Tests: `tests/chaos/test_memory_pressure.py`, `tests/chaos/test_slow_llm.py`, `tests/chaos/test_network_timeout.py`
  - CI: `.github/workflows/chaos-tests.yml` (scheduled daily at 03:00 UTC)
  - Marker: `@pytest.mark.chaos`

- **REL-004**: Request timeout middleware (TimeoutMiddleware with 504 responses)
  - Code: `src/mlsdm/api/middleware.py::TimeoutMiddleware`
  - Config: `config/default_config.yaml::api.request_timeout_seconds` (default: 30s)
  - Tests: `tests/api/test_middleware_reliability.py::TestTimeoutMiddleware`
  - Response: HTTP 504 with `X-Request-Timeout` header

- **REL-005**: Request prioritization (PriorityMiddleware with X-MLSDM-Priority header)
  - Code: `src/mlsdm/api/middleware.py::PriorityMiddleware`, `RequestPriority`
  - Header: `X-MLSDM-Priority: high|normal|low` (or numeric 1-10)
  - Tests: `tests/api/test_middleware_reliability.py::TestRequestPriority`, `TestPriorityMiddleware`
  - Docs: `API_REFERENCE.md` (X-MLSDM-Priority section), `USAGE_GUIDE.md` (Request Priority section)

### Implementation Verification

Run these commands to verify Core Reliability implementation:

```bash
# REL-001: Auto-recovery tests (15 tests)
pytest tests/unit/test_cognitive_controller.py -k "AutoRecovery or TimeBasedRecovery" -v

# REL-002: Bulkhead tests (10 tests)
pytest tests/resilience/test_bulkhead_integration.py -v

# REL-003: Chaos engineering tests (17 tests, ~3 min)
pytest tests/chaos/ -m chaos -v

# REL-004: Timeout middleware tests
pytest tests/api/test_middleware_reliability.py::TestTimeoutMiddleware -v

# REL-005: Priority middleware tests
pytest tests/api/test_middleware_reliability.py::TestRequestPriority -v
pytest tests/api/test_middleware_reliability.py::TestPriorityMiddleware -v

# All Core Reliability tests
pytest tests/unit/test_cognitive_controller.py tests/api/test_middleware_reliability.py tests/resilience/test_bulkhead_integration.py -v
```

Example curl for priority testing:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "X-MLSDM-Priority: high" \
  -d '{"prompt": "Test high priority request"}'
```

---

## BLOCKER (Must fix before production)

_All blockers resolved._

---

## HIGH Priority

### ~~CICD-001: Add linting and type checking to CI workflows~~ ✅ COMPLETED

**Block**: CI/CD  
**Criticality**: ~~BLOCKER~~ COMPLETED  
**Type**: CI

**Description**: ~~Currently `ruff check` and `mypy` are only available via `make lint` and `make type` locally. CI workflows run tests but do not enforce linting or type safety, allowing regressions to reach main.~~ Added linting and type checking steps to CI workflow.

**Acceptance Criteria**:
- ✅ Add `ruff check src tests` step to `ci-neuro-cognitive-engine.yml`
- ✅ Add `mypy src/mlsdm` step to `ci-neuro-cognitive-engine.yml`
- ✅ Both steps must pass for PR to be mergeable

**Affected Files**:
- `.github/workflows/ci-neuro-cognitive-engine.yml`

---

## HIGH Priority

### ~~REL-001: Implement automated health-based recovery~~ ✅ COMPLETED

**Block**: Core Reliability  
**Criticality**: ~~HIGH~~ COMPLETED  
**Type**: Code

**Description**: ~~After `emergency_shutdown` is triggered due to memory threshold, manual intervention is required via `reset_emergency_shutdown()`. Production deployments need automated recovery.~~

**Solution**: Implemented dual-mode auto-recovery in `CognitiveController`:
- **Time-based recovery**: Attempts recovery after `auto_recovery_cooldown_seconds` (default: 60s)
- **Step-based recovery**: Attempts recovery after `recovery_cooldown_steps` (default: 10 steps)
- **Safety guards**: Memory must be below `recovery_memory_safety_ratio` (80%) of threshold
- **Max attempts**: Stops auto-recovery after `recovery_max_attempts` (default: 3) to prevent infinite loops

**Acceptance Criteria**:
- ✅ Add optional auto-recovery after configurable cooldown period
- ✅ Log recovery events
- ✅ Add tests for recovery behavior

**Implementation**:
- `CognitiveController._try_auto_recovery()`: Core recovery logic
- `CognitiveController._enter_emergency_shutdown()`: Records time and step for cooldown tracking
- Parameters: `auto_recovery_enabled`, `auto_recovery_cooldown_seconds`, `recovery_cooldown_steps`, `recovery_memory_safety_ratio`, `recovery_max_attempts`

**Affected Files**:
- `src/mlsdm/core/cognitive_controller.py`
- `tests/unit/test_cognitive_controller.py`
- `config/calibration.py`
- `config/default_config.yaml`

---

### ~~REL-002: Add bulkhead pattern for request isolation~~ ✅ COMPLETED

**Block**: Core Reliability  
**Criticality**: ~~HIGH~~ COMPLETED  
**Type**: Code

**Description**: ~~No resource isolation between concurrent requests. A slow request can impact all others.~~

**Solution**: Implemented semaphore-based `BulkheadMiddleware` and `BulkheadSemaphore`:
- **Concurrency limiting**: Max concurrent requests configurable via `MLSDM_MAX_CONCURRENT` (default: 100)
- **Queue timeout**: Requests wait up to `MLSDM_QUEUE_TIMEOUT` (default: 5s) before 503 rejection
- **Prometheus metrics**: Queue depth, active requests, rejected count exported to `/metrics`
- **Graceful degradation**: Returns 503 with `Retry-After` header when capacity exceeded

**Acceptance Criteria**:
- ✅ Implement semaphore-based concurrency limiting
- ✅ Configure max concurrent requests per endpoint
- ✅ Add metrics for queue depth

**Implementation**:
- `BulkheadSemaphore`: Core semaphore with metrics tracking
- `BulkheadMiddleware`: FastAPI middleware with Prometheus integration
- Metrics: `mlsdm_bulkhead_queue_depth`, `mlsdm_bulkhead_active_requests`, `mlsdm_bulkhead_rejected_total`, `mlsdm_bulkhead_max_queue_depth`

**Affected Files**:
- `src/mlsdm/api/middleware.py`
- `src/mlsdm/api/app.py`
- `src/mlsdm/observability/metrics.py`

---

### OBS-001: Implement OpenTelemetry distributed tracing

**Block**: Observability  
**Criticality**: HIGH  
**Type**: Code

**Description**: Dependencies (`opentelemetry-api`, `opentelemetry-sdk`) are installed but not integrated. Distributed tracing is critical for debugging production issues.

**Acceptance Criteria**:
- Add span creation in key paths (API handlers, generate, process_event)
- Export traces to configurable backend (Jaeger/OTLP)
- Add trace context propagation
- Document configuration

**Affected Files**:
- `src/mlsdm/observability/tracing.py` (new)
- `src/mlsdm/api/app.py`
- `src/mlsdm/engine/neuro_cognitive_engine.py`

---

### OBS-002: Deploy Alertmanager rules

**Block**: Observability  
**Criticality**: HIGH  
**Type**: Config/Docs

**Description**: Alert rules are defined in `SLO_SPEC.md` but not deployed as actual Alertmanager configuration.

**Acceptance Criteria**:
- Create `deploy/monitoring/alertmanager-rules.yaml`
- Add alerts for: availability breach, latency breach, error budget burn
- Document alert routing configuration

**Affected Files**:
- `deploy/monitoring/alertmanager-rules.yaml` (new)
- `DEPLOYMENT_GUIDE.md`

---

### SEC-001: Implement RBAC for API endpoints

**Block**: Security  
**Criticality**: HIGH  
**Type**: Code

**Description**: Current authentication is binary (authenticated = authorized). Production needs role-based access control.

**Acceptance Criteria**:
- Define roles: `read`, `write`, `admin`
- Add role validation middleware
- Document role assignment process

**Affected Files**:
- `src/mlsdm/security/rbac.py` (new)
- `src/mlsdm/api/app.py`
- `SECURITY_POLICY.md`

---

### SEC-002: Add automated secret rotation support

**Block**: Security  
**Criticality**: HIGH  
**Type**: Code/Docs

**Description**: API keys are static. Production needs mechanism for rotation without downtime.

**Acceptance Criteria**:
- Support multiple valid API keys simultaneously during rotation
- Document rotation procedure
- Add key expiration logging

**Affected Files**:
- `src/mlsdm/api/app.py`
- `RUNBOOK.md`

---

### ~~SEC-003: Add dependency vulnerability scanning to PR workflow~~ ✅ COMPLETED

**Block**: Security  
**Criticality**: ~~HIGH~~ COMPLETED  
**Type**: CI

**Description**: ~~Trivy scan only runs on release, not on PRs. Vulnerabilities can be introduced and merged.~~ Added pip-audit security scanning to CI workflow.

**Acceptance Criteria**:
- ✅ Add `pip-audit` or `safety` check to PR CI
- ✅ Fail PR if high/critical vulnerabilities found
- ✅ Document exception process (fails build with clear error message)

**Affected Files**:
- `.github/workflows/ci-neuro-cognitive-engine.yml`

---

### CICD-002: Add required status checks on main branch

**Block**: CI/CD  
**Criticality**: HIGH  
**Type**: Config

**Description**: No branch protection rules enforcing CI checks before merge. PRs can be merged without passing tests.

**Acceptance Criteria**:
- Enable branch protection on `main`
- Require status checks: test, lint, type-check
- Require at least 1 approval (optional)

**Affected Files**:
- GitHub repository settings (manual)
- Document in `CONTRIBUTING.md`

---

### CICD-003: Separate smoke tests from slow tests

**Block**: CI/CD  
**Criticality**: HIGH  
**Type**: CI

**Description**: All tests run together. Fast feedback loop is lost when running full test suite.

**Acceptance Criteria**:
- Create `ci-smoke.yml` for fast unit tests only (<2 min)
- Keep full tests in existing workflow
- Add `@pytest.mark.slow` to integration/property tests
- Run smoke on all pushes, full on PRs to main

**Affected Files**:
- `.github/workflows/ci-smoke.yml` (new)
- `tests/` (add markers)

---

### CICD-004: Add SAST scanning to PR workflow

**Block**: CI/CD  
**Criticality**: HIGH  
**Type**: CI

**Description**: No static application security testing (SAST) in CI. CodeQL or bandit should scan for security issues.

**Acceptance Criteria**:
- Add CodeQL or bandit scan to PR workflow
- Fail on high-severity findings
- Document false positive handling

**Affected Files**:
- `.github/workflows/ci-neuro-cognitive-engine.yml` or new workflow

---

### CICD-005: Add production deployment gate workflow

**Block**: CI/CD  
**Criticality**: HIGH  
**Type**: CI

**Description**: No explicit production gate. Release workflow doesn't verify all production criteria.

**Acceptance Criteria**:
- Create `prod-gate.yml` that runs all pre-release checks
- Block release if any check fails
- Include manual approval step

**Affected Files**:
- `.github/workflows/prod-gate.yml` (new)
- `.github/workflows/release.yml` (add dependency)

---

### PERF-001: Implement SLO-based release gates

**Block**: Performance  
**Criticality**: HIGH  
**Type**: CI

**Description**: SLOs are defined but not enforced in CI. Regressions can be released without detection.

**Acceptance Criteria**:
- Add benchmark assertions to CI
- Fail if P95 latency exceeds SLO
- Store benchmark results as artifacts

**Affected Files**:
- `.github/workflows/ci-neuro-cognitive-engine.yml`
- `benchmarks/test_neuro_engine_performance.py`

---

### ~~DOC-001: Create Architecture Decision Records (ADRs)~~ ✅ COMPLETED

**Block**: Docs  
**Criticality**: ~~HIGH~~ COMPLETED  
**Type**: Docs

**Description**: ~~No documented rationale for key architecture decisions. Makes it hard for new contributors to understand design choices.~~ Created ADR directory with template and initial ADRs.

**Acceptance Criteria**:
- ✅ Create `docs/adr/` directory
- ✅ Add ADRs for: PELM design, moral filter algorithm, memory bounds
- ✅ Template for future ADRs

**Affected Files**:
- `docs/adr/0000-adr-template.md` (new)
- `docs/adr/0001-use-adrs.md` (new)
- `docs/adr/0002-pelm-design.md` (new)
- `docs/adr/0003-moral-filter.md` (new)
- `docs/adr/0004-memory-bounds.md` (new)

---

## MEDIUM Priority

### ~~REL-003: Add chaos engineering tests to CI~~ ✅ COMPLETED

**Block**: Core Reliability  
**Criticality**: ~~MEDIUM~~ COMPLETED  
**Type**: Tests

**Description**: ~~No automated failure injection tests. System resilience not verified continuously.~~

**Solution**: Implemented comprehensive chaos engineering test suite:
- **Memory Pressure** (`test_memory_pressure.py`): 5 tests for emergency shutdown, recovery, and graceful degradation
- **Slow LLM** (`test_slow_llm.py`): 5 tests for timeout handling, concurrent slow requests, degrading performance
- **Network Timeout** (`test_network_timeout.py`): 7 tests for connection errors, gradual degradation, failure patterns
- **CI Workflow**: Scheduled daily at 03:00 UTC via `.github/workflows/chaos-tests.yml`
- **Test Marker**: All chaos tests marked with `@pytest.mark.chaos`

**Acceptance Criteria**:
- ✅ Add tests that inject: memory pressure, slow LLM, network timeouts
- ✅ Verify graceful degradation
- ✅ Run in scheduled CI (not every PR)

**Implementation**:
- `create_slow_llm()`, `create_failing_llm()`, `create_timeout_llm()`: Fault injection helpers
- Tests verify: no panics, expected errors returned, system recovers
- CI produces JUnit XML artifacts for test reporting

**Affected Files**:
- `tests/chaos/test_memory_pressure.py`
- `tests/chaos/test_slow_llm.py`
- `tests/chaos/test_network_timeout.py`
- `.github/workflows/chaos-tests.yml`
- `TESTING_STRATEGY.md`

---

### ~~REL-004: Add request timeout middleware~~ ✅ COMPLETED

**Block**: Core Reliability  
**Criticality**: ~~MEDIUM~~ COMPLETED  
**Type**: Code

**Description**: ~~No explicit request-level timeout in API layer. Long requests can block workers.~~

**Solution**: Implemented `TimeoutMiddleware` in FastAPI middleware stack:
- **Configurable timeout**: Via `MLSDM_REQUEST_TIMEOUT` env var or `api.request_timeout_seconds` config (default: 30s)
- **504 response**: Returns HTTP 504 Gateway Timeout with structured error JSON
- **Excluded paths**: Health endpoints (`/health`, `/health/live`, `/health/ready`) bypass timeout
- **Logging**: Logs timeout events with path, method, elapsed time, request_id
- **Response header**: `X-Request-Timeout` indicates configured timeout value

**Acceptance Criteria**:
- ✅ Add configurable request timeout middleware
- ✅ Return 504 on timeout
- ✅ Log timeout events

**Implementation**:
- `TimeoutMiddleware`: Uses `asyncio.wait_for()` for async timeout
- Error response: `{"error": {"error_code": "E902", "message": "Request timed out"}}`

**Affected Files**:
- `src/mlsdm/api/middleware.py`
- `config/default_config.yaml`

---

### OBS-003: Create Grafana dashboard templates

**Block**: Observability  
**Criticality**: MEDIUM  
**Type**: Config

**Description**: Prometheus metrics exist but no dashboards provided for visualization.

**Acceptance Criteria**:
- Create JSON dashboard for: latency, throughput, error rate, memory
- Add SLO compliance panel
- Document import process

**Affected Files**:
- `deploy/monitoring/grafana-dashboard.json` (new)
- `DEPLOYMENT_GUIDE.md`

---

### OBS-004: Add structured error logging with error codes

**Block**: Observability  
**Criticality**: MEDIUM  
**Type**: Code

**Description**: Errors logged as strings. Need structured error codes for automated alerting.

**Acceptance Criteria**:
- Define error code enum (E001, E002, etc.)
- Add error code to all error logs
- Document error code meanings

**Affected Files**:
- `src/mlsdm/utils/errors.py` (new)
- `src/mlsdm/observability/logger.py`

---

### OBS-005: Add log aggregation configuration examples

**Block**: Observability  
**Criticality**: MEDIUM  
**Type**: Docs

**Description**: No documentation for setting up log aggregation (ELK/Loki).

**Acceptance Criteria**:
- Add Loki config example to `deploy/`
- Document log shipping setup
- Add FluentBit sidecar example

**Affected Files**:
- `deploy/monitoring/loki-config.yaml` (new)
- `DEPLOYMENT_GUIDE.md`

---

### SEC-004: Add OAuth 2.0 / OIDC support

**Block**: Security  
**Criticality**: MEDIUM  
**Type**: Code

**Description**: Only API key auth supported. Enterprise deployments need OAuth/OIDC.

**Acceptance Criteria**:
- Add optional OIDC provider integration
- Support JWT validation
- Document configuration

**Affected Files**:
- `src/mlsdm/security/oidc.py` (new)
- `src/mlsdm/api/app.py`
- `SECURITY_POLICY.md`

---

### SEC-005: Generate SBOM on release

**Block**: Security  
**Criticality**: MEDIUM  
**Type**: CI

**Description**: No Software Bill of Materials generated. Required for supply chain security.

**Acceptance Criteria**:
- Add syft or cyclonedx-bom to release workflow
- Attach SBOM to GitHub release
- Document SBOM usage

**Affected Files**:
- `.github/workflows/release.yml`

---

### PERF-002: Add continuous benchmark tracking

**Block**: Performance  
**Criticality**: MEDIUM  
**Type**: CI

**Description**: Benchmarks run but results not tracked over time. Can't detect gradual regression.

**Acceptance Criteria**:
- Store benchmark results as workflow artifacts
- Compare with previous run
- Alert on significant regression (>20%)

**Affected Files**:
- `.github/workflows/ci-neuro-cognitive-engine.yml`

---

### PERF-003: Add error budget tracking dashboard

**Block**: Performance  
**Criticality**: MEDIUM  
**Type**: Config

**Description**: Error budget defined in SLO_SPEC but not tracked.

**Acceptance Criteria**:
- Add error budget calculation to metrics
- Create dashboard panel for burn rate
- Document budget policy

**Affected Files**:
- `deploy/monitoring/grafana-dashboard.json`
- `SLO_SPEC.md`

---

### CICD-006: Add container image signing

**Block**: CI/CD  
**Criticality**: MEDIUM  
**Type**: CI

**Description**: Docker images not signed. Can't verify image integrity.

**Acceptance Criteria**:
- Add cosign to release workflow
- Sign images with GitHub Actions OIDC
- Document verification

**Affected Files**:
- `.github/workflows/release.yml`

---

### CICD-007: Add canary deployment workflow

**Block**: CI/CD  
**Criticality**: MEDIUM  
**Type**: CI

**Description**: No canary or blue-green deployment support. All-or-nothing releases are risky.

**Acceptance Criteria**:
- Add canary deployment K8s manifests
- Add traffic splitting configuration
- Document rollback procedure

**Affected Files**:
- `deploy/k8s/canary-deployment.yaml` (new)
- `DEPLOYMENT_GUIDE.md`

---

### DOC-002: Add API versioning documentation

**Block**: Docs  
**Criticality**: MEDIUM  
**Type**: Docs

**Description**: No documented API versioning strategy or breaking change policy.

**Acceptance Criteria**:
- Document version header usage
- Define breaking change criteria
- Add deprecation timeline policy

**Affected Files**:
- `API_REFERENCE.md`

---

### DOC-003: Auto-generate OpenAPI spec

**Block**: Docs  
**Criticality**: MEDIUM  
**Type**: CI

**Description**: FastAPI generates OpenAPI at runtime but not exported as static file.

**Acceptance Criteria**:
- Add script to export openapi.json
- Commit to repo or generate in CI
- Add to documentation

**Affected Files**:
- `scripts/export_openapi.py` (new)
- `docs/openapi.json` (generated)

---

## LOW Priority

### ~~REL-005: Add request prioritization~~ ✅ COMPLETED

**Block**: Core Reliability  
**Criticality**: ~~LOW~~ COMPLETED  
**Type**: Code

**Description**: ~~All requests treated equally. Production may need priority lanes.~~

**Solution**: Implemented `PriorityMiddleware` for request prioritization:
- **Header**: `X-MLSDM-Priority: high|normal|low` (or numeric 1-10)
- **Weights**: high=3, normal=2, low=1 (higher processed first under load)
- **Request state**: Priority stored in `request.state.priority` and `request.state.priority_weight`
- **Response header**: `X-MLSDM-Priority-Applied` confirms applied priority
- **Integration**: Works with BulkheadMiddleware to prioritize during resource contention
- **Documentation**: API_REFERENCE.md and USAGE_GUIDE.md updated with examples

**Acceptance Criteria**:
- ✅ Add priority header support (X-MLSDM-Priority: high|normal|low)
- ✅ Implement priority queue
- ✅ Document usage

**Implementation**:
- `RequestPriority`: Enum-like class with weights and header parsing
- `PriorityQueueItem`: Dataclass for priority queue ordering (higher weight = processed first)
- `PriorityMiddleware`: Parses header, stores in request state, logs high-priority requests

**Affected Files**:
- `src/mlsdm/api/middleware.py`
- `API_REFERENCE.md`
- `USAGE_GUIDE.md`

---

### OBS-006: Add business metrics

**Block**: Observability  
**Criticality**: LOW  
**Type**: Code

**Description**: Only technical metrics tracked. No business-level metrics (events by type, etc.).

**Acceptance Criteria**:
- Add custom metric registration API
- Document metric creation pattern
- Add example business metrics

---

### SEC-006: Add mTLS support

**Block**: Security  
**Criticality**: LOW  
**Type**: Code

**Description**: Only server-side TLS. Some enterprises require mutual TLS.

**Acceptance Criteria**:
- Add client certificate validation option
- Document CA configuration
- Add tests

---

### SEC-007: Add request signing verification

**Block**: Security  
**Criticality**: LOW  
**Type**: Code

**Description**: No request signature verification. May be needed for high-security environments.

**Acceptance Criteria**:
- Add optional HMAC signature verification
- Document signing protocol
- Add client SDK support

---

### PERF-004: Add caching layer

**Block**: Performance  
**Criticality**: LOW  
**Type**: Code

**Description**: No caching for repeated queries. May improve performance for common patterns.

**Acceptance Criteria**:
- Add optional Redis/in-memory cache
- Cache embedding results
- Add cache hit metrics

---

### CICD-008: Add changelog automation

**Block**: CI/CD  
**Criticality**: LOW  
**Type**: CI

**Description**: CHANGELOG manually maintained. Could be automated from commit messages.

**Acceptance Criteria**:
- Add conventional commits enforcement
- Auto-generate changelog on release
- Document commit format

---

### DOC-004: Add interactive API playground

**Block**: Docs  
**Criticality**: LOW  
**Type**: Docs

**Description**: Swagger UI available at /docs but no curated examples.

**Acceptance Criteria**:
- Add example requests to OpenAPI spec
- Create tutorial notebook
- Document common use cases

---

### DOC-005: Add troubleshooting decision tree

**Block**: Docs  
**Criticality**: LOW  
**Type**: Docs

**Description**: RUNBOOK has troubleshooting but no decision tree for quick diagnosis.

**Acceptance Criteria**:
- Create visual decision tree
- Add to RUNBOOK
- Cover top 10 issues

---

## Completed

_Track completed items here:_

| ID | Description | Completed Date | PR |
|----|-------------|----------------|-----|
| CICD-001 | Add linting and type checking to CI workflows | 2025-11-27 | #124 |
| SEC-003 | Add dependency vulnerability scanning to PR workflow | 2025-11-27 | #124 |
| DOC-001 | Create Architecture Decision Records (ADRs) | 2025-11-30 | #157 |
| REL-001 | Implement automated health-based recovery | 2025-12-03 | #185 |
| REL-002 | Add bulkhead pattern for request isolation | 2025-12-03 | #185 |
| REL-003 | Add chaos engineering tests to CI | 2025-12-03 | #185 |
| REL-004 | Add request timeout middleware | 2025-12-03 | #185 |
| REL-005 | Add request prioritization | 2025-12-03 | #185 |
