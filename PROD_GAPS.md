# Production Gaps

**Version**: 1.2.0  
**Last Updated**: November 2025  
**Purpose**: Prioritized task list for production-readiness improvements

---

## Summary

| Block | Blockers | High | Medium | Low |
|-------|----------|------|--------|-----|
| Core Reliability | 0 | 2 | 2 | 1 |
| Observability | 0 | 2 | 3 | 1 |
| Security | 0 | 2 | 2 | 2 |
| Performance | 0 | 1 | 2 | 1 |
| CI/CD | 0 | 4 | 2 | 1 |
| Docs | 0 | 1 | 2 | 2 |
| **Total** | **0** | **12** | **13** | **8** |

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

### REL-001: Implement automated health-based recovery

**Block**: Core Reliability  
**Criticality**: HIGH  
**Type**: Code

**Description**: After `emergency_shutdown` is triggered due to memory threshold, manual intervention is required via `reset_emergency_shutdown()`. Production deployments need automated recovery.

**Acceptance Criteria**:
- Add optional auto-recovery after configurable cooldown period
- Log recovery events
- Add tests for recovery behavior

**Affected Files**:
- `src/mlsdm/core/cognitive_controller.py`
- `tests/unit/test_cognitive_controller.py`

---

### REL-002: Add bulkhead pattern for request isolation

**Block**: Core Reliability  
**Criticality**: HIGH  
**Type**: Code

**Description**: No resource isolation between concurrent requests. A slow request can impact all others.

**Acceptance Criteria**:
- Implement semaphore-based concurrency limiting
- Configure max concurrent requests per endpoint
- Add metrics for queue depth

**Affected Files**:
- `src/mlsdm/api/middleware.py`
- `src/mlsdm/api/app.py`

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

### ~~SEC-001: Implement RBAC for API endpoints~~ ✅ COMPLETED

**Block**: Security  
**Criticality**: ~~HIGH~~ COMPLETED  
**Type**: Code

**Description**: ~~Current authentication is binary (authenticated = authorized). Production needs role-based access control.~~ Added RBAC middleware integration with roles: `read`, `write`, `admin`.

**Acceptance Criteria**:
- ✅ Define roles: `read`, `write`, `admin`
- ✅ Add role validation middleware
- ✅ Document role assignment process

**Affected Files**:
- `src/mlsdm/security/rbac.py` (implemented)
- `src/mlsdm/api/app.py` (integrated)
- `tests/integration/test_rbac_integration.py` (new)

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

### REL-003: Add chaos engineering tests to CI

**Block**: Core Reliability  
**Criticality**: MEDIUM  
**Type**: Tests

**Description**: No automated failure injection tests. System resilience not verified continuously.

**Acceptance Criteria**:
- Add tests that inject: memory pressure, slow LLM, network timeouts
- Verify graceful degradation
- Run in scheduled CI (not every PR)

**Affected Files**:
- `tests/chaos/` (new directory)
- `.github/workflows/chaos-tests.yml` (new)

---

### REL-004: Add request timeout middleware

**Block**: Core Reliability  
**Criticality**: MEDIUM  
**Type**: Code

**Description**: No explicit request-level timeout in API layer. Long requests can block workers.

**Acceptance Criteria**:
- Add configurable request timeout middleware
- Return 504 on timeout
- Log timeout events

**Affected Files**:
- `src/mlsdm/api/middleware.py`

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

### ~~SEC-005: Generate SBOM on release~~ ✅ COMPLETED

**Block**: Security  
**Criticality**: ~~MEDIUM~~ COMPLETED  
**Type**: CI

**Description**: ~~No Software Bill of Materials generated. Required for supply chain security.~~ Added SBOM generation workflow using CycloneDX.

**Acceptance Criteria**:
- ✅ Add syft or cyclonedx-bom to release workflow
- ✅ Attach SBOM to GitHub release
- ✅ Document SBOM usage

**Affected Files**:
- `.github/workflows/sbom.yml` (new)

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

### REL-005: Add request prioritization

**Block**: Core Reliability  
**Criticality**: LOW  
**Type**: Code

**Description**: All requests treated equally. Production may need priority lanes.

**Acceptance Criteria**:
- Add priority header support
- Implement priority queue
- Document usage

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
| SEC-001 | Implement RBAC for API endpoints | 2025-11-30 | - |
| SEC-005 | Generate SBOM on release | 2025-11-30 | - |
