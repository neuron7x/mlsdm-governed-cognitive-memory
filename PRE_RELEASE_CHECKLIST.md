# Pre-Release Checklist

**Version**: 1.2.0  
**Last Updated**: November 2025  
**Purpose**: Verifiable pre-release gate checks with executable commands

---

## Quick Verification (Minimum Gate)

Run these commands to verify basic production readiness:

```bash
# 1. Install dependencies
pip install -e ".[test]"

# 2. Run linting
ruff check src tests

# 3. Run type checking
mypy src/mlsdm

# 4. Run all tests (excluding load tests)
pytest --ignore=tests/load -q

# 5. Verify coverage >= 90%
pytest --ignore=tests/load --cov=src --cov-fail-under=90 -q

# 6. Run security tests
pytest tests/security/ -v

# 7. Run property-based tests
pytest tests/property/ -v

# 8. Run benchmarks
pytest benchmarks/test_neuro_engine_performance.py -v -s
```

---

## Detailed Checklist

### Code Quality

- [ ] **Linting passes with no errors**
  ```bash
  ruff check src tests
  # Expected: No output (all checks pass)
  ```

- [ ] **Type checking passes**
  ```bash
  mypy src/mlsdm
  # Expected: Success: no issues found
  ```

- [ ] **No TODO/FIXME in critical paths**
  ```bash
  grep -r "TODO\|FIXME" src/mlsdm/core/ src/mlsdm/engine/ src/mlsdm/api/ | wc -l
  # Expected: 0 (or documented exceptions)
  ```

### Testing

- [ ] **All unit tests pass**
  ```bash
  pytest tests/unit/ -q
  # Expected: X passed, 0 failed
  ```

- [ ] **All integration tests pass**
  ```bash
  pytest tests/integration/ -q
  # Expected: X passed, 0 failed
  ```

- [ ] **All property tests pass**
  ```bash
  pytest tests/property/ -v --maxfail=3
  # Expected: All invariants verified
  ```

- [ ] **All validation tests pass**
  ```bash
  pytest tests/validation/ -v
  # Expected: Effectiveness criteria met
  ```

- [ ] **All security tests pass**
  ```bash
  pytest tests/security/ -v
  # Expected: All security controls verified
  ```

- [ ] **Coverage >= 90%**
  ```bash
  pytest --ignore=tests/load --cov=src --cov-report=term-missing --cov-fail-under=90
  # Expected: TOTAL coverage >= 90%
  ```

### Performance

- [ ] **Pre-flight latency P95 < 20ms**
  ```bash
  pytest benchmarks/test_neuro_engine_performance.py::test_benchmark_pre_flight_latency -v -s
  # Expected: ✓ SLO met: P95 < 20ms
  ```

- [ ] **End-to-end latency P95 < 500ms**
  ```bash
  pytest benchmarks/test_neuro_engine_performance.py::test_benchmark_end_to_end_small_load -v -s
  # Expected: ✓ SLO met: P95 < 500ms
  ```

- [ ] **Heavy load latency within SLO**
  ```bash
  pytest benchmarks/test_neuro_engine_performance.py::test_benchmark_end_to_end_heavy_load -v -s
  # Expected: ✓ All token counts meet SLO: P95 < 500ms
  ```

### Security

- [ ] **Rate limiting functional**
  ```bash
  pytest tests/security/test_robustness.py -v
  # Expected: Rate limiting tests pass
  ```

- [ ] **Secure mode blocks training**
  ```bash
  pytest tests/security/test_secure_mode.py -v
  # Expected: All secure mode tests pass
  ```

- [ ] **Checkpoint path validation**
  ```bash
  pytest tests/security/test_neurolang_checkpoint_security.py -v
  # Expected: Path restriction enforced
  ```

- [ ] **No secrets in codebase**
  ```bash
  # Note: May have false positives in tests/examples - manual review required
  grep -r "sk-\|api_key\s*=\s*['\"]" src/mlsdm/ --include="*.py" | grep -v "os.getenv\|environ\|test_\|example" | wc -l
  # Expected: 0
  ```

### API & Health

- [ ] **Health endpoints respond correctly**
  ```bash
  # Start server in background and save PID
  uvicorn mlsdm.api.app:app --port 8000 &
  SERVER_PID=$!
  sleep 5
  
  # Test health endpoints
  curl -s http://localhost:8000/health/liveness | grep -q "ok"
  curl -s http://localhost:8000/health/readiness | grep -q "ok"
  curl -s http://localhost:8000/health/metrics | grep -q "mlsdm"
  
  # Stop server using saved PID
  kill $SERVER_PID 2>/dev/null
  ```

- [ ] **API authentication enforced**
  ```bash
  pytest tests/integration/test_neuro_engine_http_api.py -v -k "auth"
  # Expected: Auth tests pass
  ```

### Docker

- [ ] **Docker image builds successfully**
  ```bash
  docker build -f Dockerfile.neuro-engine-service -t mlsdm-test:latest .
  # Expected: Successfully built
  ```

- [ ] **Docker container starts and health check passes**
  ```bash
  docker run -d --name mlsdm-test -p 8080:8000 mlsdm-test:latest
  sleep 10
  curl -s http://localhost:8080/health/liveness | grep -q "ok"
  docker stop mlsdm-test && docker rm mlsdm-test
  # Expected: Health check responds with ok
  ```

### Documentation

- [ ] **README has no broken links (spot check)**
  ```bash
  grep -oE '\[.*\]\(([^)]+)\)' README.md | grep -v "http" | head -10
  # Manually verify internal links exist
  ```

- [ ] **API Reference exists**
  ```bash
  test -f API_REFERENCE.md && echo "API_REFERENCE.md exists"
  # Expected: API_REFERENCE.md exists
  ```

- [ ] **RUNBOOK exists**
  ```bash
  test -f RUNBOOK.md && echo "RUNBOOK.md exists"
  # Expected: RUNBOOK.md exists
  ```

- [ ] **CHANGELOG updated**
  ```bash
  head -20 CHANGELOG.md
  # Expected: Current version documented
  ```

---

## CI Verification

- [ ] **CI workflows pass on main branch**
  - Check GitHub Actions status for `ci-neuro-cognitive-engine.yml`
  - Check GitHub Actions status for `property-tests.yml`

- [ ] **No blocking security advisories**
  - Check GitHub Security tab
  - Review Dependabot alerts

---

## Final Sign-Off

| Check | Verified By | Date |
|-------|-------------|------|
| All tests pass | | |
| Linting clean | | |
| Coverage >= 90% | | |
| Benchmarks pass SLO | | |
| Security tests pass | | |
| Docker builds | | |
| Docs complete | | |

---

## Rollback Criteria

Immediately rollback if after deployment:
- Error rate > 10% for 5+ minutes
- P95 latency > 500ms for 5+ minutes
- More than 50% of pods failing health checks
- OOM kills observed

```bash
# Kubernetes rollback
kubectl rollout undo deployment/mlsdm-api -n mlsdm-production

# Verify rollback
kubectl rollout status deployment/mlsdm-api -n mlsdm-production
```

---

## Notes

_Space for deployment-specific notes:_

```

```
