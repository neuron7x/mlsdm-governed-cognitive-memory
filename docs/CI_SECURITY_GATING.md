# CI Security Gating Policy

**Document Version:** 1.0.0  
**Last Updated:** December 13, 2025  
**Security Level:** Production-Grade

## Overview

This document defines the security gating policy for MLSDM's CI/CD pipeline. Security gates are automated checks that **MUST PASS** before code can be merged or released. This policy implements a zero-tolerance approach for critical security issues.

## Security Gate Philosophy

**Principle:** Security checks are not informational—they are blocking gates.

- ✅ **PASS** → Code can proceed to merge/release
- ❌ **FAIL** → Code is blocked, must be fixed
- 🚫 **NO BYPASS** → Security gates cannot use `continue-on-error` for critical checks

## Critical Security Gates

These jobs **BLOCK** merges and releases if they fail:

### Summary Table

| Tool | Workflow | Job/Step | Gate Type | Status |
|------|----------|----------|-----------|--------|
| **Bandit (HIGH)** | `sast-scan.yml` | `bandit` / Check for high severity issues | BLOCKING | ✅ No `continue-on-error` |
| **Semgrep** | `sast-scan.yml` | `semgrep` / Run Semgrep | BLOCKING | ✅ No `continue-on-error` |
| **pip-audit** | `ci-neuro-cognitive-engine.yml` | `security` / Run pip-audit | BLOCKING | ✅ No `continue-on-error` |
| **pip-audit** | `prod-gate.yml` | `preflight` / Security vulnerability scan | BLOCKING | ✅ No `continue-on-error` |
| **Trivy** | `release.yml` | `security-scan` / Run Trivy | BLOCKING | ✅ No `continue-on-error` |

### 1. SAST (Static Application Security Testing)

**Workflow:** `.github/workflows/sast-scan.yml`

#### Bandit Security Scan (CRITICAL GATE)
- **Job:** `bandit`
- **Tool:** Bandit SAST scanner
- **Threshold:** High severity and high confidence
- **Blocks on:** Any high severity security issues
- **Exit behavior:** Non-zero exit code on high severity findings
- **Status:** ✅ BLOCKING (no `continue-on-error`)
- **Note:** Medium+ findings are generated for SARIF tracking but only high severity blocks CI

#### Semgrep SAST Scan (CRITICAL GATE)
- **Job:** `semgrep`
- **Tool:** Semgrep with security rulesets
- **Rulesets:** `p/python`, `p/security-audit`, `p/owasp-top-ten`
- **Blocks on:** Any security finding from configured rulesets
- **Exit behavior:** semgrep-action exits with non-zero code when findings are detected
- **Status:** ✅ BLOCKING (no `continue-on-error`)
- **Note:** Semgrep blocking behavior is default for semgrep-action@v1

### 2. Dependency Vulnerability Scanning (CRITICAL GATE)

**Workflow:** `.github/workflows/ci-neuro-cognitive-engine.yml`

#### pip-audit Dependency Scan
- **Job:** `security`
- **Tool:** pip-audit
- **Threshold:** Any known vulnerability
- **Blocks on:** CVEs in dependencies
- **Exit behavior:** Non-zero exit code on vulnerabilities
- **Status:** ✅ BLOCKING (no `continue-on-error`)

**Production Gate Workflow:** `.github/workflows/prod-gate.yml`
- Same pip-audit check with `--strict` flag
- Part of pre-flight checks before production deployment

### 3. Container Image Vulnerability Scanning (CRITICAL GATE)

**Workflow:** `.github/workflows/release.yml`

#### Trivy Container Scan
- **Job:** `security-scan`
- **Tool:** Trivy vulnerability scanner
- **Threshold:** CRITICAL and HIGH severity
- **Blocks on:** Critical or high vulnerabilities in container image
- **Exit behavior:** Non-zero exit code (`exit-code: '1'`)
- **Status:** ✅ BLOCKING (no `continue-on-error`)

## Informational Checks

These checks run but **DO NOT BLOCK** CI (with justification):

### Summary Table

| Check | Workflow | Job/Step | Justification | Status |
|-------|----------|----------|---------------|--------|
| **Chaos Tests** | `chaos-tests.yml` | All chaos test steps | Exploratory resilience testing | ✅ `continue-on-error: true` |
| **Cognitive Safety Eval** | `ci-neuro-cognitive-engine.yml` | `neuro-engine-eval` | Research-grade validation | ✅ `continue-on-error: true` |
| **Nightly Comprehensive Tests** | `perf-resilience.yml` | `resilience-comprehensive` | Long-running monitoring | ✅ `continue-on-error: true` |
| **TestPyPI Publishing** | `release.yml` | `publish-to-testpypi` | Optional publishing step | ✅ `continue-on-error: true` |
| **SARIF Uploads** | All workflows | Upload SARIF steps | Reporting after scan completes | ✅ `continue-on-error: true` |

### 1. Chaos Engineering Tests
**Workflow:** `.github/workflows/chaos-tests.yml`

**Justification:** Chaos tests validate resilience under extreme conditions. They are exploratory and test graceful degradation, not correctness. Failures indicate areas for improvement, not blocking defects.

- Memory pressure tests
- Slow LLM response tests
- Network timeout tests

**Status:** 🔵 INFORMATIONAL (`continue-on-error: true`)

### 2. Cognitive Safety Evaluation
**Workflow:** `.github/workflows/ci-neuro-cognitive-engine.yml`

**Justification:** Sapolsky evaluation suite is research-grade and validates cognitive safety properties. It tests experimental features and AI behavior patterns that are important for research but not blocking for core functionality.

- Job: `neuro-engine-eval`
- Tool: Custom Sapolsky Validation Suite

**Status:** 🔵 INFORMATIONAL (`continue-on-error: true`)

### 3. Comprehensive Performance Tests (Nightly)
**Workflow:** `.github/workflows/perf-resilience.yml`

**Justification:** Long-running performance and resilience tests that run nightly or on-demand. These are for monitoring trends and do not block PRs. Fast performance tests with SLO validation are blocking.

- Comprehensive resilience tests
- Comprehensive performance tests

**Status:** 🔵 INFORMATIONAL (`continue-on-error: true`)

### 4. TestPyPI Publishing
**Workflow:** `.github/workflows/release.yml`

**Justification:** Publishing to TestPyPI is optional and depends on credentials. The package has already passed all security gates before this step.

**Status:** 🔵 INFORMATIONAL (`continue-on-error: true`)

### 5. SARIF Upload Steps
**All workflows with SARIF uploads**

**Justification:** Uploading scan results to GitHub Security tab is for visibility and tracking. Upload failures should not block CI—the actual security scan has already passed or failed.

**Status:** 🔵 INFORMATIONAL (`continue-on-error: true`)

## Security Gate Enforcement

### How Gates Work

1. **Security job fails** → Workflow status = FAILED
2. **Downstream jobs blocked** → Jobs with `needs: [security-job]` won't run
3. **PR merge blocked** → GitHub branch protection prevents merge
4. **Release blocked** → Release workflow fails before deployment

### Gate Dependencies

```yaml
# Example: ci-neuro-cognitive-engine.yml
all-ci-passed:
  needs: [lint, security, test, coverage, e2e-tests, effectiveness-validation, benchmarks]
  # Does NOT include neuro-engine-eval (informational)
```

```yaml
# Example: prod-gate.yml
gate-passed:
  needs: [preflight, property-tests, slo-validation, security-scan, docs-validation, approval]
  # All security checks must pass
```

## Local Development Commands

Developers **MUST** run these commands before pushing:

### Security Checks
```bash
# 1. Run SAST with Bandit (HIGH severity - matches CI gate)
# Path src/mlsdm is the main Python package directory
# Exit code: 0 = no issues, 1 = issues found
bandit -r src/mlsdm --severity-level high --confidence-level high

# 2. Run dependency audit (requires requirements.txt installed)
# Exit code: 0 = no vulnerabilities, 1 = vulnerabilities found
pip-audit --requirement requirements.txt --strict

# 3. Run all tests (includes security tests)
pytest --ignore=tests/load

# 4. Run linting and type checking
ruff check src tests
mypy src/mlsdm
```

### Quick Security Validation
```bash
# Run all security checks before pushing (exits on first failure):
bandit -r src/mlsdm --severity-level high --confidence-level high && \
  pip-audit --requirement requirements.txt --strict && \
  pytest tests/security/ -v
```

## Incident Response

### What to do when a security gate fails:

1. **DO NOT** add `continue-on-error` to bypass the gate
2. **DO NOT** commit with `--no-verify` to skip pre-commit hooks
3. **DO** investigate the security finding
4. **DO** fix the root cause
5. **DO** verify the fix locally before pushing

### False Positive Handling

If a security tool reports a false positive:

1. Document the false positive in code comments
2. Use tool-specific suppression with justification:
   ```python
   # nosec B101 - False positive: assert used in test context only
   assert user.is_authenticated
   ```
3. Do NOT disable the entire security check

## Compliance and Auditing

### Audit Trail

- All security scans upload SARIF results to GitHub Security tab
- Security findings are tracked in GitHub Security Advisories
- Workflow run history provides audit trail

### Metrics

Track these security metrics:
- Security gate failure rate
- Time to fix security issues
- Number of security findings by severity
- False positive rate

## Policy Updates

This policy is versioned and stored in the repository. Changes require:
1. Review by security/DevSecOps team
2. Documentation update
3. Workflow update
4. Communication to development team

## References

- [SECURITY_POLICY.md](../SECURITY_POLICY.md) - Overall security policy
- [CI_GUIDE.md](../CI_GUIDE.md) - CI/CD workflow documentation
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Development guidelines
- [OWASP Secure Coding Practices](https://owasp.org/www-project-secure-coding-practices-quick-reference-guide/)

## Contacts

- **Security Issues:** Report via GitHub Security Advisories
- **CI/CD Questions:** See CI_GUIDE.md
- **Policy Questions:** Open a GitHub Discussion

---

**Remember:** Security gates exist to protect users, not to slow down development. Fix the issue, don't bypass the gate.
