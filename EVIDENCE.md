# Evidence Pack (PR Evidence)

## Gates Executed
- Determinism: `make determinism-check`
- Security audit: `make security-audit`
- Test hygiene: `make test-hygiene`
- Logging hygiene: `make log-hygiene`
- Docs lint: `make docs-lint`
- Policy drift: `make policy-drift-check`
- Evidence pack: `make evidence` + `make verify-evidence`

## Evidence Artifacts (Paths)
- Evidence pack root: `artifacts/evidence/<date>/<tag>/<sha>/`
- Manifest + hashes: `artifacts/evidence/<date>/<tag>/<sha>/manifest.json`
- Coverage: `artifacts/evidence/<date>/<tag>/<sha>/coverage/coverage.xml`
- JUnit: `artifacts/evidence/<date>/<tag>/<sha>/pytest/junit.xml`
- Audit output: `artifacts/evidence/<date>/<tag>/<sha>/audit/pip-audit.json`
- CI summary: `artifacts/evidence/<date>/<tag>/<sha>/ci/summary.json`

## Local Reproduction
```bash
make test
make determinism-check
make security-audit
make evidence
make verify-evidence
```
