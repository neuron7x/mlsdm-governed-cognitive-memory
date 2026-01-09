# Debt Prevention Policy (Future-Drift Guardrails)

**Scope:** This policy is enforceable, CI-blocking, and applies to all branches.
**Canonical thresholds:** `policy/debt_prevention_thresholds.json`.

## Future-Debt Threat Map (V1–V9)
| Vector | Invariant (must always be true) | Verifier | CI Gate | Evidence Artifact |
|---|---|---|---|---|
| V1 Dependency drift & non-determinism | `uv.lock` is canonical and installs are frozen | `make determinism-check` | `debt-prevention` job | `artifacts/evidence/.../env/uv_lock_sha256.txt` |
| V2 CI drift / flaky tests | CI gates run from Make targets | `make determinism-check`, `make test-hygiene` | `debt-prevention` job | `artifacts/tmp/policy-drift.json` |
| V3 Evidence gaps | Evidence pack contains coverage, junit, audit, hashes, env, summary | `make verify-evidence` | `release` + evidence job | `artifacts/evidence/.../manifest.json` |
| V4 Security drift | High/Critical audit findings fail | `make security-audit` | `debt-prevention` job | `artifacts/security/pip-audit.json` |
| V5 Governance drift | Threshold changes require approval token | `make policy-drift-check` | `debt-prevention` job | `policy/policy_drift_approval.json` |
| V6 Config ambiguity | Canonical policy file is source of truth | `policy/debt_prevention_thresholds.json` | `debt-prevention` job | `policy/debt_prevention_thresholds.json` |
| V7 Observability erosion | Structured logging (no f-strings) | `make log-hygiene` | `debt-prevention` job | `artifacts/tmp/ci-summary.json` |
| V8 Docs drift | Duplicate headings banned per file | `make docs-lint` | `debt-prevention` job | `artifacts/tmp/ci-summary.json` |
| V9 Release hygiene | Release tag must emit evidence pack | `make evidence` + `make verify-evidence` | `release` workflow | `artifacts/evidence/...` |

## Non-Negotiable Invariants (I1–I8)
1. **I1 Deterministic installs:** `uv.lock` is canonical; CI runs `make determinism-check`.
2. **I2 No generated-file drift:** `requirements.txt` regenerated via `python scripts/ci/export_requirements.py --check`.
3. **I3 Evidence-first releases:** release tags must produce a verified evidence pack.
4. **I4 Security baseline:** `make security-audit` fails on HIGH/CRITICAL findings.
5. **I5 Policy drift detection:** changes to thresholds require `policy/policy_drift_approval.json` update.
6. **I6 Test integrity:** skips/xfails require a reason with issue link (`make test-hygiene`).
7. **I7 Observability hygiene:** f-string logging is forbidden (`make log-hygiene`).
8. **I8 Docs single-source:** duplicate headings per file are blocked (`make docs-lint`).

## Enforcement Locations
- CI gate: `.github/workflows/ci-neuro-cognitive-engine.yml` → `debt-prevention` job.
- Release gate: `.github/workflows/release.yml` → `evidence-pack` job.
- Evidence verifier: `scripts/evidence/verify_evidence_snapshot.py`.
