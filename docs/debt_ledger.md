# Technical Debt Ledger

**Last Updated:** 2026-01-04 (main branch baseline)
**Related:**
- [TECHNICAL_DEBT_REGISTER.md](TECHNICAL_DEBT_REGISTER.md) - **Єдиний реєстр технічного боргу (Unified Technical Debt Register)**
- [ENGINEERING_DEFICIENCIES_REGISTER.md](ENGINEERING_DEFICIENCIES_REGISTER.md)

---

## Baseline Snapshot (main-only, “here and now”)

- Scope: only `main`, last five runs per key workflow, current coverage, and active alerts/jobs.
- Key workflows (main, last 5 runs):
  - CI Smoke (`ci-smoke.yml`) — ✅ all green.
  - Other key pipelines (CI engine, property tests, SAST/security, coverage badge) — no failing jobs observed on main in latest runs.
- Coverage: **78.62%** overall from the latest available main artifact (2025-12-22); no newer coverage uploads exist since that run (see `reports/coverage/COVERAGE_REPORT_2025-12-22.md`), so this value is carried forward for the 2026-01-04 baseline.
  - Historical DL-003 milestone was recorded at 78.13%; 78.62% is the current baseline.
  - Note: coverage data is 13 days old; rerun the coverage workflow to refresh.
- Security/alerts: no active failing jobs or blocked workflows on main observed in current runs; GitHub security alert API access is restricted in this environment, so use the repo dashboard as the authoritative source.

See [TECHNICAL_DEBT_REGISTER.md](TECHNICAL_DEBT_REGISTER.md) for the **complete unified technical debt register** with all identified issues, classifications, and remediation plans.
See [ENGINEERING_DEFICIENCIES_REGISTER.md](ENGINEERING_DEFICIENCIES_REGISTER.md) for detailed engineering deficiency analysis.

---

## DL-001 (RESOLVED)

- Priority: P3
- Gate: test
- Symptom: RuntimeWarning about overflow encountered in dot during TestMemoryContentSafety::test_extreme_magnitude_vectors.
- Evidence: artifacts/baseline/test.log (numpy/linalg/_linalg.py:2792 RuntimeWarning: overflow encountered in dot, triggered by tests/safety/test_memory_leakage.py::TestMemoryContentSafety::test_extreme_magnitude_vectors).
- Likely root cause: Test inputs use extremely large vectors causing numpy.linalg dot product to overflow.
- Fix applied: Implemented safe_norm() function in src/mlsdm/utils/math_constants.py that uses scaled norm computation to prevent overflow. Updated phase_entangled_lattice_memory.py and multi_level_memory.py to use safe_norm() instead of np.linalg.norm().
- Proof command: source .venv/bin/activate && make test
- Risk: None - safe_norm() produces identical results for normal vectors and handles extreme magnitudes safely.
- Date: 2025-12-15
- Fixed: 2025-12-17
- Owner: @copilot
- Status: resolved
- Next action: None - issue is resolved.

---

## DL-002 (RESOLVED)

- Priority: P4 (Low)
- Gate: type-check
- Symptom: 37 mypy errors when running `mypy src/mlsdm`
- Evidence: mypy output showed errors including:
  - 9x "Class cannot subclass 'BaseHTTPMiddleware' (has type 'Any')"
  - 15x "Untyped decorator makes function untyped"
  - 6x "Returning Any from function"
  - 2x "Library stubs not installed"
- Likely root cause: FastAPI/Starlette typing limitations and missing type stubs
- Fix applied: types-PyYAML and types-requests properly configured in pyproject.toml dev dependencies. mypy configuration correctly handles optional modules.
- Proof command: `pip install -e ".[dev]" && mypy src/mlsdm` → "Success: no issues found in 109 source files"
- Risk: None
- Date: 2025-12-19
- Fixed: 2025-12-19
- Owner: @copilot
- Status: resolved
- Next action: None - issue is resolved.

---

## DL-003 (RESOLVED)

- Priority: P3 (Medium)
- Gate: coverage
- Symptom: Test coverage at 70.85%, below target of 75%
- Evidence: COVERAGE_REPORT_2025.md showed 70.85% overall coverage
- Likely root cause: Insufficient tests for api/, security/, observability/ modules
- Fix applied: Coverage verified at 78.13% (above 75% target) after including state tests. Unit tests passing: 1932 passed, 12 skipped.
- Proof command: `pytest tests/unit/ tests/state/ --cov=src/mlsdm` → "Required test coverage of 75.0% reached. Total coverage: 78.13%"
- Risk: None
- Date: 2025-12-19
- Fixed: 2025-12-19
- Owner: @copilot
- Status: resolved
- Next action: None - issue is resolved.
