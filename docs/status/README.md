# Readiness Policy

- **What it is:** `docs/status/READINESS.md` is the single source of truth for the current, evidence-backed system status. Any conflicting claim elsewhere is superseded by that file.
- **When to update:** Update `docs/status/READINESS.md` whenever changes touch `src/`, `tests/`, `config/`, `deploy/`, any `Dockerfile*`, or `.github/workflows/` (defaults mirrored in `scripts/readiness_check.py`, overridable via `READINESS_SCOPED_PREFIXES`). Include dated evidence (tests, CI jobs, commands) for any status shift.
- **Why CI enforces it:** `.github/workflows/readiness.yml` runs `scripts/readiness_check.py` to ensure the file exists, is fresh (≤14 days), and is updated when scoped code or workflow changes occur. The workflow fails the build with a clear `::error::` message if the readiness record is stale or missing.
- **Authority:** READINESS.md is authoritative for auditors and reviewers. If evidence is missing, mark the relevant area as NOT VERIFIED or PARTIAL rather than inferring readiness.

## Running readiness automation locally

1. Collect change list and analysis:
   ```bash
   git diff --name-only origin/main..HEAD > /tmp/paths.txt
   python scripts/readiness/change_analyzer.py --files /tmp/paths.txt --base-ref origin/main --output /tmp/change.json
   ```
2. Collect evidence (artifacts optional; missing sources are tolerated but marked measured=false):
   ```bash
   python scripts/readiness/evidence_collector.py --output /tmp/evidence.json
   ```
3. Evaluate policy:
   ```bash
   python scripts/readiness/policy_engine.py --change-analysis /tmp/change.json --evidence /tmp/evidence.json --output /tmp/policy.json
   ```
4. Generate changelog entry:
   ```bash
   python scripts/readiness/changelog_generator.py --title "Local run" --base-ref origin/main --mode preview
   ```
   Use `--mode apply` to update `docs/status/READINESS.md`; `--mode dry-run` behaves like preview.

## CI behaviour

- `.github/workflows/readiness.yml` computes changes, collects evidence, evaluates policy, and updates `docs/status/READINESS.md`.
- Auto-commit is gated by the `readiness-auto-apply` label on the PR and only runs for same-repo PRs; without the label, the workflow stays in preview mode and posts a summary comment only.
- Evidence sources are expected under `reports/` (e.g., `junit-*.xml`, `coverage.xml`, `bandit.json`, `semgrep.json`, `gitleaks.json`). Missing artifacts do not fail the run but reduce confidence and can produce `approve_with_conditions` or `reject` verdicts when required for rules.
