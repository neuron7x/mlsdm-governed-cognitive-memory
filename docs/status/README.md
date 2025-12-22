# Readiness Policy

- **What it is:** `docs/status/READINESS.md` is the single source of truth for the current, evidence-backed system status. Any conflicting claim elsewhere is superseded by that file.
- **When to update:** Update `docs/status/READINESS.md` whenever changes touch `src/`, `tests/`, `config/`, `deploy/`, any `Dockerfile*`, or `.github/workflows/`. Include dated evidence (tests, CI jobs, commands) for any status shift.
- **Why CI enforces it:** `.github/workflows/readiness.yml` runs `scripts/readiness_check.py` to ensure the file exists, is fresh (≤14 days), and is updated when scoped code or workflow changes occur. The workflow fails the build with a clear `::error::` message if the readiness record is stale or missing.
- **Authority:** READINESS.md is authoritative for auditors and reviewers. If evidence is missing, mark the relevant area as NOT VERIFIED or PARTIAL rather than inferring readiness.
