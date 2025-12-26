# Evidence Snapshots

## Policy

This directory contains committed evidence snapshots for reproducibility and auditability.

### Structure

```
evidence/
  YYYY-MM-DD/<git_sha>/
    manifest.json          # Metadata (timestamp, sha, python, platform, etc.)
    coverage/
      coverage.xml         # Coverage report
      coverage.log         # Optional: coverage gate stdout
    pytest/
      junit.xml            # JUnit test results (unit + state)
    benchmarks/
      benchmark-metrics.json      # Schema expected by check_benchmark_drift.py
      raw_neuro_engine_latency.json  # Optional: raw per-scenario percentiles
    memory/
      memory_footprint.json       # PELM + controller memory metrics
    env/
      python_version.txt          # Python version
      uname.txt                   # OS info
      uv_lock_sha256.txt          # SHA256 of uv.lock for reproducibility
```

### Rules

1. **Small files only** — Keep evidence compact; no large dumps or binary blobs.
2. **No secrets** — Never commit credentials, tokens, .env files, or API keys.
3. **Dated folders** — Each snapshot lives in a dated subfolder; do NOT overwrite previous snapshots.
4. **Reproducible** — Evidence is regenerated via `make evidence` using `scripts/evidence/capture_evidence.py`.
5. **Read-only archive** — Treat committed snapshots as immutable historical records.

### Regenerating Evidence

```bash
make evidence
```

This runs `uv run python scripts/evidence/capture_evidence.py` which:
- Creates a new dated folder under `artifacts/evidence/`
- Runs coverage gate and captures `coverage.xml`
- Runs unit/state tests with JUnit output
- Computes benchmark metrics
- Measures memory footprint
- Records environment metadata

### Retention

Snapshots are kept for historical reference. To clean up old snapshots, manually delete
dated folders that are no longer needed for audit trails.
