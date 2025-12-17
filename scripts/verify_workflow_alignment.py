"""
Validate that CI workflows use canonical Makefile targets.

This guard prevents workflow drift by ensuring the primary CI workflow
invokes the agreed make targets for linting, typing, testing, coverage,
E2E, effectiveness validation, and benchmarks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

WORKFLOW_PATH = Path(".github/workflows/ci-neuro-cognitive-engine.yml")

# Mapping of job name -> list of required substrings that must appear
# in at least one run step for the job.
EXPECTED_COMMANDS: dict[str, list[str]] = {
    "lint": ["make lint", "make type"],
    "test": ["make test"],
    "coverage": ["make coverage-gate"],
    "coverage-full": ["make coverage-full"],
    "e2e-tests": ["make test-e2e"],
    "effectiveness-validation": ["make test-effectiveness"],
    "benchmarks": ["make bench"],
}


def _collect_run_steps(job: dict) -> Iterable[str]:
    for step in job.get("steps", []):
        run_cmd = step.get("run")
        if run_cmd:
            yield run_cmd


def main() -> None:
    if not WORKFLOW_PATH.exists():
        raise SystemExit(f"Workflow not found: {WORKFLOW_PATH}")

    data = yaml.safe_load(WORKFLOW_PATH.read_text())
    jobs: dict = data.get("jobs", {})

    missing: list[str] = []

    for job_name, required_cmds in EXPECTED_COMMANDS.items():
        job = jobs.get(job_name)
        if not job:
            missing.append(f"Job '{job_name}' is missing")
            continue

        run_steps = list(_collect_run_steps(job))
        for command in required_cmds:
            if not any(command in step for step in run_steps):
                missing.append(f"Job '{job_name}' missing command: {command}")

    if missing:
        formatted = "\n - ".join(missing)
        raise SystemExit(f"CI workflow alignment failed:\n - {formatted}")

    print("✓ CI workflow alignment verified (make targets in use)")


if __name__ == "__main__":
    main()
